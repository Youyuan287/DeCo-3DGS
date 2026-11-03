"""CPU-only view-consensus diagnostic on frozen R0 geometry.

This module deliberately uses center projection only.  It reports proxy
repeatability and incremental descriptor signal; it is not an occlusion-aware
renderer and cannot produce segmentation accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


def project_centers(points, world_to_camera, intrinsics, width, height):
    p = np.asarray(points, dtype=float)
    m = np.asarray(world_to_camera, dtype=float)
    cam = (m[:3, :3] @ p.T + m[:3, 3:4]).T
    z = cam[:, 2]
    uv = np.full((len(p), 2), np.nan, dtype=float)
    good = z > 0
    uv[good, 0] = intrinsics[0, 0] * cam[good, 0] / z[good] + intrinsics[0, 2]
    uv[good, 1] = intrinsics[1, 1] * cam[good, 1] / z[good] + intrinsics[1, 2]
    valid = good & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, valid, z


def vote_distribution(labels, classes):
    x = np.asarray(labels)
    c = np.asarray(classes)
    counts = np.asarray([(x == k).sum() for k in c], dtype=float)
    total = counts.sum()
    return counts / total if total else np.zeros(len(c), dtype=float)


def normalized_entropy(q):
    q = np.asarray(q, dtype=float)
    q = q[q > 0]
    if len(q) <= 1:
        return 0.0
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def js_divergence(p, q):
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def split_half_tv(a, b, classes):
    return 0.5 * float(np.abs(vote_distribution(a, classes) - vote_distribution(b, classes)).sum())


def _ridge_fit_predict(x_train, y_train, x_test, alpha):
    mu = x_train.mean(0); sd = x_train.std(0); sd[sd < 1e-12] = 1.0
    a = (x_train - mu) / sd; b = (x_test - mu) / sd
    aa = np.c_[np.ones(len(a)), a]; bb = np.c_[np.ones(len(b)), b]
    reg = np.eye(aa.shape[1]); reg[0, 0] = 0
    beta = np.linalg.solve(aa.T @ aa + alpha * reg, aa.T @ y_train)
    return bb @ beta


def _r2(y, pred):
    den = float(((y - y.mean()) ** 2).sum())
    return 0.0 if den <= 1e-12 else 1.0 - float(((y - pred) ** 2).sum()) / den


def ridge_delta_r2(x0, x1, y, folds=5, alpha=1.0):
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float); y = np.asarray(y, float)
    if len(y) < max(5, folds): return 0.0
    idx = np.array_split(np.arange(len(y)), folds)
    p0 = np.zeros(len(y)); p1 = np.zeros(len(y))
    for test in idx:
        train = np.setdiff1d(np.arange(len(y)), test)
        p0[test] = _ridge_fit_predict(x0[train], y[train], x0[test], alpha)
        p1[test] = _ridge_fit_predict(x1[train], y[train], x1[test], alpha)
    return _r2(y, p1) - _r2(y, p0)


def _read_ply_xyz(path):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line: raise ValueError('PLY header incomplete')
            text = line.decode('ascii', errors='strict').strip()
            header.append(text)
            if text == 'end_header': break
        n = next(int(x.split()[2]) for x in header if x.startswith('element vertex'))
        fmt = '<fffBBB'
        item = struct.Struct(fmt)
        return np.fromiter((v for _ in range(n) for v in item.unpack(f.read(item.size))[:3]), dtype=np.float32).reshape(n, 3)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def _canonical_labels(seg, feat, node_feat):
    """Map existing per-view segment features to fixed NAG labels."""
    seg = np.asarray(seg)
    out = np.full(seg.shape, -1, dtype=np.int32)
    flat = seg.ravel()
    good = (flat >= 0) & (flat < len(feat))
    ids = np.unique(flat[good])
    if len(ids):
        f = np.asarray(feat[ids], dtype=np.float32)
        nf = np.asarray(node_feat, dtype=np.float32)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        nf /= np.maximum(np.linalg.norm(nf, axis=1, keepdims=True), 1e-12)
        mapped = np.argmax(f @ nf.T, axis=1).astype(np.int32)
        lut = dict(zip(ids.tolist(), mapped.tolist()))
        out_flat = out.ravel()
        for k, v in lut.items(): out_flat[flat == k] = v
    return out


def _parent_map(point_child, point_parent, n_child, n_parent):
    result = np.full(n_child, -1, dtype=np.int32)
    for c in range(n_child):
        p = np.asarray(point_parent)[np.asarray(point_child) == c]
        if len(p): result[c] = int(np.bincount(p, minlength=n_parent).argmax())
    return result


def _icc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3: return float('nan')
    x = np.c_[a[ok], b[ok]]; grand = x.mean(); ms_between = 2 * ((x.mean(1) - grand) ** 2).sum() / (len(x) - 1)
    ms_error = ((x - x.mean(1, keepdims=True)) ** 2).sum() / len(x)
    return float((ms_between - ms_error) / max(ms_between + ms_error, 1e-12))


def _load_views(root, cameras, level, node_feat, max_views=None):
    labels = []
    for cam in cameras[:max_views] if max_views else cameras:
        stem = cam['img_name']; s = np.load(root / f'{stem}_s.npy', mmap_mode='r'); f = np.load(root / f'{stem}_f.npy', mmap_mode='r')
        labels.append(_canonical_labels(s[level], f, node_feat))
    return labels


def run(args):
    import torch
    torch.set_num_threads(4)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path(args.data_root)
    model = Path(args.model); ply = Path(args.ply); cam_path = Path(args.cameras)
    obj = torch.load(model, map_location='cpu')
    nag = [np.asarray(x.detach().cpu()) for x in obj['nag']]
    feats = [np.asarray(x.detach().cpu()) for x in obj['nag_feat']]
    points = _read_ply_xyz(ply)
    cameras = sorted(json.loads(cam_path.read_text()), key=lambda c: c['img_name'])
    if args.max_views: cameras = cameras[:args.max_views]
    records = []; level_reports = []; all_observed = []
    rng = np.random.default_rng(args.seed)
    for li, node_feat in enumerate(feats):
        point_labels = nag[li + 1]
        n_nodes = len(node_feat); centers = np.zeros((n_nodes, 3), np.float32); counts = np.bincount(point_labels, minlength=n_nodes)
        for i in range(n_nodes):
            if counts[i]: centers[i] = points[point_labels == i].mean(0)
        parent = _parent_map(point_labels, nag[li], n_nodes, len(feats[li - 1])) if li else np.full(n_nodes, -1, dtype=np.int32)
        if li:
            assert point_labels.shape == nag[li].shape and parent.max(initial=-1) < len(feats[li - 1]), 'adjacent NAG level shape/index mismatch'
        view_labels = _load_views(root, cameras, li, node_feat, None)
        n_views = len(view_labels); q_full = np.zeros((n_nodes, n_nodes), float); valid = np.zeros((n_nodes, n_views), bool); observed_labels = np.full((n_nodes, n_views), -1, dtype=np.int32)
        depths = np.full((n_nodes, n_views), np.nan); borders = np.full((n_nodes, n_views), np.nan); boundary = np.full((n_nodes, n_views), np.nan)
        for vi, (cam, lab) in enumerate(zip(cameras, view_labels)):
            R = np.asarray(cam['rotation'], float); pos = np.asarray(cam['position'], float); w, h = int(cam['width']), int(cam['height'])
            wc = np.eye(4); wc[:3, :3] = R.T; wc[:3, 3] = -R.T @ pos
            K = np.array([[cam['fx'], 0, w / 2], [0, cam['fy'], h / 2], [0, 0, 1.]])
            uv, ok, z = project_centers(centers, wc, K, w, h); pix = np.zeros_like(uv, dtype=int); finite = np.isfinite(uv).all(1); pix[finite] = np.rint(uv[finite]).astype(int); ok &= finite & (pix[:, 1] >= 0) & (pix[:, 1] < h) & (pix[:, 0] >= 0) & (pix[:, 0] < w)
            y = np.full(n_nodes, -1, np.int32); y[ok] = lab[pix[ok, 1], pix[ok, 0]]; ok &= y >= 0; observed_labels[:, vi] = y; valid[:, vi] = ok; depths[ok, vi] = z[ok]
            borders[ok, vi] = np.minimum.reduce([pix[ok, 0], w - 1 - pix[ok, 0], pix[ok, 1], h - 1 - pix[ok, 1]]) / max(w, h)
            for i in np.flatnonzero(ok):
                x0, y0 = pix[i]; patch = lab[max(0, y0 - 1):min(h, y0 + 2), max(0, x0 - 1):min(w, x0 + 2)]
                boundary[i, vi] = 1.0 - float((patch == y[i]).mean())
            for i in np.flatnonzero(ok): q_full[i, y[i]] += 1
        q = q_full / np.maximum(q_full.sum(1, keepdims=True), 1); H = np.array([normalized_entropy(x) for x in q]); V = valid.sum(1).astype(float)
        mean_depth = np.divide(np.nansum(depths, 1), np.maximum(np.isfinite(depths).sum(1), 1)); mean_border = np.divide(np.nansum(borders, 1), np.maximum(np.isfinite(borders).sum(1), 1)); mean_boundary = np.divide(np.nansum(boundary, 1), np.maximum(np.isfinite(boundary).sum(1), 1))
        area_proxy = np.log1p(counts) - 2.0 * np.log(np.maximum(mean_depth, 1e-6)); coverage = q.max(1)
        geom = np.c_[V, area_proxy, mean_depth, mean_border, np.full(n_nodes, li)]
        parent_counts = np.bincount(nag[li], minlength=(len(feats[li - 1]) if li else n_nodes)) if li else np.ones(n_nodes, dtype=int)
        rel = np.c_[np.where(parent >= 0, 1.0, 0.0), np.where(parent >= 0, np.log1p(parent_counts[np.maximum(parent, 0)]) - np.log1p(counts), 0.0)]
        split_stats = []; delta_r2 = []; delta_auc = []; lift = []; tvs = []; iccs = []; d_fdrs = []
        for rep in range(args.repeats):
            idx = rng.permutation(n_views); A, B = idx[:n_views // 2], idx[n_views // 2:]
            qa = np.zeros_like(q); qb = np.zeros_like(q)
            arrA = observed_labels[:, A].T; arrB = observed_labels[:, B].T
            for i in range(n_nodes):
                qa[i] = vote_distribution(arrA[:, i], np.arange(n_nodes))
                qb[i] = vote_distribution(arrB[:, i], np.arange(n_nodes))
            hA = np.array([normalized_entropy(x) for x in qa]); hB = np.array([normalized_entropy(x) for x in qb]); tv = np.array([0.5 * np.abs(qa[i] - qb[i]).sum() for i in range(n_nodes)]); tvs.extend(tv[V > 1].tolist()); iccs.append(_icc(hA, hB))
            child = np.flatnonzero(parent >= 0)
            if li:
                parr = all_observed[li - 1]
                def own_binary(obs, views, node):
                    x = obs[node, views]; x = x[x >= 0]
                    p = float((x == node).mean()) if len(x) else 0.0
                    return np.array([p, 1.0 - p])
                dA = np.array([js_divergence(own_binary(observed_labels, A, i), own_binary(parr, A, parent[i])) for i in child]); dB = np.array([js_divergence(own_binary(observed_labels, B, i), own_binary(parr, B, parent[i])) for i in child])
            else:
                dA = np.zeros(len(child)); dB = np.zeros(len(child))
            iccs.append(_icc(dA, dB))
            if len(child) == 0:
                split_stats.append({'rep': rep, 'h_tv_median': float(np.median(tv[V > 1])) if np.any(V > 1) else None, 'h_icc': iccs[-1], 'd_icc': None, 'nodes': int(n_nodes), 'parent_pairs': 0})
                continue
            depthA = np.divide(np.nansum(depths[:, A], 1), np.maximum(np.isfinite(depths[:, A]).sum(1), 1)); borderA = np.divide(np.nansum(borders[:, A], 1), np.maximum(np.isfinite(borders[:, A]).sum(1), 1)); boundaryA = np.divide(np.nansum(boundary[:, A], 1), np.maximum(np.isfinite(boundary[:, A]).sum(1), 1)); areaA = np.log1p(counts) - 2.0 * np.log(np.maximum(depthA, 1e-6)); coverageA = qa.max(1); VA = (arrA >= 0).sum(0).astype(float)
            x0 = np.c_[VA, areaA, depthA, borderA, np.full(n_nodes, li)][child]; x1 = np.c_[x0, coverageA, boundaryA, rel][child]
            if len(child) >= 10:
                delta_r2.append(ridge_delta_r2(x0, x1, dB, folds=5)); ybin = dB >= np.quantile(dB, .75); p0 = _ridge_fit_predict(x0, dB, x0, 1.0); p1 = _ridge_fit_predict(x1, dB, x1, 1.0); delta_auc.append(float(np.mean(ybin[np.argsort(p1)] > np.median(p1)))); lift.append(float(np.mean(ybin[np.argsort(p1)[-max(1, len(ybin)//4):]]) / max(ybin.mean(), 1e-12)))
            split_stats.append({'rep': rep, 'h_tv_median': float(np.median(tv[V > 1])) if np.any(V > 1) else None, 'h_icc': iccs[-2], 'd_icc': iccs[-1], 'nodes': int(n_nodes), 'parent_pairs': int(len(child))})
        report = {'level': li + 1, 'nodes': n_nodes, 'views': n_views, 'valid_projection_rate': float(valid.mean()), 'entropy_median': float(np.median(H[V > 0])) if np.any(V > 0) else None, 'coverage_median': float(np.median(coverage[V > 0])) if np.any(V > 0) else None, 'boundary_proxy_median': float(np.median(mean_boundary[V > 0])) if np.any(V > 0) else None, 'area_proxy_median': float(np.median(area_proxy[V > 0])) if np.any(V > 0) else None, 'split_half_tv_median': float(np.median(tvs)) if tvs else None, 'split_half_icc_median': float(np.nanmedian(iccs)) if iccs else None, 'delta_r2_median': float(np.median(delta_r2)) if delta_r2 else None, 'delta_auc_median': float(np.median(delta_auc)) if delta_auc else None, 'precision_lift_median': float(np.median(lift)) if lift else None, 'split_stats': split_stats, 'descriptor_schema': 'M0=V/logA/depth/border/level; M1 adds coverage c, local boundary proxy b, structural parent relation', 'proxy': 'center projection into existing seg_map; no occlusion visibility'}
        all_observed.append(observed_labels)
        level_reports.append(report)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / 'diagnostic_report.json').write_text(json.dumps({'status': 'CPU_PROXY_ONLY', 'levels': level_reports, 'input_sha256': {str(p): _sha(p) for p in [model, ply, cam_path]}, 'source_sha256': _sha(Path(__file__))}, indent=2))
    with (out / 'level_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['level','nodes','views','valid_projection_rate','entropy_median','split_half_tv_median','split_half_icc_median','delta_r2_median','delta_auc_median','precision_lift_median','proxy']); w.writeheader(); [w.writerow({k: x.get(k) for k in w.fieldnames}) for x in level_reports]
    return level_reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data-root', required=True); p.add_argument('--model', required=True); p.add_argument('--ply', required=True); p.add_argument('--cameras', required=True); p.add_argument('--out', required=True); p.add_argument('--repeats', type=int, default=20); p.add_argument('--max-views', type=int); p.add_argument('--seed', type=int, default=20261025); run(p.parse_args())


if __name__ == '__main__': main()
