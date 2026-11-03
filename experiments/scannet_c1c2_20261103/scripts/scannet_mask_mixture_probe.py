#!/usr/bin/env python3
"""Diagnostic: is the per-cluster mask mixture the reason root clusters are impure?

For every root cluster (the partition the ScanNet readout actually labels) and every
training view, reproduce merge_proj's own mask assignment (`trace` via
`SAI3D.extract_gaussian_features`, level 3) and record the distribution of 2D masks
covering that cluster in that view. Then reduce to per-cluster statistics:

  n_masks_30   mean number of masks holding >30% of the cluster (merge_proj's threshold)
  entropy      mean entropy of the normalised mask distribution
  top1         mean share of the dominant mask
  frac_multi   fraction of views where more than one mask passes 0.3

and correlate with the cluster's GT purity (dominant NYU40 class share).

Mechanism test: if impure clusters are exactly the ones with blended mask
distributions, the loss is the descriptor and a dominant-mask field should fix it.
If purity is unrelated to mixture entropy, the loss is region growing itself.

Writes <out>/<scene>_obs.npz (CSR of the raw distributions) and <out>/rows_<scene>.csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ext"))

from arguments import ModelParams, get_combined_args  # noqa: E402
from gaussian_renderer import render_point  # noqa: E402
from merge_proj import SAI3D  # noqa: E402
from scene import GaussianModel, Scene  # noqa: E402
from types import SimpleNamespace  # noqa: E402


def read_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64) if "label" in v.data.dtype.names else None
    return xyz, lab


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--nag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--iteration", type=int, default=0)
    args = get_combined_args(parser)
    pargs = model.extract(args)

    torch.manual_seed(0)
    np.random.seed(0)

    agent = SAI3D(SimpleNamespace(max_neighbor_distance=2, view_freq=5, dis_decay=0.5))
    gaussians = GaussianModel(3, 0)
    scene = Scene(pargs, gaussians, load_iteration=args.iteration, shuffle=False,
                  load_sem=True, load_test=False, vlm_type="clip")
    views = scene.getTrainCameras()
    N = gaussians.get_xyz.shape[0]

    root = np.load(os.path.join(pargs.model_path, "saifine_labels.npy")).astype(np.int64)
    if root.min() >= 1:
        root = root - 1
    assert root.shape[0] == N
    n_sp = int(root.max()) + 1
    print("cluster source: saifine_labels.npy, n_sp=%d, N=%d" % (n_sp, N), flush=True)

    g_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    gt_xyz, gt_lab = read_ply(args.gt)
    g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]
    cls_cnt = np.bincount(root * 41 + g_cls, minlength=n_sp * 41).reshape(n_sp, 41)
    purity = cls_cnt.max(1) / np.maximum(cls_cnt.sum(1), 1)
    dom_cls = cls_cnt.argmax(1)

    sp_all, view_all, mask_all, cnt_all = [], [], [], []
    vis_portion = np.zeros((n_sp, len(views)), np.float32)
    has_obs = np.zeros((n_sp, len(views)), bool)
    mask_cls = np.zeros((len(views), 512, 41), np.int32)  # per (view, mask) GT class histogram

    with torch.no_grad():
        for v0 in range(0, len(views), args.chunk):
            chunk = views[v0:v0 + args.chunk]
            lab = agent.extract_gaussian_features(gaussians, chunk, args.level)  # (N, len)
            lab = lab.cpu().numpy().astype(np.int64)
            for j, view in enumerate(chunk):
                vi = v0 + j
                seg_map = view.semantic["seg_map"][args.level]
                img_mask = view.semantic["fg_mask"][args.level]
                seg_min = int(seg_map[img_mask].min())
                seg_max = int(seg_map[img_mask].max())
                render_pkg = render_point(view, gaussians, agent.pipeline, agent.background)
                vis = (render_pkg["weight"] > 1e-4).cpu().numpy()

                lj = lab[:, j]
                ok = lj > 0
                if not ok.any():
                    continue
                gm = lj[ok] + seg_min - 1                      # global mask id
                sp = root[ok]
                mc = np.bincount(gm.astype(np.int64)[gm < 512] * 41 + g_cls[ok][gm < 512],
                                 minlength=512 * 41).reshape(512, 41)
                mask_cls[vi] += mc.astype(np.int32)
                key = sp * 512 + gm
                uniq, cnt = np.unique(key, return_counts=True)
                s_, m_ = uniq // 512, uniq % 512
                sp_all.append(s_.astype(np.int32))
                view_all.append(np.full(s_.size, vi, np.int32))
                mask_all.append(m_.astype(np.int32))
                cnt_all.append(cnt.astype(np.int32))

                tot = np.bincount(s_, weights=cnt, minlength=n_sp)
                seen = tot > 0
                has_obs[seen, vi] = True
                nseed = np.bincount(root[vis], minlength=n_sp).astype(np.float32)
                vis_portion[:, vi] = nseed / np.maximum(np.bincount(root, minlength=n_sp), 1)
            print("view %d/%d" % (min(v0 + args.chunk, len(views)), len(views)), flush=True)

    sp_all = np.concatenate(sp_all)
    view_all = np.concatenate(view_all)
    mask_all = np.concatenate(mask_all)
    cnt_all = np.concatenate(cnt_all)

    # per (cluster, view) summary
    key = sp_all.astype(np.int64) * 100000 + view_all
    order = np.argsort(key, kind="stable")
    sp_s, view_s, mask_s, cnt_s = sp_all[order], view_all[order], mask_all[order], cnt_all[order]
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, key.size]

    n_masks_30 = np.zeros(n_sp)
    entropy = np.zeros(n_sp)
    top1 = np.zeros(n_sp)
    frac_multi = np.zeros(n_sp)
    n_views = np.zeros(n_sp, np.int64)
    for a, b in zip(starts, ends):
        s = sp_s[a]
        c = cnt_s[a:b].astype(np.float64)
        p = c / c.sum()
        n_masks_30[s] += float((p > 0.3).sum())
        pp = p[p > 0]
        entropy[s] += float(-(pp * np.log(pp)).sum())
        top1[s] += float(p.max())
        frac_multi[s] += float((p > 0.3).sum() > 1)
        n_views[s] += 1
    d = np.maximum(n_views, 1)
    n_masks_30 /= d
    entropy /= d
    top1 /= d
    frac_multi /= d

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "%s_L%d_obs.npz" % (os.path.basename(pargs.model_path), args.level)),
                        sp=sp_all, view=view_all, mask=mask_all, count=cnt_all,
                        n_sp=n_sp, n_views=len(views), mask_cls=mask_cls,
                        names=np.array([v.image_name for v in views]))
    rows = [dict(scene=os.path.basename(pargs.model_path), cluster=i,
                 n_gauss=int((root == i).sum()), gt_dom=int(dom_cls[i]),
                 purity=float(purity[i]), n_views=int(n_views[i]),
                 n_masks_30=float(n_masks_30[i]), entropy=float(entropy[i]),
                 top1=float(top1[i]), frac_multi=float(frac_multi[i]))
            for i in range(n_sp)]
    with open(os.path.join(args.out, "rows_%s_L%d.csv" % (os.path.basename(pargs.model_path), args.level)),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = n_views > 0
    tot = mask_cls.sum(2)
    mpur = mask_cls.max(2) / np.maximum(tot, 1)
    big = tot >= 200
    print(json.dumps(dict(scene=os.path.basename(pargs.model_path), level=args.level, n_sp=n_sp,
                          obs_clusters=int(ok.sum()),
                          purity_mean=float(purity[ok].mean()),
                          n_masks_30_mean=float(n_masks_30[ok].mean()),
                          entropy_mean=float(entropy[ok].mean()),
                          frac_multi_mean=float(frac_multi[ok].mean()),
                          n_masks_big=int(big.sum()),
                          mask_purity_mean=float(mpur[big].mean()) if big.any() else 0.0,
                          mask_purity_med=float(np.median(mpur[big])) if big.any() else 0.0,
                          mask_purity_ge90=float((mpur[big] >= 0.9).mean()) if big.any() else 0.0)))


if __name__ == "__main__":
    main()