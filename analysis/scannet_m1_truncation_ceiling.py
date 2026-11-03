"""M1: truncation ceiling for the superpoint-merge stage.

Question: if every mask could be perfectly cleaned of the superpoints that do not
belong to its dominant GT class, how much would the final partition improve?
This is the common upper bound of every truncation-shaped method (Robust P^n Potts
hypergraph, bridge-cutting Multicut, soft/UOT association).

Method
  * reproduce merge_proj's own progressive region growing (3 iterations, mask
    levels 1/2/3) -- the partition depends ONLY on the per-gaussian mask labels
    and the merge thresholds, so no CLIP feature projection is needed.
  * per-gaussian mask labels are extracted once and cached; every arm reuses the
    same labels, so arms are paired and the renderer's random codes cannot leak
    into the comparison.
  * truncation is applied to the cached labels: per (view, mask), the majority GT
    class is taken as the mask's class and gaussians of other classes are dropped,
    capped at a fraction gamma of the mask.
  * `--mode rand` drops the same number of gaussians uniformly at random: if that
    matches GT truncation, the gain is from fragmentation, not from correct removal.

GT is used to choose what to drop (diagnostic oracle only).  The reported number is
a ceiling, not a method result.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

REL = "/root/rivermind-data/ReLaGS_Reproduction/ReLaGS"
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "ext"))

import utils.sai3d_utils as utils  # noqa: E402
from arguments import ArgumentParser, ModelParams, get_combined_args  # noqa: E402
from merge_proj import SAI3D  # noqa: E402

SUBSETS = {
    19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    10: [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}


def read_ply(path):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    lab = np.asarray(v["label"]).astype(np.int64)
    return xyz, lab


def partition_ceiling(lab, valid, g_cls, classes):
    """Best mIoU any classifier can reach on this partition (dominant class per unit)."""
    n_cand = int(lab.max()) + 1
    idx = lab[valid].astype(np.int64) * 41 + g_cls[valid]
    cnt = np.bincount(idx, minlength=n_cand * 41).reshape(n_cand, 41)
    dom = cnt.argmax(1)
    pred = np.zeros_like(g_cls)
    pred[valid] = dom[lab[valid]]
    ious = []
    for c in classes:
        g = g_cls == c
        p = pred == c
        u = np.logical_or(g, p).sum()
        ious.append(float(np.logical_and(g, p).sum() / u) if u else 0.0)
    return float(np.mean(ious)), ious


def reassign(lab, g_cls, neighbors):
    """Reassign each mislabelled gaussian to a mask of its own GT class.

    This is the oracle form of what a soft / UOT association would do: instead of
    deleting the wrong claim it moves it to the right one.  The target is chosen by
    the majority of the gaussian's 8 spatial neighbours whose mask has the right
    dominant class.  GT decides only the class, never the merge.
    """
    out = lab.copy()
    N, V = out.shape
    ar = np.arange(N)
    for v in range(V):
        col = out[:, v]
        nz = col > 0
        if not nz.any():
            continue
        nm = int(col.max()) + 1
        hist = np.bincount(col[nz] * 41 + g_cls[nz],
                           minlength=nm * 41).reshape(nm, 41)
        dom = hist.argmax(1)
        safe = np.clip(col, 0, nm - 1)
        bad = nz & (dom[safe] != g_cls)
        idx = np.flatnonzero(bad)
        if idx.size == 0:
            continue
        nb = col[neighbors[idx]]                       # (K, 8)
        nb_safe = np.clip(nb, 0, nm - 1)
        ok = (nb > 0) & (dom[nb_safe] == g_cls[idx][:, None])
        cnt = np.zeros((idx.size, nm), np.int32)
        rows = np.repeat(ar[:idx.size][:, None], nb.shape[1], 1)
        np.add.at(cnt, (rows[ok], nb[ok]), 1)
        allowed = dom[None, :] == g_cls[idx][:, None]
        cnt[~allowed] = 0
        target = cnt.argmax(1)
        hit = cnt.max(1) > 0
        col2 = out[:, v]
        col2[idx[hit]] = target[hit]
        out[:, v] = col2
    return out


def truncate(lab, g_cls, gamma, mode, rng):
    """Drop, per (view, mask), up to gamma of the mask's gaussians.

    mode='gt'   : drop the ones whose GT class differs from the mask's majority
    mode='rand' : drop the same number uniformly at random
    """
    out = lab.copy()
    N, V = out.shape
    for v in range(V):
        col = out[:, v]
        nz = np.flatnonzero(col > 0)
        if nz.size == 0:
            continue
        m = col[nz]
        nm = int(m.max()) + 1
        cls = g_cls[nz]
        hist = np.bincount(m * 41 + cls, minlength=nm * 41).reshape(nm, 41)
        tot = hist.sum(1)
        dom = hist.argmax(1)
        if mode == "gt":
            bad = cls != dom[m]
        else:
            bad = np.zeros(nz.size, bool)
        order = np.argsort(m, kind="stable")
        m_s, bad_s, nz_s = m[order], bad[order], nz[order]
        starts = np.searchsorted(m_s, np.arange(nm), side="left")
        ends = np.searchsorted(m_s, np.arange(nm), side="right")
        drop = np.zeros(nz.size, bool)
        for k in range(nm):
            a, b = starts[k], ends[k]
            n = b - a
            if n == 0 or tot[k] == 0:
                continue
            cap = int(np.floor(gamma * tot[k]))
            if cap <= 0:
                continue
            if mode == "gt":
                idxs = np.flatnonzero(bad_s[a:b]) + a
            else:
                idxs = rng.permutation(n)[:cap] + a
            if idxs.size > cap:
                idxs = idxs[:cap]
            drop[idxs] = True
        out[nz_s[drop], v] = 0
    return out


def grow(agent, labels, thres_connect, thres_merge, max_neighbor_distance,
         gamma, mode, rng, log):
    """Reproduce SAI3D.assign_label, minus the feature projection (not needed for
    the partition), with truncation applied to the cached per-view mask labels."""
    agent.seg_ids = torch.load(os.path.join(agent.args.model_path, "nag-l1.pt"),
                               map_location="cpu").cuda()
    nag = [agent.seg_ids]
    pt_prim_label = None
    for i, th in enumerate(thres_connect):
        lab = labels[i + 1]
        if mode == "reassign":
            lab = reassign(np.asarray(lab), agent.g_cls,
                           agent.points_neighbors.cpu().numpy())
        elif gamma > 0:
            lab = truncate(lab, agent.g_cls, gamma, mode, rng)
        lab_t = torch.from_numpy(lab).cuda()
        (agent.seg_ids, agent.seg_num, agent.seg_members,
         agent.seg_indirect_neighbors) = agent.get_seg_data_torch()
        agent.seg_direct_neighbors = agent.seg_indirect_neighbors[0]
        seg_adj = agent.get_seg_adjacency(points_label=lab_t, points_seen=None)
        seg_labels = agent.assign_seg_label_torch(
            seg_adj, th, max_neighbor_distance=max_neighbor_distance)
        if i == len(thres_connect) - 1 and thres_merge > 0:
            seg_labels = agent.merge_small_segs_torch(seg_labels, thres_merge,
                                                      seg_adj)
            seg_labels = utils.num_to_natural_torch(seg_labels) + 1
        pt_prim_label = torch.zeros(agent.N, dtype=torch.int,
                                    device=seg_labels.device)
        for j in range(agent.seg_num):
            if j in agent.seg_members:
                pt_prim_label[agent.seg_members[j]] = seg_labels[j]
        nag.append(pt_prim_label - 1)
        agent.seg_ids = pt_prim_label
        log("    iter %d (level %d) -> %d regions"
            % (i, i + 1, int(pt_prim_label.max().item()) + 1))
    return (pt_prim_label - 1).cpu().numpy().astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--thres_connect", default="0.9,0.7,0.7")
    parser.add_argument("--thres_merge", type=int, default=20)
    parser.add_argument("--max_neighbor_distance", type=int, default=2)
    parser.add_argument("--view_freq", type=int, default=5)
    parser.add_argument("--dis_decay", type=float, default=0.5)
    parser.add_argument("--thres_trunc", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=3.0)
    parser.add_argument("--seg_enhance", action="store_true")
    parser.add_argument("--arm", default="gt:0,gt:0.1,rand:0.1,gt:0.3,gt:1.0",
                        help="mode:gamma,...")
    args = get_combined_args(parser)
    args.thres_connect = [float(x) for x in args.thres_connect.split(",")]
    args.vlm_type = getattr(args, "vlm_type", "clip")
    tag = os.path.basename(os.path.dirname(args.model_path))
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, tag + "_m1.log"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    agent = SAI3D(args)
    agent.init_data(args)
    g_xyz = agent.gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    gt_xyz, gt_lab = read_ply(args.gt)
    agent.g_cls = gt_lab[cKDTree(gt_xyz).query(g_xyz, workers=8)[1]]
    log("[load] %s  gaussians=%d  views=%d  (%.1fs)"
        % (tag, agent.N, agent.M, time.time() - t0))

    cache = args.cache or args.out
    os.makedirs(cache, exist_ok=True)
    views = agent.scene.getTrainCameras()
    labels = {}
    for lvl in (1, 2, 3):
        p = os.path.join(cache, "%s_L%d_labels.npy" % (tag, lvl))
        if os.path.exists(p):
            labels[lvl] = np.load(p, mmap_mode="r")
            log("[cache] %s" % p)
            continue
        t = time.time()
        lab = agent.extract_gaussian_features(agent.gaussians, views, lvl)
        lab = lab.cpu().numpy().astype(np.int16)
        np.save(p, lab)
        labels[lvl] = lab
        log("[extract] level %d -> %s  %s  (%.1fs)"
            % (lvl, lab.shape, p, time.time() - t))
        del lab
        torch.cuda.empty_cache()

    g_cls = agent.g_cls
    rows = []
    for arm in args.arm.split(","):
        mode, g = arm.split(":")
        gamma = float(g)
        t = time.time()
        rng = np.random.default_rng(20261103)
        log("[arm] mode=%s gamma=%.2f" % (mode, gamma))
        lab_final = grow(agent, labels, args.thres_connect, args.thres_merge,
                         args.max_neighbor_distance, gamma, mode, rng, log)
        out = dict(scene=tag, mode=mode, gamma=gamma,
                   n_units=int(lab_final.max()) + 1,
                   sec=round(time.time() - t, 1))
        for k in (10, 19):
            v = np.isin(g_cls, SUBSETS[k])
            m, _ = partition_ceiling(lab_final, v, g_cls, SUBSETS[k])
            out["oracle_miou_%d" % k] = round(m, 4)
        rows.append(out)
        log("  -> units=%d  oracle10=%.4f  oracle19=%.4f  (%.1fs)"
            % (out["n_units"], out["oracle_miou_10"], out["oracle_miou_19"],
               time.time() - t))
        np.save(os.path.join(args.out, "%s_%s_g%.2f_labels.npy"
                             % (tag, mode, gamma)), lab_final)

    with open(os.path.join(args.out, tag + "_m1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log("[done] " + json.dumps(rows))


if __name__ == "__main__":
    main()