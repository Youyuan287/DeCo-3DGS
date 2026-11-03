#!/usr/bin/env python3
"""Class-agnostic coverage of ScanNet GT regions by the existing NAG candidate bank.

Question: does the hierarchical candidate field contain, for each GT region, any
single cluster that represents it well? No text, no query, no rendering, no GT in
any selection: every candidate is scored against every GT region and we keep the best.

Target unit: pseudo-instance = 3D connected component (5 cm voxel, 26-conn) of the
GT points of one NYU40 class. This avoids the multi-instance pitfall of scoring a
whole class region (three chairs 2 m apart would cap IoU at 1/3 for every method).

Outputs: <out>/<scene>.json and <out>/summary.csv
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from plyfile import PlyData
from scipy import ndimage
from scipy.sparse import csr_matrix

NYU40 = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop",
}
SUBSETS = {
    19: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    10: [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}
STRUCT = {1, 2, 22}  # wall / floor / ceiling: trivially large, reported separately


def partition_ceiling(lab, valid, g_cls, classes):
    """Best mIoU any semantic classifier could reach on this partition.

    Each cluster is labelled with its dominant GT class (perfect oracle within the
    partition), then per-class point-set IoU is computed exactly as the evaluator does.
    The gap between this and the measured mIoU is the text/readout loss; the gap
    between this and 1.0 is the partition's own expressiveness loss.
    """
    n_cand = int(lab.max()) + 1
    idx = lab[valid].astype(np.int64) * 41 + g_cls[valid]
    cnt = np.bincount(idx, minlength=n_cand * 41).reshape(n_cand, 41)
    dom = cnt.argmax(1)
    pred = np.zeros_like(g_cls)
    pred[valid] = dom[lab[valid]]

    k = len(classes)
    g = np.zeros_like(g_cls)
    p = np.zeros_like(pred)
    for i, c in enumerate(classes):
        g[g_cls == c] = i + 1
        p[pred == c] = i + 1
    p[g == 0] = 0  # points outside the class subset cannot be false positives (matches metrics())
    ious = []
    per_class = {}
    for i in range(1, k + 1):
        if (g == i).sum() == 0:
            continue
        u = np.logical_or(g == i, p == i).sum()
        v = float(np.logical_and(g == i, p == i).sum() / u) if u else 0.0
        ious.append(v)
        per_class[classes[i - 1]] = v
    return (float(np.mean(ious)) if ious else 0.0), len(ious), per_class


def read_ply(path):
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    names = v.data.dtype.names
    lab = np.asarray(v["label"]).astype(np.int64) if "label" in names else None
    return xyz, lab


def voxel_instances(pts, voxel=0.05, min_pts=200):
    key = np.floor(pts / voxel).astype(np.int64)
    key0 = key - key.min(0)
    dim = key0.max(0) + 1
    grid = np.zeros(dim, dtype=bool)
    grid[key0[:, 0], key0[:, 1], key0[:, 2]] = True
    lab_grid, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    comp = lab_grid[key0[:, 0], key0[:, 1], key0[:, 2]]
    cnt = np.bincount(comp, minlength=n + 1)
    keep = cnt >= min_pts
    keep[0] = False
    return comp, keep, cnt




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--nag", required=True)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--min-points", type=int, default=200)
    args = ap.parse_args()

    g_xyz = read_ply(os.path.join(args.model, "point_cloud",
                                  "iteration_%d" % args.iteration, "point_cloud.ply"))[0]
    N = g_xyz.shape[0]
    gt_xyz, gt_lab = read_ply(args.gt)
    nag = __import__("torch").load(args.nag, map_location="cpu")["nag"]
    levels = [t.numpy().astype(np.int64).reshape(-1) for t in nag]
    assert levels[0].shape[0] == N, (levels[0].shape, N)

    # gaussian -> nearest GT point -> (class, component)
    from scipy.spatial import cKDTree
    idx = cKDTree(gt_xyz).query(g_xyz, workers=8)[1]
    g_cls = gt_lab[idx]

    inst_of_gt = np.full(gt_xyz.shape[0], -1, dtype=np.int64)
    inst_cls, inst_size, inst_vox = [], [], []
    next_id = 0
    for c in range(1, 41):
        sel = gt_lab == c
        if sel.sum() < args.min_points:
            continue
        comp, keep, cnt = voxel_instances(gt_xyz[sel], args.voxel, args.min_points)
        for cid in np.flatnonzero(keep):
            m = comp == cid
            inst_of_gt[np.flatnonzero(sel)[m]] = next_id
            inst_cls.append(c)
            inst_size.append(int(m.sum()))
            inst_vox.append(float(cnt[cid]))
            next_id += 1
    inst_cls = np.asarray(inst_cls, dtype=np.int64)
    inst_size = np.asarray(inst_size, dtype=np.int64)
    n_inst = len(inst_cls)
    g_inst = inst_of_gt[idx]

    rows = []
    per_level = {}
    for L, lab in enumerate(levels):
        valid = lab >= 0
        n_cand = int(lab.max()) + 1
        M = csr_matrix((np.ones(int(valid.sum()), np.float32),
                        (np.flatnonzero(valid), lab[valid])), shape=(N, n_cand))
        cand_size = np.bincount(lab[valid], minlength=n_cand).astype(np.float64)

        sel = g_inst >= 0
        G = csr_matrix((np.ones(int(sel.sum()), np.float32),
                        (np.flatnonzero(sel), g_inst[sel])), shape=(N, n_inst))
        inst_real = np.bincount(g_inst[sel], minlength=n_inst).astype(np.float64)
        inter = (M.T @ G).toarray()
        union = cand_size[:, None] + inst_real[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        best = iou.max(0)
        best_c = iou.argmax(0)

        Gc = csr_matrix((np.ones(int(sel.sum()), np.float32),
                         (np.flatnonzero(sel), g_cls[sel])), shape=(N, 41))
        cls_cnt = (M.T @ Gc).toarray()
        purity = cls_cnt.max(1) / np.maximum(cand_size, 1)
        dom = cls_cnt.argmax(1)

        ceil10, n10, per10 = partition_ceiling(lab, valid, g_cls, SUBSETS[10])
        ceil19, n19, _ = partition_ceiling(lab, valid, g_cls, SUBSETS[19])

        per_level[L] = dict(
            n_cand=int(n_cand),
            ceil10=ceil10, n_cls10=n10, ceil10_per_class={str(k): v for k, v in per10.items()},
            ceil19=ceil19, n_cls19=n19,
            mean_best=float(best.mean()) if n_inst else 0.0,
            median_best=float(np.median(best)) if n_inst else 0.0,
            frac_ge30=float((best >= 0.3).mean()) if n_inst else 0.0,
            frac_ge50=float((best >= 0.5).mean()) if n_inst else 0.0,
            frac_ge70=float((best >= 0.7).mean()) if n_inst else 0.0,
            purity_mean=float(purity.mean()),
            purity_median=float(np.median(purity)),
            pure_ge90=float((purity >= 0.9).mean()),
        )
        for i in range(n_inst):
            rows.append(dict(scene=args.scene, level=L, inst=i, cls=int(inst_cls[i]),
                             cls_name=NYU40.get(int(inst_cls[i]), "?"),
                             struct=int(inst_cls[i]) in STRUCT,
                             gt_pts=int(inst_size[i]), gauss=int(inst_real[i]),
                             best_iou=float(best[i]), best_cand=int(best_c[i]),
                             cand_purity=float(purity[best_c[i]])))

    out = dict(scene=args.scene, n_gauss=int(N), n_gt=int(len(gt_lab)),
               n_inst=int(n_inst), levels=per_level,
               level_counts=[int(l.max()) + 1 for l in levels])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.scene + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    import csv
    p = os.path.join(args.out, "rows_%s.csv" % args.scene)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({k: v for k, v in out.items() if k != "levels"}))
    for L in sorted(per_level):
        d = per_level[L]
        print("L%d cand=%6d mean=%.3f med=%.3f >=.3=%.3f >=.5=%.3f >=.7=%.3f pur=%.3f "
              "ceil10=%.3f ceil19=%.3f"
              % (L, d["n_cand"], d["mean_best"], d["median_best"], d["frac_ge30"],
                 d["frac_ge50"], d["frac_ge70"], d["purity_mean"],
                 d["ceil10"], d["ceil19"]))


if __name__ == "__main__":
    main()