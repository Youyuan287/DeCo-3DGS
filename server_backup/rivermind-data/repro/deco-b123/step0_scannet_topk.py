"""Top-k oracle on the ScanNet DeCo (C1+C2) main arm.

Each root takes its GT majority class whenever that class is among the main-score
top-k; k sweeps 1..19, k=19 reproduces the root_any partition ceiling. Tells how
deep a selector has to look beyond the current argmax prediction.
"""
import sys
import numpy as np

CODE = "/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code"
REL = "/root/rivermind-data/repro/deco-b123/frozen_ramen_relags"
sys.path[:0] = [CODE, REL, REL + "/ext"]
from eval_scannet_3d import metrics, SUBSETS

B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
S = ("scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 "
     "scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00").split()
CL = list(SUBSETS[19])
KS = [1, 2, 3, 4, 5, 7, 10, 15, 19]
inter = {k: {} for k in KS}
union = {k: {} for k in KS}
mac = {k: [] for k in KS}
n_in = {k: 0 for k in KS}
n_valid = 0


def maj_class(g):
    g = g[np.isin(g, CL)]
    if len(g) == 0:
        return None
    v, c = np.unique(g, return_counts=True)
    return int(v[c.argmax()])


for s in S:
    R = f"{B}/ablation/v2_on_on/{s}"
    z = np.load(R + "/predictions_and_scores.npz")
    gt, lab, off = z["gt"], z["labels"], z["pred_off"]
    sc = z["scores_off"].astype(np.float64)
    cl = np.array(CL)
    rank = np.argsort(-sc, 1)
    preds = {k: off.copy() for k in KS}
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    for r in range(len(sc)):
        g = order[bounds[r]:bounds[r + 1]]
        if len(g) == 0:
            continue
        m = maj_class(gt[g])
        if m is None:
            continue
        n_valid += 1
        gt_rank = 1 + int(np.where(cl[rank[r]] == m)[0][0])
        for k in KS:
            if gt_rank <= k:
                preds[k][g] = m
                n_in[k] += 1
    for k in KS:
        mm = metrics(gt, preds[k], CL)
        mac[k].append(mm["mIoU"])
        for c, v in mm["per_class"].items():
            inter[k][c] = inter[k].get(c, 0) + v["inter"]
            union[k][c] = union[k].get(c, 0) + v["union"]
    print("done", s, flush=True)

print()
print("%3s %10s %8s %8s %10s" % ("k", "pooled", "macro", "roots<>", "%of roots"))
for k in KS:
    pooled = np.mean([inter[k][c] / union[k][c] for c in inter[k]]) * 100
    macro = np.mean(mac[k]) * 100
    print("%3d %10.2f %8.2f %8d %9.1f%%" % (k, pooled, macro, n_in[k], 100 * n_in[k] / max(n_valid, 1)))
print("valid roots:", n_valid)