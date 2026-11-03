"""GT-oracle headroom on the ScanNet DeCo (C1+C2) arm. Each oracle picks, per group, the
candidate label that equals the GT majority of that group (if it is among the candidates).
  off            : actual DeCo prediction
  root_top2      : root picks from {main top-1, main top-2}
  root_modes     : root picks from {main top-1} + argmax of each eligible mode (V2 root-level ceiling)
  root_any       : root takes its GT majority (partition ceiling at root level)
  mode_split     : Gaussians grouped by mode footprint; each group picks from {root label, its mode argmax}
  mode_split_any : each mode-footprint group takes its GT majority (partition ceiling at mode level)"""
import sys, json
import numpy as np
import pandas as pd

CODE = "/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code"
REL = "/root/rivermind-data/repro/deco-b123/frozen_ramen_relags"
sys.path[:0] = [CODE, REL, REL + "/ext"]
from eval_scannet_3d import metrics, SUBSETS

B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
S = ("scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 "
     "scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00").split()
CL = list(SUBSETS[19])
ARMS = ["off", "root_top2", "root_modes", "root_any", "mode_split", "mode_split_any"]
inter = {a: {} for a in ARMS}
union = {a: {} for a in ARMS}
mac = {a: [] for a in ARMS}


def majority(g):
    g = g[np.isin(g, CL)]
    if len(g) == 0:
        return None
    v, c = np.unique(g, return_counts=True)
    return v[c.argmax()]


for s in S:
    R = f"{B}/ablation/v2_on_on/{s}"
    E = f"{B}/{s}/export"
    z = np.load(R + "/predictions_and_scores.npz")
    gt, lab, off = z["gt"], z["labels"], z["pred_off"]
    sc = z["scores_off"]
    top = np.array(CL)[np.argsort(-sc, 1)[:, :2]]
    t = pd.read_csv(R + "/SCANNET_M3_WEIGHT_TRACE.csv",
                    usecols=["arm", "class_id", "root_id", "mode_id", "valid", "eligible", "mode_score"])
    t = t[(t.arm == "m1_fixed") & t.valid & t.eligible]
    best = t.loc[t.groupby(["root_id", "mode_id"]).mode_score.idxmax()].set_index(["root_id", "mode_id"]).class_id
    sd = np.load(E + "/observation_semantic_sidecar.npz")
    so, sid, sw = sd["support_offsets"], sd["support_gaussian_ids"], sd["support_weights"]
    modes = {}
    for m in json.load(open(E + "/B1.members.json"))["memberships"]:
        k = (int(m["region_id"]), int(m["slot"]))
        if k in best.index:
            modes.setdefault(k[0], []).append((k[1], m["observation_indices"]))
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    pos = np.full(len(lab), -1)
    preds = {a: off.copy() for a in ARMS}
    for r in range(len(sc)):
        g = order[bounds[r]:bounds[r + 1]]
        if len(g) == 0:
            continue
        maj = majority(gt[g])
        if maj is None:
            continue
        preds["root_any"][g] = maj
        if maj in top[r]:
            preds["root_top2"][g] = maj
        cands = {off[g[0]]} | {best[(r, k)] for k, _ in modes.get(r, [])}
        if maj in cands:
            preds["root_modes"][g] = maj
        ms = modes.get(r, [])
        if len(ms) < 2:
            preds["mode_split"][g] = preds["root_modes"][g]
            preds["mode_split_any"][g] = maj
            continue
        pos[g] = np.arange(len(g))
        sup = np.zeros((len(ms), len(g)), np.float32)
        for j, (_, obs) in enumerate(ms):
            for o in obs:
                ids = sid[so[o]:so[o + 1]]
                p = pos[ids]
                ok = p >= 0
                np.add.at(sup[j], p[ok], sw[so[o]:so[o + 1]][ok])
            sup[j] /= max(len(obs), 1)
        pos[g] = -1
        a = np.where(sup.max(0) > 0, sup.argmax(0), -1)
        for j in range(-1, len(ms)):
            gg = g[a == j]
            if len(gg) == 0:
                continue
            mj = majority(gt[gg])
            if mj is None:
                continue
            preds["mode_split_any"][gg] = mj
            c = {off[g[0]]} if j < 0 else {off[g[0]], best[(r, ms[j][0])]}
            preds["mode_split"][gg] = mj if mj in c else off[g[0]]
    line = [s]
    for a in ARMS:
        m = metrics(gt, preds[a], CL)
        mac[a].append(m["mIoU"])
        for c, v in m["per_class"].items():
            inter[a][c] = inter[a].get(c, 0) + v["inter"]
            union[a][c] = union[a].get(c, 0) + v["union"]
        line.append("%s %.2f" % (a, m["mIoU"] * 100))
    print(" | ".join(line), flush=True)

print()
for a in ARMS:
    p = np.mean([inter[a][c] / union[a][c] for c in inter[a] if union[a][c] > 0]) * 100
    print("%-15s pooled %.2f macro %.2f" % (a, p, np.mean(mac[a]) * 100))
