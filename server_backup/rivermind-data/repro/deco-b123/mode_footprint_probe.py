"""Feasibility probe: can B1 modes be mapped to Gaussians and labelled at mode level?
Single pre-declared rule, no parameters tuned on GT:
  a Gaussian in a root with >=2 eligible modes goes to the mode with the largest mean support weight
  over the observations of that mode, and takes the argmax class of that mode (C1+C2 scores).
  Unsupported Gaussians keep the root label."""
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
CL = SUBSETS[19]
inter = {"off": {}, "mode": {}}
union = {"off": {}, "mode": {}}
mac = {"off": [], "mode": []}
cov, nondom, jac = [], [], []
split_roots = tot_roots = multi_roots = moved = NG = 0

for s in S:
    R = f"{B}/ablation/v2_on_on/{s}"
    E = f"{B}/{s}/export"
    z = np.load(R + "/predictions_and_scores.npz")
    gt, lab, pred_off = z["gt"], z["labels"], z["pred_off"]
    NG += len(lab)
    sd = np.load(E + "/observation_semantic_sidecar.npz")
    so, sid, sw = sd["support_offsets"], sd["support_gaussian_ids"], sd["support_weights"]
    mem = json.load(open(E + "/B1.members.json"))["memberships"]
    t = pd.read_csv(R + "/SCANNET_M3_WEIGHT_TRACE.csv",
                    usecols=["arm", "class_id", "root_id", "mode_id", "valid", "eligible", "mode_score"])
    t = t[(t.arm == "m1_fixed") & t.valid & t.eligible]
    best = t.loc[t.groupby(["root_id", "mode_id"]).mode_score.idxmax()].set_index(["root_id", "mode_id"]).class_id
    modes = {}
    for m in mem:
        k = (int(m["region_id"]), int(m["slot"]))
        if k in best.index:
            modes.setdefault(k[0], []).append((k[1], m["observation_indices"]))
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    pos = np.full(len(lab), -1)
    pred = pred_off.copy()
    for r, ms in modes.items():
        tot_roots += 1
        if len(ms) < 2:
            continue
        multi_roots += 1
        g = order[bounds[r]:bounds[r + 1]]
        pos[g] = np.arange(len(g))
        sup = np.zeros((len(ms), len(g)), np.float32)
        for j, (_, obs) in enumerate(ms):
            for o in obs:
                ids = sid[so[o]:so[o + 1]]
                w = sw[so[o]:so[o + 1]]
                p = pos[ids]
                ok = p >= 0
                np.add.at(sup[j], p[ok], w[ok])
            sup[j] /= max(len(obs), 1)
        pos[g] = -1
        fp = sup > 0
        covered = fp.any(0)
        cov.append(covered.mean())
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                u = (fp[i] | fp[j]).sum()
                jac.append((fp[i] & fp[j]).sum() / u if u else 0.0)
        a = sup.argmax(0)
        cls = np.array([best[(r, k)] for k, _ in ms])
        newl = np.where(covered, cls[a], pred_off[g])
        if covered.any():
            dom = np.bincount(a[covered], minlength=len(ms)).argmax()
            nondom.append(((a != dom) & covered).mean())
        if len(np.unique(newl)) > 1:
            split_roots += 1
        moved += int((newl != pred_off[g]).sum())
        pred[g] = newl
    for k, p in (("off", pred_off), ("mode", pred)):
        m = metrics(gt, p, CL)
        mac[k].append(m["mIoU"])
        for c, v in m["per_class"].items():
            inter[k][c] = inter[k].get(c, 0) + v["inter"]
            union[k][c] = union[k].get(c, 0) + v["union"]
    print(s, "off %.4f mode-level %.4f" % (mac["off"][-1] * 100, mac["mode"][-1] * 100), flush=True)

q = lambda v: np.round(np.quantile(v, [.1, .5, .9]), 3).tolist()
print("roots with eligible modes", tot_roots, "| with >=2 modes", multi_roots,
      "| split into >1 label", split_roots, "| gaussians relabelled %d / %d" % (moved, NG))
print("support coverage of root gaussians q", q(cov))
print("non-dominant-mode gaussian share q", q(nondom))
print("pairwise mode footprint Jaccard q", q(jac))
for k in ("off", "mode"):
    p = np.mean([inter[k][c] / union[k][c] for c in inter[k] if union[k][c] > 0]) * 100
    print(k, "pooled %.4f macro %.4f" % (p, np.mean(mac[k]) * 100))
