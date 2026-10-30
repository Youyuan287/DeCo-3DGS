"""Diagnose what the selector gain actually is: more correct roots, or label
redistribution across classes (which inflates mean-per-class IoU without better
decisions). Reports root-level accuracy (pred == GT majority) for off / A / D,
per-class IoU deltas, and per-scene accuracy, all under the same LOSO protocol.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CODE = "/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code"
REL = "/root/rivermind-data/repro/deco-b123/frozen_ramen_relags"
sys.path[:0] = [CODE, REL, REL + "/ext"]
from eval_scannet_3d import metrics

B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
FEAT = "/root/rivermind-data/repro/deco-b123/step2_selector_20261030/features"
OUT = "/root/rivermind-data/repro/deco-b123/step2_selector_20261030"
S = ("scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 "
     "scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00").split()
CL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
A = ["s", "deficit", "rank", "is_top1"]
D = ["s", "deficit", "rank", "is_top1", "root_n_gauss", "root_mean_z",
     "n_vote", "adv_max", "ms_max", "rel_max", "wadv",
     "cov", "fp_views", "fp_vis", "fp_modes"]

dfs, zs = {}, {}
for s in S:
    dfs[s] = pd.read_csv(os.path.join(FEAT, "%s_features.csv" % s))
    zs[s] = np.load(f"{B}/ablation/v2_on_on/{s}/predictions_and_scores.npz")


def apply_choice(s, choice):
    z = zs[s]
    GT, LAB, OFF = z["gt"], z["labels"], z["pred_off"]
    pred = OFF.copy()
    order = np.argsort(LAB, kind="stable")
    bounds = np.searchsorted(LAB[order], np.arange(LAB.max() + 2))
    for r, c in choice.items():
        pred[order[bounds[r]:bounds[r + 1]]] = c
    return pred


def root_acc(df, pred, z):
    hor = np.argsort(z["labels"], kind="stable")
    b = np.searchsorted(z["labels"][hor], np.arange(z["labels"].max() + 2))
    ok = tot = 0
    for r, g in df.groupby("root"):
        maj = int(g["gt_majority"].iloc[0])
        if maj < 0:
            continue
        gg = hor[b[r]:b[r + 1]]
        tot += 1
        ok += int(pred[gg[0]] == maj)
    return ok, tot

acc = {}
for name, feats in [("A_ranking", A), ("D_footprint", D)]:
    acc[name] = {}
    per_scene = {}
    for s in S:
        train = pd.concat([dfs[t] for t in S if t != s], ignore_index=True)
        tr = train[train.y >= 0]
        mdl = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.05, min_samples_leaf=40)
        mdl.fit(tr[feats].values.astype(np.float64), tr["y"].values.astype(int))
        te = dfs[s]
        df = te.copy()
        df["proba"] = mdl.predict_proba(te[feats].values.astype(np.float64))[:, 1]
        choice = {}
        for r, g in df.groupby("root"):
            choice[int(r)] = int(g["cls"].values[int(np.argmax(g["proba"].values))])
        pred = apply_choice(s, choice)
        m = metrics(zs[s]["gt"], pred, CL)
        ok, tot = root_acc(te, pred, zs[s])
        per_scene[s] = dict(ok=ok, tot=tot, acc=ok / max(tot, 1), miou=m["mIoU"] * 100,
                            switched=sum(1 for r, g in te.groupby("root")
                                         if choice[int(r)] != int(g.loc[g["is_top1"] == 1, "cls"].iloc[0])))
        print(name, s, per_scene[s], flush=True)
    acc[name] = per_scene

off_scene, off_all = {}, {}
for s in S:
    z = zs[s]
    hor = np.argsort(z["labels"], kind="stable")
    b = np.searchsorted(z["labels"][hor], np.arange(z["labels"].max() + 2))
    te = dfs[s]
    ok = tot = 0
    for r, g in te.groupby("root"):
        maj = int(g["gt_majority"].iloc[0])
        if maj < 0:
            continue
        tot += 1
        ok += int(z["pred_off"][hor[b[r]:b[r + 1]][0]] == maj)
    off_scene[s] = dict(ok=ok, tot=tot, acc=ok / max(tot, 1))

def pclass_metrics(feats, name):
    inter, union = {}, {}
    for s in S:
        train = pd.concat([dfs[t] for t in S if t != s], ignore_index=True)
        tr = train[train.y >= 0]
        mdl = HistGradientBoostingClassifier(max_depth=3, max_iter=250,
                                             learning_rate=0.05, min_samples_leaf=40)
        mdl.fit(tr[feats].values.astype(np.float64), tr["y"].values.astype(int))
        te = dfs[s]
        df = te.copy()
        df["proba"] = mdl.predict_proba(te[feats].values.astype(np.float64))[:, 1]
        choice = {}
        for r, g in df.groupby("root"):
            choice[int(r)] = int(g["cls"].values[int(np.argmax(g["proba"].values))])
        pred = apply_choice(s, choice)
        m = metrics(zs[s]["gt"], pred, CL)
        for c, v in m["per_class"].items():
            inter[c] = inter.get(c, 0) + v["inter"]
            union[c] = union.get(c, 0) + v["union"]
    return {c: inter[c] / union[c] for c in inter}

D_cls = pclass_metrics(D, "D")
off_inter, off_union = {}, {}
for s in S:
    m = metrics(zs[s]["gt"], zs[s]["pred_off"], CL)
    for c, v in m["per_class"].items():
        off_inter[c] = off_inter.get(c, 0) + v["inter"]
        off_union[c] = off_union.get(c, 0) + v["union"]
off_cls = {c: off_inter[c] / off_union[c] for c in off_inter}

rows = []
for c in sorted(D_cls):
    rows.append(dict(cls=c, off_iou=off_cls[c] * 100, D_iou=D_cls[c] * 100,
                     delta=(D_cls[c] - off_cls[c]) * 100))
dc = pd.DataFrame(rows).sort_values("delta", ascending=False)
dc.to_csv(OUT + "/class_delta.csv", index=False)

summ = {name: {s: v for s, v in acc[name].items()} for name in acc}
summ["off"] = off_scene
with open(OUT + "/accuracy_diag.json", "w", encoding="utf-8") as f:
    json.dump(summ, f, ensure_ascii=False, indent=2, default=float)

for name in ["off", "A_ranking", "D_footprint"]:
    tot_ok = sum(v["ok"] for v in summ[name].values())
    tot_n = sum(v["tot"] for v in summ[name].values())
    print(name, "root acc %.3f (%d/%d)" % (tot_ok / max(tot_n, 1), tot_ok, tot_n))
print(dc.head(8).to_string(index=False))
print("worst:", dc.tail(5).to_string(index=False))