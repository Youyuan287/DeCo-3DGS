"""Feature-set ablation and conservative-gate variant for the ScanNet selector.

Guards against a selector that only learns "switch aggressively": we compare four
nested feature sets (ranking-only -> +geometry -> +mode evidence -> +footprint),
then add a switch gate delta (only rerank when the challenger's probability beats
the incumbent by delta, delta tuned on the training scenes). Switch diagnostics
(source -> target class pairs) show what the full model actually changes.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
SETS = {
    "A_ranking": ["s", "deficit", "rank", "is_top1"],
    "B_geometry": ["s", "deficit", "rank", "is_top1", "root_n_gauss", "root_mean_z"],
    "C_mode": ["s", "deficit", "rank", "is_top1", "root_n_gauss", "root_mean_z",
               "n_vote", "adv_max", "ms_max", "rel_max", "wadv"],
    "D_footprint": ["s", "deficit", "rank", "is_top1", "root_n_gauss", "root_mean_z",
                    "n_vote", "adv_max", "ms_max", "rel_max", "wadv",
                    "cov", "fp_views", "fp_vis", "fp_modes"],
}
DELTAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

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
    return metrics(GT, pred, CL)


def pooled_of(ms):
    inter, union = {}, {}
    for m in ms:
        for c, v in m["per_class"].items():
            inter[c] = inter.get(c, 0) + v["inter"]
            union[c] = union.get(c, 0) + v["union"]
    return float(np.mean([inter[c] / union[c] for c in inter])) * 100


def fit_gbm(X, y):
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                       min_samples_leaf=40)
    m.fit(X, y)
    return m


def choose_with_delta(df, proba, delta):
    df = df.copy()
    df["proba"] = proba
    choice = {}
    for r, g in df.groupby("root"):
        cls = g["cls"].values
        pr = g["proba"].values
        top1 = int(g.loc[g["is_top1"] == 1, "cls"].iloc[0])
        j = int(np.argmax(pr))
        best = int(cls[j])
        p_best = float(pr[j])
        p_top1 = float(pr[int(np.where(cls == top1)[0][0])])
        choice[int(r)] = best if (best != top1 and p_best - p_top1 > delta) else top1
    return choice


res = {k: {} for k in SETS}
switch_pairs = {}
best_models = {}

for s in S:
    train = pd.concat([dfs[t] for t in S if t != s], ignore_index=True)
    tr = train[train.y >= 0]
    te = dfs[s]
    for name, feats in SETS.items():
        Xtr, ytr = tr[feats].values.astype(np.float64), tr["y"].values.astype(int)
        mdl = fit_gbm(Xtr, ytr)
        proba = mdl.predict_proba(te[feats].values.astype(np.float64))[:, 1]
        choice = choose_with_delta(te, proba, 0.0)
        if name == "D_footprint":
            best_models[s] = (mdl, proba)
            for r, g in te.groupby("root"):
                top1 = int(g.loc[g["is_top1"] == 1, "cls"].iloc[0])
                ch = choice[int(r)]
                if ch != top1:
                    switch_pairs[(top1, ch)] = switch_pairs.get((top1, ch), 0) + 1
        res[name][s] = apply_choice(s, choice)
    print("fold", s, flush=True)

gate = {}
for s in S:
    train = pd.concat([dfs[t] for t in S if t != s], ignore_index=True)
    tr = train[train.y >= 0]
    feats = SETS["D_footprint"]
    mdl = fit_gbm(tr[feats].values.astype(np.float64), tr["y"].values.astype(int))
    te = dfs[s]
    proba = mdl.predict_proba(te[feats].values.astype(np.float64))[:, 1]
    opts = {}
    for d in DELTAS:
        opts[d] = apply_choice(s, choose_with_delta(te, proba, d))
    gate[s] = opts
    print("gate fold", s, flush=True)

rows = []
for name in SETS:
    per = {s: res[name][s]["mIoU"] * 100 for s in S}
    rows.append(dict(method=name, pooled=pooled_of([res[name][s] for s in S]),
                     macro=float(np.mean(list(per.values()))), **per))
best_d = {}
for s in S:
    def train_pool(d):
        return np.mean([gate[t][d]["mIoU"] for t in S if t != s]) * 100
    best_d[s] = max(DELTAS, key=train_pool)
g_per = {s: gate[s][best_d[s]]["mIoU"] * 100 for s in S}
rows.append(dict(method="D_delta_gate",
                 pooled=pooled_of([gate[s][best_d[s]] for s in S]),
                 macro=float(np.mean(list(g_per.values()))), **g_per))
resdf = pd.DataFrame(rows)
resdf.to_csv(OUT + "/ablation_results.csv", index=False)

sp = pd.DataFrame([dict(src=int(a), dst=int(b), n=int(c))
                   for (a, b), c in sorted(switch_pairs.items(), key=lambda kv: -kv[1])])
sp.to_csv(OUT + "/switch_pairs.csv", index=False)

summary = {
    "ablation": {r["method"]: {"pooled": r["pooled"], "macro": r["macro"]} for r in rows},
    "gate_delta": {s: best_d[s] for s in S},
    "top_switch_pairs": sp.head(12).to_dict("records"),
}
with open(OUT + "/ablation_stats.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

print(resdf[["method", "pooled", "macro"]].to_string(index=False))
print("gate deltas:", best_d)
print(sp.head(12).to_string(index=False))