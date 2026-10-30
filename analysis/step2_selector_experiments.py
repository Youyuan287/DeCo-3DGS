"""ScanNet step-2 selector experiments with leave-one-scene-out.

Methods compared on the certified DeCo C1+C2 arm:
  off          : current DeCo (per-root argmax of the main scores)
  lam          : evidence rerank, score_c = s_c + lam * reliability-weighted mode adv,
                 lam chosen on the 9 training scenes only
  logit / gbm  : discriminative reranker over the candidate features, trained
                 per fold on the other 9 scenes (rows = (root, class), label = GT match)
  oracle_top5  : candidate-set ceiling using only main-score top-5
  oracle_cand  : candidate-set ceiling using top-5 plus mode-argmax classes
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
FEATS = ["s", "deficit", "rank", "is_top1", "n_vote", "adv_max", "ms_max",
         "rel_max", "wadv", "cov", "fp_views", "fp_vis", "fp_modes",
         "root_n_gauss", "root_mean_z"]
LAMS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

dfs, zs = {}, {}
for s in S:
    dfs[s] = pd.read_csv(os.path.join(FEAT, "%s_features.csv" % s))
    zs[s] = np.load(f"{B}/ablation/v2_on_on/{s}/predictions_and_scores.npz")


def root_tables(df):
    tab = {}
    for r, g in df.groupby("root", sort=True):
        tab[int(r)] = (g["cls"].values.astype(np.int64),
                       g[FEATS].values.astype(np.float64))
    return tab


TAB = {s: root_tables(dfs[s]) for s in S}


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


i_s, i_w = FEATS.index("s"), FEATS.index("wadv")

lam_stats = {}
for lam in LAMS:
    lam_stats[lam] = {}
    for s in S:
        choice = {}
        for r, (cls, X) in TAB[s].items():
            sc_ = X[:, i_s] + lam * X[:, i_w]
            choice[r] = int(cls[int(np.argmax(sc_))])
        lam_stats[lam][s] = apply_choice(s, choice)

off_stats = {s: metrics(zs[s]["gt"], zs[s]["pred_off"], CL) for s in S}

oracle = {"oracle_top5": {}, "oracle_cand": {}}
for s in S:
    df = dfs[s]
    ch5, chc = {}, {}
    for r, g in df.groupby("root"):
        maj = int(g["gt_majority"].iloc[0])
        top1 = int(g.loc[g["is_top1"] == 1, "cls"].iloc[0])
        cands = g["cls"].values
        cands5 = g.loc[g["rank"] <= 5, "cls"].values
        ch5[int(r)] = maj if (maj >= 0 and maj in cands5) else top1
        chc[int(r)] = maj if (maj >= 0 and maj in cands) else top1
    oracle["oracle_top5"][s] = apply_choice(s, ch5)
    oracle["oracle_cand"][s] = apply_choice(s, chc)

res = {m: {} for m in ["off", "lam", "logit", "gbm", "oracle_top5", "oracle_cand"]}
for s in S:
    res["off"][s] = off_stats[s]
    res["oracle_top5"][s] = oracle["oracle_top5"][s]
    res["oracle_cand"][s] = oracle["oracle_cand"][s]

fold_info = []
switch_info = {"logit": [0, 0, 0], "gbm": [0, 0, 0]}  # switched, switched_correct, n_roots

for s in S:
    train = pd.concat([dfs[t] for t in S if t != s], ignore_index=True)
    tr = train[train.y >= 0]
    Xtr, ytr = tr[FEATS].values.astype(np.float64), tr["y"].values.astype(int)

    best_lam, best_pool = 0.0, -1.0
    for lam in LAMS:
        pool = pooled_of([lam_stats[lam][t] for t in S if t != s])
        if pool > best_pool + 1e-9:
            best_lam, best_pool = lam, pool
    res["lam"][s] = lam_stats[best_lam][s]

    models = {}
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced"))
    logit.fit(Xtr, ytr)
    models["logit"] = logit
    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                         min_samples_leaf=40)
    gbm.fit(Xtr, ytr)
    models["gbm"] = gbm

    for name, mdl in models.items():
        choice = {}
        proba = mdl.predict_proba(dfs[s][FEATS].values.astype(np.float64))[:, 1]
        df = dfs[s].copy()
        df["proba"] = proba
        for r, g in df.groupby("root"):
            j = int(np.argmax(g["proba"].values))
            choice[int(r)] = int(g["cls"].values[j])
            top1 = int(g.loc[g["is_top1"] == 1, "cls"].iloc[0])
            maj = int(g["gt_majority"].iloc[0])
            if choice[int(r)] != top1:
                switch_info[name][0] += 1
                switch_info[name][2] += 1
                if maj >= 0 and choice[int(r)] == maj:
                    switch_info[name][1] += 1
        res[name][s] = apply_choice(s, choice)
    fold_info.append(dict(scene=s, lam=best_lam))
    print("fold", s, "lam", best_lam, flush=True)

rows = []
for m in res:
    per = {s: res[m][s]["mIoU"] * 100 for s in S}
    pooled = pooled_of([res[m][s] for s in S])
    macro = float(np.mean(list(per.values())))
    rows.append(dict(method=m, pooled=pooled, macro=macro, **per))
resdf = pd.DataFrame(rows)
resdf.to_csv(OUT + "/method_results.csv", index=False)

sw_rows = []
for name, (sw, ok, n) in switch_info.items():
    sw_rows.append(dict(method=name, switched=sw, switched_correct=ok,
                        switch_precision=ok / max(sw, 1)))
pd.DataFrame(sw_rows).to_csv(OUT + "/switch_stats.csv", index=False)

summary = {
    "methods": {r["method"]: {"pooled": r["pooled"], "macro": r["macro"]} for r in rows},
    "folds": fold_info,
    "switches": sw_rows,
}


def tab(head, body):
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in body])


L = []
L.append("# ScanNet 第 2 步：选择器原型实验结果（leave-one-scene-out）")
L.append("")
L.append("- 数据：认证 DeCo C1+C2（off = 55.04 / 50.18），10 个场景，LOSO：每折在 9 个场景上训练/选参，第 10 个场景评估。")
L.append("- 候选池：主分数 top-5 ∪ 全部 eligible 模式的 argmax 类。")
L.append("- 特征（全部无 GT、无场景先验）：主分、deficit、rank、is_top1、模式票数、mode_score 最大值、advantage 最大值、M2 reliability、可靠性加权 advantage、足迹覆盖率、视图数、可见性、足迹模式数、root 高斯数、root 平均高度。")
L.append("")
L.append("## 主结果")
L.append("")
L.append(tab(["方法", "pooled", "macro", "vs off (pooled)"], [
    [r["method"], "%.2f" % r["pooled"], "%.2f" % r["macro"],
     "%+.2f" % (r["pooled"] - resdf.loc[resdf.method == "off", "pooled"].iloc[0])]
    for r in rows]))
L.append("")
L.append("## 逐场景（off → gbm）")
L.append("")
L.append(tab(["场景", "off", "logit", "gbm", "lam", "oracle_cand"],
             [[s,
               "%.2f" % (res["off"][s]["mIoU"] * 100),
               "%.2f" % (res["logit"][s]["mIoU"] * 100),
               "%.2f" % (res["gbm"][s]["mIoU"] * 100),
               "%.2f" % (res["lam"][s]["mIoU"] * 100),
               "%.2f" % (res["oracle_cand"][s]["mIoU"] * 100)] for s in S]))
L.append("")
L.append("## 每折 lambda 与切换统计")
L.append("")
L.append(tab(["折", "lambda"], [[f["scene"], f["lam"]] for f in fold_info]))
L.append("")
L.append(tab(["方法", "切换 root 数", "切换后正确", "切换精度"],
             [[r["method"], r["switched"], r["switched_correct"],
               "%.1f%%" % (100 * r["switch_precision"])] for r in sw_rows]))
L.append("")
with open(OUT + "/summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
with open(OUT + "/stats.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

print(resdf[["method", "pooled", "macro"]].to_string(index=False))
print("switches:", sw_rows)