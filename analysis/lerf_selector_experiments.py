"""LERF-OVS step-2 reranker experiments (leave-one-scene-out).

Candidate-level features -> GBM regression on candidate IoU; pick the argmax per
query (single-label selection). Compared against the released gap selection
(main, multi-label), the rank-1 candidate (no rerank), and the per-query oracle.
Also a conservative gate: switch away from rank-1 only if the predicted IoU gain
exceeds tau (tau picked on the training scenes' out-of-fold predictions).
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

OUT = "/root/rivermind-data/repro/deco-b123/lerf_selector_20261030"
SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
KEY = ["scene", "image", "prompt"]

df = pd.concat([pd.read_csv(os.path.join(OUT, "%s_candidates.csv" % s)) for s in SCENES],
               ignore_index=True)
df["log_n_points"] = np.log1p(df["n_points"])
FEATS = ["score", "deficit", "rank", "n_cands", "is_leaf", "root_score",
         "log_n_points", "n_points_ratio", "n_leaf", "leaf_gain", "area_ratio", "gap12"]


def per_query(df_):
    g = df_.groupby(KEY)
    out = pd.DataFrame({
        "main": g["main_iou"].first(),
        "oracle": g["iou"].max(),
        "n": g.size(),
    })
    r1 = df_[df_["rank"] == 1].set_index(KEY)["iou"]
    out = out.join(r1.rename("rank1"))
    return out.reset_index()


# out-of-fold predictions
oof = {}
for s in SCENES:
    tr = df[df.scene != s]
    te = df[df.scene == s].copy()
    mdl = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05,
                                        min_samples_leaf=30)
    mdl.fit(tr[FEATS].values, tr["iou"].values)
    te["pred"] = mdl.predict(te[FEATS].values)
    oof[s] = te
    ord_rows = []
    for key, g in te.groupby(KEY):
        g = g.sort_values("rank")
        perm = np.argsort(-g["pred"].values)
        ord_rows.append(dict(scene=key[0], image=key[1], prompt=key[2],
                             order=" ".join(str(int(g["rank"].values[j])) for j in perm)))
    pd.DataFrame(ord_rows).to_csv(os.path.join(OUT, "%s_orders.csv" % s), index=False)
    print("fit fold", s, flush=True)


def choose(te, tau):
    recs = []
    for key, g in te.groupby(KEY):
        g = g.reset_index(drop=True)
        j = int(np.argmax(g["pred"].values))
        r1 = int(np.where(g["rank"].values == 1)[0][0])
        pick = j if (j != r1 and g["pred"].values[j] - g["pred"].values[r1] > tau) else r1
        recs.append(dict(scene=key[0], image=key[1], prompt=key[2],
                         iou=float(g["iou"].values[pick]),
                         switched=int(pick != r1)))
    return pd.DataFrame(recs)


gate_tau = {}
for s in SCENES:
    scores = {}
    for tau in TAUS:
        vals = []
        for t in SCENES:
            if t == s:
                continue
            ch = choose(oof[t], tau)
            vals.append(ch["iou"].mean())
        scores[tau] = float(np.mean(vals))
    gate_tau[s] = max(TAUS, key=lambda t: scores[t])

rows = []
per_scene = {}
for s in SCENES:
    q = per_query(oof[s])
    sel = choose(oof[s], 0.0)
    gat = choose(oof[s], gate_tau[s])
    q = q.merge(sel.rename(columns={"iou": "selector", "switched": "sel_sw"}),
                on=KEY, how="left")
    q = q.merge(gat.rename(columns={"iou": "gate", "switched": "gate_sw"}),
                on=KEY, how="left")
    rr_path = os.path.join(OUT, "%s_rerank.csv" % s)
    if os.path.exists(rr_path):
        q = q.merge(pd.read_csv(rr_path)[KEY + ["rerank_iou"]], on=KEY, how="left")
        rerank_mean = float(q.rerank_iou.mean())
    else:
        rerank_mean = float("nan")
    per_scene[s] = dict(
        n=int(len(q)), main=float(q.main.mean()), rank1=float(q.rank1.mean()),
        selector=float(q.selector.mean()), gate=float(q.gate.mean()),
        rerank=rerank_mean,
        oracle=float(q.oracle.mean()),
        sel_switch_rate=float(q.sel_sw.mean()),
        gate_switch_rate=float(q.gate_sw.mean()), gate_tau=gate_tau[s])
    rows.append(per_scene[s])
    q.to_csv(os.path.join(OUT, "%s_per_query.csv" % s), index=False)

summ = {k: dict(main=np.mean([r["main"] for r in rows]),
                rank1=np.mean([r["rank1"] for r in rows]),
                selector=np.mean([r["selector"] for r in rows]),
                gate=np.mean([r["gate"] for r in rows]),
                rerank=np.nanmean([r.get("rerank", np.nan) for r in rows]),
                oracle=np.mean([r["oracle"] for r in rows])) for k in ["all"]}[ "all"]
for s, r in per_scene.items():
    print(s, {k: round(v, 4) for k, v in r.items() if k != "gate_tau"}, "tau", r["gate_tau"])
print("ALL", {k: round(v, 4) for k, v in summ.items()})

with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(dict(per_scene=per_scene, overall=summ, gate_tau=gate_tau),
              f, ensure_ascii=False, indent=2, default=float)


def tab(head, body):
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in body])


L = []
L.append("# LERF-OVS 第 2 步：候选重排器实验（leave-one-scene-out）")
L.append("")
L.append("- 协议：4 折，每折 3 个场景训练、第 4 个评估；候选 = released top-5 pool；标签 = 逐候选渲染 IoU。")
L.append("- 选择器：GBM 回归候选 IoU，每查询取 argmax（单候选选择）；gate 变体为预测增益超过 tau 才从 rank-1 切换。")
L.append("")
L.append("## 主结果（逐查询平均 IoU）")
L.append("")
L.append(tab(["方法", "平均 IoU"], [
    ["DeCo main（gap 多标签）", "%.4f" % summ["main"]],
    ["rank-1 单候选（不重排）", "%.4f" % summ["rank1"]],
    ["选择器（重排后取 argmax）", "%.4f" % summ["selector"]],
    ["选择器 + gate", "%.4f" % summ["gate"]],
    ["重排 + gap（多标签）", "%.4f" % summ["rerank"]],
    ["oracle 单候选上限", "%.4f" % summ["oracle"]],
]))
L.append("")
L.append("## 逐场景")
L.append("")
L.append(tab(["场景", "查询数", "main", "rank1", "选择器", "gate", "重排+gap", "oracle", "切换率"],
             [[s, r["n"], "%.4f" % r["main"], "%.4f" % r["rank1"],
               "%.4f" % r["selector"], "%.4f" % r["gate"], "%.4f" % r.get("rerank", float("nan")),
               "%.4f" % r["oracle"],
               "%.1f%% / %.1f%%" % (100 * r["sel_switch_rate"], 100 * r["gate_switch_rate"])]
              for s, r in per_scene.items()]))
L.append("")
with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))