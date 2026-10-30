"""Why do the hard LERF-OVS queries stay hard?

Uses the step-0 attribution join and the per-candidate dump to split every query
into failure families and profile the mechanism:
  recall_fail : no candidate in the top-5 pool reaches IoU 0.3 (object missing /
                fragmented at candidate-generation time)
  found       : the best candidate is rank-1 (ranking already fine)
  rank2       : the best candidate is the runner-up
  rank_deep   : the best candidate sits at rank >= 3
Also measures, per family, the score deficit of the best candidate, its size and
leaf/root level, how much of the gap best-prefix truncation could recover, and how
much the V2 per-query-best coefficient recovers (the +1.96 ceiling).
"""
import os
import pandas as pd
import numpy as np

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
A = os.path.join(R, "lerf_attribution_20261030")
C = os.path.join(R, "lerf_selector_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
KEY = ["scene", "image", "prompt"]

q = pd.read_csv(os.path.join(A, "per_query_joined.csv"))
cand = pd.concat([pd.read_csv(os.path.join(C, "%s_candidates.csv" % s)) for s in SC],
                 ignore_index=True)

best_rows = []
for key, g in cand.groupby(KEY):
    g = g.reset_index(drop=True)
    j = int(g["iou"].idxmax())
    top1 = g.loc[g["rank"] == 1].iloc[0]
    best_rows.append(dict(
        scene=key[0], image=key[1], prompt=key[2],
        best_iou=float(g.loc[j, "iou"]), best_rank=int(g.loc[j, "rank"]),
        best_deficit=float(top1["score"] - g.loc[j, "score"]),
        best_area=float(g.loc[j, "area_ratio"]),
        best_is_leaf=int(g.loc[j, "is_leaf"]),
        best_np_ratio=float(g.loc[j, "n_points_ratio"]),
        top1_area=float(top1["area_ratio"]),
        n_cands=int(len(g)),
        any_leaf=int(g["is_leaf"].max()),
    ))
best = pd.DataFrame(best_rows)
q = q.merge(best, on=KEY, how="left")

V2 = ["v2_c0", "v2_c0.25", "v2_c0.5", "v2_c1", "v2_c2", "v2_c4"]
O = os.path.join(R, "lerf_oracle_20261030")
orac = pd.concat([pd.read_csv(os.path.join(O, "%s_oracle.csv" % s)) for s in SC],
                 ignore_index=True)
q = q.drop(columns=[c for c in ("v2_c0",) if c in q.columns])
q = q.merge(orac[KEY + V2], on=KEY, how="left")
q["v2_best"] = q[V2].max(axis=1)
q["v2_gain"] = q["v2_best"] - q["main"]
q["prefix_gain"] = q["prefix_best"] - q["main"]
q["combo"] = q["greedy5"] - q["best_iou"]


def family(r):
    if r["best_iou"] < 0.3:
        return "recall_fail"
    if r["best_rank"] == 1:
        return "found"
    if r["best_rank"] == 2:
        return "rank2"
    return "rank_deep"


q["family"] = q.apply(family, axis=1)

order = ["found", "rank2", "rank_deep", "recall_fail"]
tab = q.groupby("family").agg(
    n=("main", "size"),
    main_mean=("main", "mean"),
    best_iou_mean=("best_iou", "mean"),
    deficit_med=("best_deficit", "median"),
    area_med=("best_area", "median"),
    leaf_share=("best_is_leaf", "mean"),
    ncands_med=("n_cands", "median"),
    prefix_gain_mean=("prefix_gain", "mean"),
    prefix_fixable=("prefix_gain", lambda x: float((x > 0.05).mean())),
    v2_gain_mean=("v2_gain", "mean"),
    combo_mean=("combo", "mean"),
).reindex(order)
print("== 失败族统计 ==")
print(tab.round(3).to_string())

print("\n== 每族中，最优截断能救回多少（per-query mean）==")
print(q.groupby("family")[["main", "prefix_best", "v2_best", "greedy5", "best_iou"]]
      .mean().reindex(order).round(3).to_string())

rank_stats = []
for r in range(1, 8):
    sub = cand[cand["rank"] == r]
    rank_stats.append(dict(rank=r, n=len(sub), iou_mean=sub["iou"].mean(),
                           area_med=sub["area_ratio"].median(),
                           score_med=sub["score"].median(),
                           leaf_share=sub["is_leaf"].mean()))
print("\n== 候选排名 → IoU ==")
print(pd.DataFrame(rank_stats).round(3).to_string(index=False))

worst = q.sort_values("main").head(15)[
    ["scene", "prompt", "main", "best_iou", "best_rank", "best_deficit",
     "best_area", "n_cands", "family"]]
print("\n== 最差 15 条查询 ==")
print(worst.round(3).to_string(index=False))

print("\n== 各场景族分布 ==")
print(pd.crosstab(q["scene"], q["family"])[order].to_string())

print("\n== 组合需求（greedy5 - best_iou > 0.1）==")
combo_q = q[q["combo"] > 0.1]
print("count", len(combo_q), "of", len(q), "; by family:",
      combo_q["family"].value_counts().to_dict())
print("mean combo gain", round(combo_q["combo"].mean(), 3))

q.to_csv(os.path.join(A, "hard_query_diagnosis.csv"), index=False)
print("\nsaved", os.path.join(A, "hard_query_diagnosis.csv"))