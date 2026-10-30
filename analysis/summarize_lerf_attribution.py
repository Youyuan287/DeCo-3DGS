"""Summarize the LERF-OVS step-0 attribution run.

Joins <scene>_attribution.csv with the earlier <scene>_oracle.csv (main /
prefix_best must match), then reports the gap-error composition:
ok_gap / cut_early (p* > g) / cut_late (p* < g), gap-length vs best-prefix
offsets, ranking quality (is the best single candidate the rank-1 one), and the
combination gain (greedy5 over prefix_best). Writes summary.md + stats.json.
"""
import os, sys, json
import numpy as np
import pandas as pd

ATTR = sys.argv[1]
ORACLE = sys.argv[2]
OUT = sys.argv[3]
SCENES = ["figurines", "ramen", "teatime", "waldo_kitchen"]
EPS = 1e-9

a = pd.concat([pd.read_csv(os.path.join(ATTR, "%s_attribution.csv" % s)) for s in SCENES],
              ignore_index=True)
o = pd.concat([pd.read_csv(os.path.join(ORACLE, "%s_oracle.csv" % s)) for s in SCENES],
              ignore_index=True)
m = a.merge(o[["scene", "image", "prompt", "main", "prefix_best", "greedy5",
               "single_best", "greedy20"]],
            on=["scene", "image", "prompt"], how="left", suffixes=("", "_ref"))
m["d_main"] = (m.main - m.main_ref).abs()
m["d_prefix"] = (m.prefix_best - m.prefix_best_ref).abs()

N = len(m)
cnt = m.cat.value_counts()
g = m.groupby("cat")

def pct(x):
    return 100.0 * x / N

stats = dict(
    n_queries=N,
    consistency_max_diff_main=float(m.d_main.max()),
    consistency_max_diff_prefix=float(m.d_prefix.max()),
    counts={k: int(cnt.get(k, 0)) for k in ["ok_gap", "cut_early", "cut_late"]},
    mean_main=float(m.main.mean()), mean_prefix_best=float(m.prefix_best.mean()),
    mean_main_by_cat=dict(g.main.mean()), mean_prefix_by_cat=dict(g.prefix_best.mean()),
    empty_gap=int((m.gap_len == 0).sum()),
    cut_early_offset={str(k): int(v) for k, v in (m[m.cat == "cut_early"].bestk - m[m.cat == "cut_early"].gap_len).value_counts().sort_index().items()},
    cut_late_offset={str(k): int(v) for k, v in (m[m.cat == "cut_late"].gap_len - m[m.cat == "cut_late"].bestk).value_counts().sort_index().items()},
    best_single_rank={str(k): int(v) for k, v in m.best_single_rank.value_counts().sort_index().items()},
    mean_top1=float(m.top1_iou.mean()), mean_best_single=float(m.best_single_iou.mean()),
    n_best_single_rank1=int((m.best_single_rank == 1).sum()),
    main_zero=int((m.main <= EPS).sum()),
    greedy5_mean=float(m.greedy5.mean()),
    greedy5_gt_prefix=int((m.greedy5 > m.prefix_best + 1e-6).sum()),
    greedy5_gt_main=int((m.greedy5 > m.main + 1e-6).sum()),
    greedy20_mean=float(m.greedy20.mean()),
    per_scene={s: dict(n=int((m.scene == s).sum()),
                       ok=int(((m.scene == s) & (m.cat == "ok_gap")).sum()),
                       early=int(((m.scene == s) & (m.cat == "cut_early")).sum()),
                       late=int(((m.scene == s) & (m.cat == "cut_late")).sum()),
                       main=float(m.loc[m.scene == s, "main"].mean()),
                       prefix=float(m.loc[m.scene == s, "prefix_best"].mean()))
               for s in SCENES},
)
stats = json.loads(json.dumps(stats, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
m.to_csv(os.path.join(OUT, "per_query_joined.csv"), index=False)


def tab(head, body):
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        + ["| " + " | ".join(str(v) for v in r) + " |" for r in body])


L = []
L.append("# LERF-OVS 第 0 步：gap 截断失败归因")
L.append("")
L.append("- 口径：主路径 pool=top-5 roots（展开到 leaves），gap 规则取前缀；p* = IoU 最优前缀长度。")
L.append("- 校验：重算的 main / prefix_best 与 20261030 oracle 输出逐查询一致（最大差 %.2e / %.2e）。"
         % (stats["consistency_max_diff_main"], stats["consistency_max_diff_prefix"]))
L.append("")
L.append("## 总体构成（%d 个查询）" % N)
L.append("")
L.append(tab(["类别", "查询数", "占比", "main 均值", "prefix_best 均值"], [
    ["ok_gap（截断已最优）", cnt.get("ok_gap", 0), "%.1f%%" % pct(cnt.get("ok_gap", 0)),
     "%.3f" % stats["mean_main_by_cat"].get("ok_gap", float("nan")),
     "%.3f" % stats["mean_prefix_by_cat"].get("ok_gap", float("nan"))],
    ["cut_early（截早了，p*>g）", cnt.get("cut_early", 0), "%.1f%%" % pct(cnt.get("cut_early", 0)),
     "%.3f" % stats["mean_main_by_cat"].get("cut_early", float("nan")),
     "%.3f" % stats["mean_prefix_by_cat"].get("cut_early", float("nan"))],
    ["cut_late（截晚了，p*<g）", cnt.get("cut_late", 0), "%.1f%%" % pct(cnt.get("cut_late", 0)),
     "%.3f" % stats["mean_main_by_cat"].get("cut_late", float("nan")),
     "%.3f" % stats["mean_prefix_by_cat"].get("cut_late", float("nan"))],
]))
L.append("")
L.append("全局：main 均值 %.3f → best prefix 均值 %.3f；gap 选 0 个候选的查询 %d 个；main=0 的查询 %d 个。"
         % (stats["mean_main"], stats["mean_prefix_best"], stats["empty_gap"], stats["main_zero"]))
L.append("")
L.append("## 截断偏差幅度")
L.append("")
L.append(tab(["截早：p* − g", "查询数"], [[k, v] for k, v in stats["cut_early_offset"].items()]))
L.append("")
L.append(tab(["截晚：g − p*", "查询数"], [[k, v] for k, v in stats["cut_late_offset"].items()]))
L.append("")
L.append("## 排序质量（单候选层面）")
L.append("")
L.append(tab(["指标", "数值"], [
    ["单候选 IoU 均值（rank-1）", "%.3f" % stats["mean_top1"]],
    ["单候选 IoU 均值（最好的一个）", "%.3f" % stats["mean_best_single"]],
    ["最好单候选就是 rank-1 的占比", "%.1f%%" % pct(stats["n_best_single_rank1"])],
    ["最好单候选的排名分布", ", ".join("rank%s:%s" % (k, v) for k, v in stats["best_single_rank"].items())],
]))
L.append("")
L.append("## 组合增益（greedy5，已在 oracle CSV 中）")
L.append("")
L.append(tab(["指标", "数值"], [
    ["greedy5 均值", "%.3f" % stats["greedy5_mean"]],
    ["greedy5 > best prefix 的查询数", stats["greedy5_gt_prefix"]],
    ["greedy5 > main 的查询数", stats["greedy5_gt_main"]],
    ["greedy20 均值", "%.3f" % stats["greedy20_mean"]],
]))
L.append("")
L.append("## 逐场景")
L.append("")
L.append(tab(["场景", "查询", "ok_gap", "cut_early", "cut_late", "main 均值", "prefix_best 均值"], [
    [s, r["n"], r["ok"], r["early"], r["late"], "%.3f" % r["main"], "%.3f" % r["prefix"]]
    for s, r in stats["per_scene"].items()]))
L.append("")
with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))

print("N", N, "consistency", stats["consistency_max_diff_main"], stats["consistency_max_diff_prefix"])
print("counts", stats["counts"], "main %.3f prefix %.3f" % (stats["mean_main"], stats["mean_prefix_best"]))
print("early offsets", stats["cut_early_offset"])
print("late offsets", stats["cut_late_offset"])
print("rank dist", stats["best_single_rank"], "rank1 share %.3f" % (stats["n_best_single_rank1"] / N))
print("greedy5 %.3f (gt prefix %d, gt main %d)" % (stats["greedy5_mean"], stats["greedy5_gt_prefix"], stats["greedy5_gt_main"]))
print("wrote", OUT)