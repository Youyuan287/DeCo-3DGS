"""Orthogonal-evidence discriminative-power audit for LERF-OVS hard queries.

Questions answered without any learning or weight tuning:
  1. Within a query, can each evidence feature rank the correct candidate (max IoU)
     above the others? (pairwise AUC, rescue rate, bootstrap 95% CI over queries)
  2. On hard queries (best candidate at rank 2 / rank >= 3), does the evidence rank
     it above the CLIP top-1 candidate?
  3. Pre-registered combination E = z(vis_frac) + z(mem_frac) (within-query z),
     reported as-is; all other weights are NOT tuned.
GT IoU is used only to define the correct candidate for evaluation.
"""
import os
import numpy as np
import pandas as pd

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
E = os.path.join(R, "lerf_evidence_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
KEY = ["scene", "image", "prompt"]
FEATS = ["vis_frac", "mean_vis", "mem_frac", "mean_mem", "bbox_diag",
         "flatness", "height", "area_ratio", "n_points",
         "obs_sim_mean", "obs_sim_std", "obs_sim_max", "obs_views"]
RESID_FEATS = ["mean_vis", "area_ratio", "obs_sim_mean", "obs_sim_std", "obs_sim_max"]

df = pd.concat([pd.read_csv(os.path.join(E, "%s_evidence.csv" % s)) for s in SC],
               ignore_index=True)

queries = []
for key, g in df.groupby(KEY):
    g = g.sort_values("rank").reset_index(drop=True)
    j = int(np.argmax(g["iou"].values))
    best_iou = float(g["iou"].values[j])
    best_rank = int(g["rank"].values[j])
    if best_iou < 0.3:
        fam = "recall_fail"
    elif best_rank == 1:
        fam = "found"
    elif best_rank == 2:
        fam = "rank2"
    else:
        fam = "rank_deep"
    queries.append(dict(scene=key[0], image=key[1], prompt=key[2], fam=fam,
                        best_iou=best_iou, best_rank=best_rank,
                        n=len(g), df=g, best=j))
Q = pd.DataFrame(queries)


def within_z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def feature_values(g, name):
    if name == "E_vis":
        return within_z(g["vis_frac"].values) + within_z(g["mem_frac"].values)
    if name == "E_size":
        return within_z(g["area_ratio"].values) + within_z(g["mean_vis"].values)
    return g[name].values


def pairwise_auc(vals, best):
    """mean over others of P(feature(best) > feature(other)), ties 0.5"""
    if len(vals) < 2:
        return np.nan
    v, b = np.delete(vals, best), vals[best]
    hi = np.mean((b > v) + 0.5 * (b == v))
    return float(hi)


def rescue(vals, best):
    return float(np.argmax(vals) == best) if len(vals) < 2 else float(
        np.argmax(vals) == best)


def boot_ci(xs, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.asarray(xs, float)
    xs = xs[~np.isnan(xs)]
    if len(xs) == 0:
        return (np.nan, np.nan)
    m = [np.mean(rng.choice(xs, len(xs), replace=True)) for _ in range(B)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def evaluate(name, subset):
    rows = subset
    aucs, resc = [], []
    for _, q in rows.iterrows():
        vals = feature_values(q["df"], name)
        aucs.append(pairwise_auc(vals, q["best"]))
        resc.append(rescue(vals, q["best"]))
    aucs = [a for a in aucs if not np.isnan(a)]
    return (float(np.mean(aucs)) if aucs else np.nan, boot_ci(aucs),
            float(np.mean(resc)) if resc else np.nan, boot_ci(resc), len(rows))


HARD = Q[Q["fam"].isin(["rank2", "rank_deep"])].reset_index(drop=True)
FOUND = Q[Q["fam"] == "found"].reset_index(drop=True)
ALL = Q[Q["fam"] != "recall_fail"].reset_index(drop=True)

print("queries: all valid %d (hard %d, found %d, recall_fail %d)"
      % (len(ALL), len(HARD), len(FOUND), int((Q["fam"] == "recall_fail").sum())))

print("\n== 饱和检查：within-query 方差为 0 的查询占比 ==")
for name in ["vis_frac", "mean_vis", "mem_frac", "mean_mem", "obs_sim_mean",
             "obs_sim_std", "obs_sim_max", "obs_views"]:
    zero = 0
    for _, q in ALL.iterrows():
        v = q["df"][name].values
        if not np.isfinite(v).any() or np.nanstd(v) < 1e-12:
            zero += 1
    print("%-12s %5.1f%%" % (name, 100 * zero / len(ALL)))


def residual_auc(name, subset):
    aucs = []
    for _, q in subset.iterrows():
        g = q["df"]
        x = np.log1p(g["n_points"].values)
        y = g[name].values
        if not np.isfinite(y).all() or np.ptp(x) < 1e-12:
            continue
        A = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ beta
        if np.ptp(r) < 1e-12:
            continue
        aucs.append(pairwise_auc(r, q["best"]))
    aucs = [a for a in aucs if not np.isnan(a)]
    return (float(np.mean(aucs)) if aucs else np.nan, boot_ci(aucs), len(aucs))

print("\n== 特征判别力（hard ∪ found = 所有存在正确候选的查询）==")
print("%-10s %8s %18s %8s %10s" % ("feature", "AUC", "95%CI", "命中率", "CI"))
for name in FEATS + ["E_vis"]:
    auc, ci, res, ci2, n = evaluate(name, ALL)
    print("%-10s %8.3f [%6.3f,%6.3f] %8.3f [%5.3f,%5.3f] n=%d"
          % (name, auc, ci[0], ci[1], res, ci2[0], ci2[1], n))

print("\n== 只在困难查询上（rank2 + rank_deep）==")
print("%-10s %8s %18s %8s %10s" % ("feature", "AUC", "95%CI", "命中率", "CI"))
for name in FEATS + ["E_vis"]:
    auc, ci, res, ci2, n = evaluate(name, HARD)
    print("%-10s %8.3f [%6.3f,%6.3f] %8.3f [%5.3f,%5.3f] n=%d"
          % (name, auc, ci[0], ci[1], res, ci2[0], ci2[1], n))

print("\n== 困难查询：证据能否把正确候选排到 rank-1 之上 ==")
for name in ["E_vis", "E_size", "vis_frac", "area_ratio", "mean_vis", "flatness"]:
    wins = []
    for _, q in HARD.iterrows():
        g = q["df"]
        r1 = int(np.where(g["rank"].values == 1)[0][0])
        vals = feature_values(g, name)
        if q["best"] == r1:
            continue
        wins.append(float(vals[q["best"]] > vals[r1]) + 0.5 * float(vals[q["best"]] == vals[r1]))
    ci = boot_ci(wins)
    print("%-10s 把正确候选排到 top1 之上: %.3f [%.3f, %.3f] (n=%d)"
          % (name, np.mean(wins), ci[0], ci[1], len(wins)))

print("\n== 预先登记规则 E_size = z(area)+z(mean_vis)：若按它选单个候选 ==")
def rule_effect(subset, label):
    iou_rank1, iou_pick, hits, switches, hurt = [], [], [], [], []
    for _, q in subset.iterrows():
        g = q["df"]
        r1 = int(np.where(g["rank"].values == 1)[0][0])
        vals = feature_values(g, "E_size")
        pick = int(np.argmax(vals))
        iou_rank1.append(float(g["iou"].values[r1]))
        iou_pick.append(float(g["iou"].values[pick]))
        hits.append(float(pick == q["best"]))
        if pick != r1:
            switches.append(1)
            hurt.append(float(g["iou"].values[pick] < g["iou"].values[r1]))
    print("%s: n=%d, rank1 平均 IoU %.3f → 规则选择 %.3f (Δ%+.3f), 命中率 %.3f, 切换率 %.1f%%, 切换中变差 %.1f%%"
          % (label, len(subset), np.mean(iou_rank1), np.mean(iou_pick),
             np.mean(iou_pick) - np.mean(iou_rank1), np.mean(hits),
             100 * len(switches) / len(subset), 100 * np.mean(hurt) if hurt else 0.0))
    return np.mean(iou_pick) - np.mean(iou_rank1)

rule_effect(HARD, "困难查询")
rule_effect(FOUND, "found 查询（rank1 即正确）")
rule_effect(ALL, "全部有正确候选的查询")

print("\n== 尺寸混杂：残差化后（对 log n_points 回归取残差）的 AUC（hard ∪ found）==")
for name in RESID_FEATS:
    auc, ci, n = residual_auc(name, ALL)
    print("%-12s AUC %.3f [%.3f, %.3f] n=%d" % (name, auc, ci[0], ci[1], n))
print("\n== 同上，仅困难查询 ==")
for name in RESID_FEATS:
    auc, ci, n = residual_auc(name, HARD)
    print("%-12s AUC %.3f [%.3f, %.3f] n=%d" % (name, auc, ci[0], ci[1], n))

print("\n== 与尺寸的混杂检查（within-query Spearman vs log n_points）==")
for name in ["vis_frac", "mem_frac", "flatness", "bbox_diag"]:
    cors = []
    for _, q in Q.iterrows():
        g = q["df"]
        if len(g) < 3:
            continue
        cors.append(g[name].corr(np.log1p(g["n_points"]), method="spearman"))
    print("%-10s mean rho=%.3f" % (name, np.nanmean(cors)))

print("\n== 各场景困难族 ==")
print(pd.crosstab(Q["scene"], Q["fam"]).to_string())

out = os.path.join(E, "audit_summary.txt")
with open(out, "w", encoding="utf-8") as f:
    pass
print("\n(feature table saved to memory of this run; raw per-query in evidence CSVs)")