"""Step 0 failure attribution on the ScanNet DeCo (C1+C2) main arm.

For every root (region): GT majority class, DeCo prediction, top-1/top-2 of the
main scores, and the decision margin s1 - s2. Errors are split into:
  pred_ne_top1 : GT is the top-1 class but DeCo predicted something else
  fixable_top2 : GT is the score runner-up (rerank target)
  miss_top2    : GT neither top-1 nor top-2 (out of reach for a top-2 reranker)
For fixable_top2 it reports the margin distribution (near-tie vs confident),
the share covered by V2's mode candidates, per-scene counts, and the top
confusion pairs. Also reproduces the off and root_top2 oracle mIoU.
"""
import sys, json, os
import numpy as np
import pandas as pd

CODE = "/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code"
REL = "/root/rivermind-data/repro/deco-b123/frozen_ramen_relags"
sys.path[:0] = [CODE, REL, REL + "/ext"]
from eval_scannet_3d import metrics, SUBSETS

B = "/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029"
OUT = "/root/rivermind-data/repro/deco-b123/step0_attribution_20261030"
S = ("scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 "
     "scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00").split()
CL = list(SUBSETS[19])
os.makedirs(OUT, exist_ok=True)

rows = []
scene_stats = []
inter_off, union_off = {}, {}
inter_or, union_or = {}, {}
mac_off, mac_or = [], []
THR = [0.005, 0.01, 0.02, 0.05]


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
    top1 = cl[rank[:, 0]]
    top2 = cl[rank[:, 1]]
    s1 = sc[np.arange(len(sc)), rank[:, 0]]
    s2 = sc[np.arange(len(sc)), rank[:, 1]]
    marg = s1 - s2

    t = pd.read_csv(R + "/SCANNET_M3_WEIGHT_TRACE.csv",
                    usecols=["arm", "class_id", "root_id", "mode_id",
                             "valid", "eligible", "mode_score"])
    t = t[(t.arm == "m1_fixed") & t.valid & t.eligible]
    best = t.loc[t.groupby(["root_id", "mode_id"]).mode_score.idxmax()] \
             .set_index("root_id")["class_id"]
    mcand = best.groupby(level=0).apply(lambda x: set(int(v) for v in x)).to_dict()

    pred_top2 = off.copy()
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(lab.max() + 2))
    n_ok = n_fix = n_miss = n_ne = n_nogt = 0
    for r in range(len(sc)):
        g = order[bounds[r]:bounds[r + 1]]
        if len(g) == 0:
            continue
        m = maj_class(gt[g])
        if m is None:
            n_nogt += 1
            continue
        p = int(off[g[0]])
        t1, t2 = int(top1[r]), int(top2[r])
        in_top2 = (m == t1) or (m == t2)
        if in_top2:
            pred_top2[g] = m
        gt_rank = 1 + int(np.where(cl[rank[r]] == m)[0][0])
        if p == m:
            cat = "ok"; n_ok += 1
        elif m == t1:
            cat = "pred_ne_top1"; n_ne += 1
        elif m == t2:
            cat = "fixable_top2"; n_fix += 1
        else:
            cat = "miss_top2"; n_miss += 1
        rows.append(dict(scene=s, root=r, maj=m, pred=p, top1=t1, top2=t2,
                         s1=float(s1[r]), s2=float(s2[r]), margin=float(marg[r]),
                         gt_rank=gt_rank, in_top2=bool(in_top2),
                         in_modes=bool(m in mcand.get(r, set())),
                         pred_is_top1=bool(p == t1), cat=cat))

    m_off = metrics(gt, off, CL)
    m_or = metrics(gt, pred_top2, CL)
    mac_off.append(m_off["mIoU"])
    mac_or.append(m_or["mIoU"])
    for c, v in m_off["per_class"].items():
        inter_off[c] = inter_off.get(c, 0) + v["inter"]
        union_off[c] = union_off.get(c, 0) + v["union"]
    for c, v in m_or["per_class"].items():
        inter_or[c] = inter_or.get(c, 0) + v["inter"]
        union_or[c] = union_or.get(c, 0) + v["union"]
    sub = [r for r in rows if r["scene"] == s]
    scene_stats.append(dict(
        scene=s, roots=len(sub), ok=n_ok, fixable_top2=n_fix,
        miss_top2=n_miss, pred_ne_top1=n_ne, no_gt=n_nogt,
        pred_eq_top1=float(np.mean([r["pred_is_top1"] for r in sub])) if sub else None,
        off_miou=m_off["mIoU"] * 100, oracle_miou=m_or["mIoU"] * 100,
        fix_share_of_err=(n_fix / max(n_fix + n_miss + n_ne, 1))))
    print("done", s, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT + "/per_root.csv", index=False)

pooled_off = np.mean([inter_off[c] / union_off[c] for c in inter_off]) * 100
pooled_or = np.mean([inter_or[c] / union_or[c] for c in inter_or]) * 100
macro_off = np.mean(mac_off) * 100
macro_or = np.mean(mac_or) * 100

N = len(df)
ok = df[df.cat == "ok"]
fix = df[df.cat == "fixable_top2"]
miss = df[df.cat == "miss_top2"]
ne = df[df.cat == "pred_ne_top1"]
err = N - len(ok)
q = [0, 25, 50, 75, 90, 100]


def qs(x):
    return np.percentile(x, q) if len(x) else np.array([np.nan] * len(q))


def near_share(x, t):
    return float(np.mean(x <= t)) if len(x) else float("nan")


stats = dict(
    n_roots=N, ok=len(ok), errors=err,
    fixable_top2=len(fix), pred_ne_top1=len(ne), miss_top2=len(miss),
    pred_eq_top1_share=float(df.pred_is_top1.mean()),
    pooled_off=pooled_off, macro_off=macro_off,
    pooled_oracle_top2=pooled_or, macro_oracle_top2=macro_or,
    margin_fix_q=qs(fix.margin.tolist()), margin_ok_q=qs(ok.margin.tolist()),
    margin_miss_q=qs(miss.margin.tolist()),
    near_tie_fix={str(t): near_share(fix.margin, t) for t in THR},
    near_tie_ok={str(t): near_share(ok.margin, t) for t in THR},
    modes_cover_fix=float(fix.in_modes.mean()) if len(fix) else 0.0,
    modes_cover_ok=float(ok.in_modes.mean()) if len(ok) else 0.0,
    modes_cover_miss=float(miss.in_modes.mean()) if len(miss) else 0.0,
    miss_rank_q=qs(miss.gt_rank.tolist()),
    miss_rank_le5=float(np.mean(miss.gt_rank <= 5)) if len(miss) else 0.0,
    miss_rank_le10=float(np.mean(miss.gt_rank <= 10)) if len(miss) else 0.0,
    fix_gt_share_of_err=len(fix) / max(err, 1),
)
with open(OUT + "/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def fmt_table(head, body):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for row in body:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


md = []
md.append("# ScanNet 第 0 步：gap / top-1 vs top-2 失败归因（DeCo C1+C2 主链）")
md.append("")
md.append("- 数据：`%s/ablation/v2_on_on/<scene>/predictions_and_scores.npz`（scores_off）" % B)
md.append("- 口径：每个 root 取 GT（19 类内）多数类；margin = 主分数 top-1 减 top-2。")
md.append("- 复现校验：off pooled %.2f / macro %.2f；top-2 oracle pooled %.2f / macro %.2f。"
          % (pooled_off, macro_off, pooled_or, macro_or))
md.append("")
md.append("## 总体构成")
md.append("")
md.append(fmt_table(["类别", "root 数", "占比"], [
    ["正确（pred == GT）", len(ok), "%.1f%%" % (100 * len(ok) / N)],
    ["错误：GT 是 top-1（DeCo 自己丢了）", len(ne), "%.1f%%" % (100 * len(ne) / N)],
    ["错误：GT 是 top-2（重排可救）", len(fix), "%.1f%%" % (100 * len(fix) / N)],
    ["错误：GT 不在 top-2（本步救不了）", len(miss), "%.1f%%" % (100 * len(miss) / N)],
]))
md.append("")
md.append("错误总数 %d；其中 top-2 重排可救 %d（%.1f%%），pred≠top1 %d。"
          % (err, len(fix), 100 * len(fix) / max(err, 1), len(ne)))
md.append("")
md.append("## top-2 可救组的 margin 分布（能否与\"真犹豫\"区分）")
md.append("")
md.append(fmt_table(["分位", "fixable_top2", "正确组（对照）"], [
    [("p%d" % qq), "%.4f" % a, "%.4f" % b]
    for qq, a, b in zip(q, stats["margin_fix_q"], stats["margin_ok_q"])]))
md.append("")
md.append(fmt_table(["阈值 margin ≤", "fixable 占比", "正确组占比（误伤面）"], [
    ["%.3f" % t, "%.1f%%" % (100 * stats["near_tie_fix"][str(t)]),
     "%.1f%%" % (100 * stats["near_tie_ok"][str(t)])] for t in THR]))
md.append("")
md.append("## V2 模式候选覆盖（选择证据是否已在现有数据里）")
md.append("")
md.append(fmt_table(["组", "GT 落在模式候选里", "root 数"], [
    ["fixable_top2", "%.1f%%" % (100 * stats["modes_cover_fix"]), len(fix)],
    ["正确组", "%.1f%%" % (100 * stats["modes_cover_ok"]), len(ok)],
    ["miss_top2", "%.1f%%" % (100 * stats["modes_cover_miss"]), len(miss)],
]))
md.append("")
md.append("## miss_top2 组的 GT 排名")
md.append("")
md.append(fmt_table(["GT 排名", "分位/占比"], [
    ["p25 / p50 / p75", "%.0f / %.0f / %.0f" % (stats["miss_rank_q"][1], stats["miss_rank_q"][2], stats["miss_rank_q"][3])],
    ["≤ top-5", "%.1f%%" % (100 * stats["miss_rank_le5"])],
    ["≤ top-10", "%.1f%%" % (100 * stats["miss_rank_le10"])],
]))
md.append("")
md.append("## 逐场景")
md.append("")
md.append(fmt_table(["场景", "roots", "ok", "fixable_top2", "miss_top2", "pred≠top1", "off mIoU", "oracle mIoU"], [
    [r["scene"], r["roots"], r["ok"], r["fixable_top2"], r["miss_top2"], r["pred_ne_top1"],
     "%.2f" % r["off_miou"], "%.2f" % r["oracle_miou"]] for r in scene_stats]))
md.append("")
pairs = fix.groupby(["top1", "top2"]).size().nlargest(10)
md.append("## fixable_top2 中最常见的混淆对（top-1 → 应为 top-2）")
md.append("")
md.append(fmt_table(["top1", "top2(=GT)", "count"], [[a, b, c] for (a, b), c in pairs.items()]))
md.append("")
with open(OUT + "/summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print()
print("N=%d ok=%d err=%d fixable=%d pred_ne_top1=%d miss=%d" % (N, len(ok), err, len(fix), len(ne), len(miss)))
print("off pooled %.2f macro %.2f | top2-oracle pooled %.2f macro %.2f" % (pooled_off, macro_off, pooled_or, macro_or))
print("margin fix q:", np.round(stats["margin_fix_q"], 4), " ok q:", np.round(stats["margin_ok_q"], 4))
print("near-tie fix:", {k: round(v, 3) for k, v in stats["near_tie_fix"].items()},
      " ok:", {k: round(v, 3) for k, v in stats["near_tie_ok"].items()})
print("modes cover fix %.3f ok %.3f miss %.3f" % (stats["modes_cover_fix"], stats["modes_cover_ok"], stats["modes_cover_miss"]))
print("miss rank q:", np.round(stats["miss_rank_q"], 1))
print("wrote", OUT + "/summary.md", OUT + "/per_root.csv", OUT + "/stats.json")