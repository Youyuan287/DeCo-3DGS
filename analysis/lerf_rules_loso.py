"""LOSO evaluation of the evidence-gated truncation rules.

For each held-out scene: pick the variant (family + thresholds) with the best mean
IoU on the other three scenes, then evaluate it on the held-out scene. Report
scene-mean IoU against the released main, per-scene choices, win/loss counts, and
the effect on the step-0 truncation-error queries (cut_early / cut_late).
"""
import os
import numpy as np
import pandas as pd

R = r"C:\Users\TYY\Documents\ChatGPT\ZCY-4090\scannet_seed1234_repro"
RU = os.path.join(R, "lerf_rules_20261030")
AT = os.path.join(R, "lerf_attribution_20261030")
SC = ["figurines", "ramen", "teatime", "waldo_kitchen"]
KEY = ["scene", "image", "prompt"]

df = pd.concat([pd.read_csv(os.path.join(RU, "%s_rules.csv" % s)) for s in SC],
               ignore_index=True)
VARS = [c for c in df.columns if c.startswith(("R1_", "R2_", "R3_"))]
attr = pd.read_csv(os.path.join(AT, "per_query_joined.csv"))[KEY + ["cat"]]
df = df.merge(attr, on=KEY, how="left")

FAMS = {"R1": [v for v in VARS if v.startswith("R1_")],
        "R2": [v for v in VARS if v.startswith("R2_")],
        "R3": [v for v in VARS if v.startswith("R3_")]}

sel = {}
for s in SC:
    tr = df[df.scene != s]
    for fam, cols in FAMS.items():
        scores = {c: tr[c].mean() for c in cols}
        best = max(scores, key=scores.get)
        sel[(s, fam)] = (best, scores[best])

print("== 每折选择（训练 3 场景平均 IoU 最高的变体）==")
for s in SC:
    line = [s]
    for fam in FAMS:
        v, sc_ = sel[(s, fam)]
        line.append("%s=%s(train %.3f)" % (fam, v, sc_))
    print("  ".join(line))


def scene_mean(series_by_scene):
    return float(np.mean([np.mean(v) for v in series_by_scene.values()]))


main_by = {s: df.loc[df.scene == s, "main"].values for s in SC}
print("\n== 主结果（场景平均 IoU，含全部 208 查询）==")
print("main（DeCo gap）: %.4f" % scene_mean(main_by))
for fam, cols in FAMS.items():
    by = {s: df.loc[df.scene == s, sel[(s, fam)][0]].values for s in SC}
    print("%s（LOSO 选参）  : %.4f" % (fam, scene_mean(by)))

print("\n== 逐场景（main → 各规则）==")
for s in SC:
    row = [s, "%.4f" % np.mean(main_by[s])]
    for fam in FAMS:
        row.append("%.4f" % np.mean(df.loc[df.scene == s, sel[(s, fam)][0]].values))
    print("  ".join(row))

print("\n== R1 与 main 的逐查询胜负（LOSO 变体）==")
for s in SC:
    sub = df[df.scene == s]
    d = sub[sel[(s, "R1")][0]] - sub["main"]
    print("%s: 胜 %d / 平 %d / 负 %d, 均值 Δ%+.4f"
          % (s, int((d > 1e-9).sum()), int((d.abs() <= 1e-9).sum()),
             int((d < -1e-9).sum()), d.mean()))

print("\n== 按 step-0 截断错误类别看效果（R1, LOSO）==")
for cat in ["ok_gap", "cut_early", "cut_late"]:
    sub = df[df.cat == cat].copy()
    if len(sub) == 0:
        continue
    vals_main, vals_r1 = [], []
    for s in SC:
        ss = sub[sub.scene == s]
        if len(ss) == 0:
            continue
        vals_main.append(ss["main"].mean())
        vals_r1.append(ss[sel[(s, "R1")][0]].mean())
    print("%-10s n=%3d main %.4f → R1 %.4f"
          % (cat, len(sub), np.mean(vals_main), np.mean(vals_r1)))