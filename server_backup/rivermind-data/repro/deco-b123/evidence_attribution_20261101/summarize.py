"""Summarize the frozen evidence-attribution pilot without changing its outputs.

Usage: python summarize.py [--root RUN_DIRECTORY]
The input directory is ROOT/results and outputs are ROOT/summary.json, ROOT/REPORT.md.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
FAMILIES = ("found", "rank2", "deep", "recall")
EXPECTED_SCENES = dict(zip(SCENES, (56, 71, 59, 22)))
EXPECTED_FAMILIES = dict(zip(FAMILIES, (124, 36, 19, 29)))
METHODS = ("A", "G", "I", "J")
COMPARISONS = (("G", "A"), ("I", "A"), ("J", "A"), ("J", "I"), ("J", "G"))
IOU_FIELDS = tuple(f"{tag}{suffix}" for suffix in ("_iou", "_single_iou") for tag in METHODS)
DIAGNOSTICS = ("n_roots", "competition_fraction", "changed_tokens", "valid_candidates",
               "joint_fallbacks", "solver_seconds", "geom_retention_min")
NUMERIC_FIELDS = IOU_FIELDS + ("baseline_error",) + DIAGNOSTICS
SEED, BOOTSTRAPS = 20261030, 2000
TOL = 1e-8


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return {"n_finite": int(len(valid)), "n_missing": int(len(values) - len(valid)),
            "mean": float(valid.mean()) if len(valid) else None,
            "median": float(np.median(valid)) if len(valid) else None,
            "min": float(valid.min()) if len(valid) else None,
            "max": float(valid.max()) if len(valid) else None,
            "sum": float(valid.sum()) if len(valid) else None}


def finite_mean(values):
    return finite_stats(values)["mean"]


def image_means(rows, key, scene):
    subset = [row for row in rows if row["scene"] == scene]
    result = []
    for name in sorted({row["image"] for row in subset}):
        value = finite_mean([row[key] for row in subset if row["image"] == name])
        if value is not None:
            result.append(value)
    return np.asarray(result, dtype=float)


def scene_mean(rows, key, scene):
    return finite_mean(image_means(rows, key, scene))


def macro(rows, key):
    values = [scene_mean(rows, key, scene) for scene in SCENES]
    # Never silently drop a scene from the four-scene macro.
    return float(np.mean(values)) if all(v is not None for v in values) else None


def comparison(rows, first, second, suffix):
    key_a, key_b = first + suffix, second + suffix
    paired = []
    for row in rows:
        if np.isfinite(row[key_a]) and np.isfinite(row[key_b]):
            paired.append({**row, "difference": row[key_a] - row[key_b]})
    differences = np.asarray([row["difference"] for row in paired])
    by_scene = [image_means(paired, "difference", scene) for scene in SCENES]
    complete_scene_coverage = all(len(values) > 0 for values in by_scene)
    ci = None
    delta = None
    if complete_scene_coverage:
        delta = float(np.mean([values.mean() for values in by_scene]))
        rng = np.random.default_rng(SEED)
        samples = np.empty(BOOTSTRAPS)
        for i in range(BOOTSTRAPS):
            samples[i] = np.mean([values[rng.integers(0, len(values), len(values))].mean()
                                  for values in by_scene])
        ci = np.percentile(samples, (2.5, 97.5)).tolist()
    return {"delta": delta, "ci95": ci, "paired_queries": len(paired),
            "missing_pairs": len(rows) - len(paired),
            "paired_images_per_scene": dict(zip(SCENES, map(len, by_scene))),
            "wins": int(np.count_nonzero(differences > TOL)),
            "losses": int(np.count_nonzero(differences < -TOL)),
            "ties": int(np.count_nonzero(np.abs(differences) <= TOL))}


def fmt(value, digits=6):
    return "缺失" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def report(summary, rows):
    labels = {"A": "原 gateE", "G": "几何 Hungarian 绑定＋top-3",
              "I": "独立 mask 文本 top-3", "J": "近最优几何匹配约束＋top-3"}
    lines = ["# 多视角证据归属实验结果", "",
             "**判定：" + ("PASS" if summary["pass"] else "FAIL") + "。**", "",
             f"完整性：{summary['n']} 条查询；基线最大误差 {fmt(summary['baseline_error']['max'])}。",
             "A/G/I/J 的最终输出均使用同一 z 融合、gap 与 gateE 选择器；single 列为相应信号 argmax 候选单独渲染。",
             "主指标先按图内查询平均，再按场景内图平均，最后四场景等权；分族表为族内逐查询平均。", "",
             "## 最终选择结果", "",
             "| 方法 | figurines | ramen | teatime | waldo_kitchen | macro |",
             "|---|---:|---:|---:|---:|---:|"]
    for tag in METHODS:
        key = tag + "_iou"
        lines.append("| " + tag + " " + labels[tag] + " | " + " | ".join(
            fmt(summary["scenes"][scene][key]) for scene in SCENES) +
            " | " + fmt(summary["macro"][key]) + " |")
    lines += ["", "## 配对差值", "",
              f"场景分层、以图为 cluster 配对 bootstrap；B={BOOTSTRAPS}，seed={SEED}，百分位95%CI。胜负平按逐查询IoU差统计。",
              "| 对比 | macro差值 | 95%CI | 胜/负/平 | 有效配对查询 |",
              "|---|---:|---|---|---:|"]
    for name, item in summary["comparisons"].items():
        ci = "缺失" if item["ci95"] is None else "[" + ", ".join(map(fmt, item["ci95"])) + "]"
        lines.append(f"| {name} | {fmt(item['delta'])} | {ci} | "
                     f"{item['wins']}/{item['losses']}/{item['ties']} | {item['paired_queries']} |")
    lines += ["", "## 分族（逐查询平均）", "",
              "| 族 | n | A | G | I | J | J−A |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        values = summary["families"][family]
        delta = None if values["J_iou"] is None or values["A_iou"] is None else values["J_iou"] - values["A_iou"]
        lines.append(f"| {family} | {values['n']} | " + " | ".join(fmt(values[tag + "_iou"]) for tag in METHODS)
                     + " | " + fmt(delta) + " |")
    lines += ["", "## 单候选读出（不混作最终管线结果）", "",
              "| 方法 | figurines | ramen | teatime | waldo_kitchen | macro |",
              "|---|---:|---:|---:|---:|---:|"]
    for tag in METHODS:
        key = tag + "_single_iou"
        lines.append("| " + tag + " | " + " | ".join(fmt(summary["scenes"][scene][key]) for scene in SCENES)
                     + " | " + fmt(summary["macro"][key]) + " |")
    lines += ["", "## 冻结门槛", "", "| 条件 | 是否满足 |", "|---|---|"]
    for name, value in summary["gates"].items():
        lines.append(f"| {name} | {'是' if value else '否'} |")
    lines += ["", "## 归属与求解诊断", "",
              "以下为逐查询字段统计；重复提示词或共享实例不会被当作独立实例数量。",
              "| 字段 | 有效n | 均值 | 中位数 | 最小 | 最大 |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in summary["diagnostics"].items():
        lines.append(f"| {name} | {values['n_finite']} | " + " | ".join(
            fmt(values[k]) for k in ("mean", "median", "min", "max")) + " |")
    lines += ["", "## 结果使用边界", "",
              "- 缺失值不填零；表中均值仅使用有限值，并在 summary.json 记录每字段的有效数量。任何主IoU缺失或场景/分族计数不符均不能通过门槛。",
              "- 配对CI只使用两法均有效的查询；存在缺失时，其估计对象与完整208条结果不同。",
              "- 不把单候选结果与最终集合渲染混用，不把本实验移植机制称作相关论文完整复现。",
              "- 该试验判定只针对冻结配置；不能据此自动证明新颖性、普遍因果关系或跨数据集泛化。"]
    if summary["warnings"]:
        lines += ["", "## 数据警告", ""] + ["- " + warning for warning in summary["warnings"]]
    paired = [r for r in rows if np.isfinite(r["A_iou"]) and np.isfinite(r["J_iou"])]
    for descending, title in ((True, "J相对A最大的改善"), (False, "J相对A最大的下降")):
        selected = sorted(paired, key=lambda r:r["J_iou"] - r["A_iou"], reverse=descending)[:5]
        lines += ["", "## " + title, "", "| 场景 | 图 | 查询 | 族 | A | J | 差值 |",
                  "|---|---|---|---|---:|---:|---:|"]
        for r in selected:
            name = r["prompt"].replace("|", "\\|")
            lines.append(f"| {r['scene']} | {r['image']} | {name} | {r['family']} | "
                         f"{fmt(r['A_iou'])} | {fmt(r['J_iou'])} | {fmt(r['J_iou']-r['A_iou'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().root
    rows, seen = [], set()
    for scene in SCENES:
        path = root / "results" / f"{scene}_attribution.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"scene", "image", "prompt", "family", *NUMERIC_FIELDS}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path}: 缺少字段 {sorted(missing)}")
            for raw in reader:
                row = {**raw, **{key:number(raw[key]) for key in NUMERIC_FIELDS}}
                if row["scene"] != scene:
                    raise ValueError(f"{path}: scene不匹配 {row['scene']}")
                key = tuple(row[k] for k in ("scene", "image", "prompt"))
                if key in seen:
                    raise ValueError(f"重复查询: {key}")
                seen.add(key)
                rows.append(row)
    counts, families = Counter(r["scene"] for r in rows), Counter(r["family"] for r in rows)
    warnings = []
    if dict(counts) != EXPECTED_SCENES:
        warnings.append(f"场景计数与冻结值不同：{dict(counts)}，预期{EXPECTED_SCENES}")
    if dict(families) != EXPECTED_FAMILIES:
        warnings.append(f"分族计数与冻结值不同：{dict(families)}，预期{EXPECTED_FAMILIES}")
    coverage = {key:finite_stats([row[key] for row in rows])["n_finite"] for key in NUMERIC_FIELDS}
    for key in IOU_FIELDS:
        if coverage[key] != len(rows):
            warnings.append(f"{key} 仅{coverage[key]}/{len(rows)}条为有限值")
        if any(np.isfinite(r[key]) and not 0 <= r[key] <= 1 for r in rows):
            raise ValueError(f"{key}: 有IoU超出[0,1]")
    summary = {"n":len(rows), "scene_counts":dict(counts), "family_counts":dict(families),
               "bootstrap": {"B":BOOTSTRAPS, "seed":SEED, "unit":"paired image clusters within scene"},
               "finite_counts":coverage, "baseline_error":finite_stats([r["baseline_error"] for r in rows]),
               "macro":{key:macro(rows,key) for key in IOU_FIELDS},
               "scenes":{scene:{key:scene_mean(rows,key,scene) for key in IOU_FIELDS} for scene in SCENES},
               "families":{family:{"n":families[family], **{key:finite_mean([r[key] for r in rows if r["family"]==family]) for key in IOU_FIELDS}} for family in FAMILIES},
               "comparisons":{a+"-"+b:comparison(rows,a,b,"_iou") for a,b in COMPARISONS},
               "single_comparisons":{a+"-"+b:comparison(rows,a,b,"_single_iou") for a,b in COMPARISONS},
               "diagnostics":{key:finite_stats([r[key] for r in rows]) for key in DIAGNOSTICS},
               "warnings":warnings}
    valid = (len(rows)==208 and dict(counts)==EXPECTED_SCENES and dict(families)==EXPECTED_FAMILIES
             and all(coverage[k]==208 for k in IOU_FIELDS))
    baseline = summary["baseline_error"]
    baseline_valid = baseline["n_finite"]==208 and baseline["max"] is not None and baseline["max"]<=1e-6
    diff = summary["comparisons"]["J-A"]
    found = summary["families"]["found"]
    gates = {"完整208条、场景/分族正确且全部IoU有限":valid,
             "基线208条匹配（最大误差≤1e-6）":baseline_valid,
             "J−A macro≥0.01":diff["delta"] is not None and diff["delta"]>=.01,
             "J−A 95%CI下界>0":diff["ci95"] is not None and diff["ci95"][0]>0,
             "任一场景J−A≥−0.01":all(v["J_iou"] is not None and v["A_iou"] is not None and v["J_iou"]-v["A_iou"]>=-.01 for v in summary["scenes"].values()),
             "found J−A≥−0.005":found["J_iou"] is not None and found["A_iou"] is not None and found["J_iou"]-found["A_iou"]>=-.005,
             "J macro高于I与G":all(summary["comparisons"][name]["delta"] is not None and summary["comparisons"][name]["delta"]>0 for name in ("J-I","J-G"))}
    summary["gates"], summary["pass"] = gates, bool(all(gates.values()))
    output = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    (root / "summary.json").write_text(output+"\n",encoding="utf-8")
    (root / "REPORT.md").write_text(report(summary, rows),encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
