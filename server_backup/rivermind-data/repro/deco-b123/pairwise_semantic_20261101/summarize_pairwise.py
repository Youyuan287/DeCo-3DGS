"""汇总冻结候选池、固定输出数量的 LOSO RankNet 实验。

只读取实际 CSV 和 train_summary.json；没有输入时输出 WAITING_FOR_INPUTS。
A 是原 main/gap 基线，S/J/Q 是事先固定的三个 head，均不称作 gateE。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
EXPECTED_COUNTS = dict(zip(SCENES, (56, 71, 59, 22)))
EXPECTED_PROMPTS = dict(zip(SCENES, (21, 14, 14, 18)))
ARMS = ("A", "S", "J", "Q")
COMPARISONS = (("J", "A"), ("J", "S"), ("J", "Q"))
BOOTSTRAPS = 2000
SEED = 20261101
TOL = 1e-8
BASELINE_TOL = 1e-6
REQUIRED = ("scene", "image", "prompt", "family", "g_A", "A_iou", "S_iou",
            "J_iou", "Q_iou", "baseline_error")


def number(value, key, location):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{location}: {key} 非有限值 {value!r}")
    return result


def integer(value, key, location):
    result = number(value, key, location)
    if result < 0 or int(result) != result:
        raise ValueError(f"{location}: {key} 必须为非负整数")
    return int(result)


def read_inputs(input_root):
    rows, missing, counts, count_unverified = [], [], {}, []
    seen = set()
    for scene in SCENES:
        path = input_root / f"{scene}_pairwise.csv"
        if not path.exists():
            missing.append(str(path))
            counts[scene] = 0
            continue
        scene_rows = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            absent = set(REQUIRED) - columns
            if absent:
                raise ValueError(f"{path}: 缺少字段 {sorted(absent)}")
            count_columns = all(f"g_{arm}" in columns for arm in ARMS)
            if not count_columns:
                count_unverified.append(str(path))
            for line, raw in enumerate(reader, 2):
                where = f"{path}:{line}"
                row = {key: raw[key] for key in REQUIRED}
                if row["scene"] != scene:
                    raise ValueError(f"{where}: scene 不一致")
                if not row["image"] or not row["prompt"] or not row["family"]:
                    raise ValueError(f"{where}: image/prompt/family 为空")
                identity = (scene, row["image"], row["prompt"])
                if identity in seen:
                    raise ValueError(f"重复 query-view: {identity}")
                seen.add(identity)
                row["g_A"] = integer(raw["g_A"], "g_A", where)
                if count_columns:
                    for arm in ARMS[1:]:
                        row[f"g_{arm}"] = integer(raw[f"g_{arm}"], f"g_{arm}", where)
                        if row[f"g_{arm}"] != row["g_A"]:
                            raise ValueError(f"{where}: {arm} 输出数量不等于 g_A")
                for arm in ARMS:
                    key = f"{arm}_iou"
                    row[key] = number(raw[key], key, where)
                    if not -TOL <= row[key] <= 1 + TOL:
                        raise ValueError(f"{where}: {key} 超出 [0,1]")
                row["baseline_error"] = abs(number(raw["baseline_error"], "baseline_error", where))
                scene_rows.append(row)
        counts[scene] = len(scene_rows)
        rows.extend(scene_rows)
    # Each scene/prompt uses one frozen selection; its cardinality must not vary by eval view.
    by_prompt = defaultdict(set)
    for row in rows:
        by_prompt[(row["scene"], row["prompt"])].add(row["g_A"])
    inconsistent = [key for key, values in by_prompt.items() if len(values) != 1]
    if inconsistent:
        raise ValueError(f"同 scene/prompt 的 g_A 跨图不一致: {inconsistent}")
    return rows, missing, counts, count_unverified


def aggregate(rows, key):
    """与 early_root/summarize.py 相同：图内 query 均值→场景图均值→四场等权。"""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["scene"]][row["image"]].append(row[key])
    image_means = {
        scene: {image: float(np.mean(values)) for image, values in sorted(images.items())}
        for scene, images in sorted(grouped.items())
    }
    scene_means = {scene: float(np.mean(list(images.values())))
                   for scene, images in image_means.items()}
    return {
        "image_then_scene_macro": float(np.mean(list(scene_means.values()))) if rows else None,
        "query_mean": float(np.mean([row[key] for row in rows])) if rows else None,
        "scene_query_mean": float(np.mean([
            np.mean([row[key] for row in rows if row["scene"] == scene])
            for scene in sorted(grouped)
        ])) if rows else None,
        "scene_means": scene_means,
        "image_means": image_means,
        "scenes_present": sorted(grouped),
    }


def paired(rows, first, second):
    """场景内重抽图像，所有 head 保持配对；场景等权。"""
    differences = np.asarray([row[first] - row[second] for row in rows], dtype=float)
    grouped = defaultdict(lambda: defaultdict(list))
    for row, difference in zip(rows, differences):
        grouped[row["scene"]][row["image"]].append(float(difference))
    arrays = [np.asarray([np.mean(values) for _, values in sorted(images.items())], dtype=float)
              for _, images in sorted(grouped.items())]
    rng = np.random.default_rng(SEED)
    samples = np.zeros(BOOTSTRAPS, dtype=float)
    for values in arrays:
        indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        samples += values[indices].mean(axis=1) / len(arrays)
    return {
        "first": first, "second": second,
        "delta_image_then_scene_macro": float(np.mean([values.mean() for values in arrays])),
        "delta_query_mean": float(differences.mean()),
        "ci95_image_cluster_bootstrap": np.percentile(samples, (2.5, 97.5)).tolist(),
        "wins": int(np.sum(differences > TOL)),
        "losses": int(np.sum(differences < -TOL)),
        "ties": int(np.sum(np.abs(differences) <= TOL)),
        "bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
        "unit": "image clusters within scene, equal scene weighting",
    }


def subset_summary(rows):
    return {
        "n": len(rows),
        "scene_prompt_units": len({(row["scene"], row["prompt"]) for row in rows}),
        "arms": {arm: aggregate(rows, f"{arm}_iou") for arm in ARMS},
        "comparisons": {f"{first}_minus_{second}": paired(rows, f"{first}_iou", f"{second}_iou")
                        for first, second in COMPARISONS},
        "baseline_error_max": max(row["baseline_error"] for row in rows),
        "g_A_mean": float(np.mean([row["g_A"] for row in rows])),
        "g_A_zero_rows": sum(row["g_A"] == 0 for row in rows),
    }


def load_training_summary(path, predictions_path=None):
    """记录训练诊断，只用 split/完整性证据检验协议，不用诊断数字选 head。"""
    if not path.exists():
        return {"path": str(path), "status": "MISSING", "issues": ["train_summary.json 不存在"]}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    # Reject NaN/Inf in any recorded diagnostic instead of serializing phantom values.
    json.dumps(data, allow_nan=False)
    folds = data.get("folds", [])
    issues, heldouts = [], []
    successful_optimizers = 0
    if data.get("status") != "COMPLETE":
        issues.append("训练状态不是COMPLETE")
    validation = data.get("validation", {})
    for key, expected in (("scene_prompt_units", 67), ("query_views", 208),
                          ("candidate_view_rows", 1097), ("feature_rows", 366)):
        if validation.get(key) != expected:
            issues.append(f"训练validation.{key}必须为{expected}")
    if validation.get("scene_prompt_counts") != EXPECTED_PROMPTS:
        issues.append("训练validation.scene_prompt_counts与冻结67个决策不符")
    if data.get("toy_gradient_check", {}).get("passed") is not True:
        issues.append("toy gradient check未通过")

    def pair_check(value, expected_units, where):
        if not isinstance(value, dict):
            issues.append(f"{where}: 缺少pair_counts")
            return
        if value.get("scene_prompt_units") != expected_units:
            issues.append(f"{where}: scene_prompt_units不符")
        eligible = value.get("eligible_scene_prompt_units")
        if not isinstance(eligible, int) or not 0 < eligible <= expected_units:
            issues.append(f"{where}: eligible_scene_prompt_units无效")
        if not isinstance(value.get("non_tied_pairs"), int) or value["non_tied_pairs"] <= 0:
            issues.append(f"{where}: non_tied_pairs无效")
        if not isinstance(value.get("weight_sum"), (int, float)) or abs(value["weight_sum"] - 1) > 1e-6:
            issues.append(f"{where}: pair weight_sum不为1")

    def diagnostic_check(value, expected_units, where):
        if not isinstance(value, dict):
            issues.append(f"{where}: 缺少排序诊断")
            return
        metrics = ("single_candidate_iou", "query_consistent_single_mask_oracle", "regret",
                   "weighted_pair_accuracy_ties_half")
        valid = True
        for key in metrics:
            metric = value.get(key)
            if not isinstance(metric, (int, float)) or not np.isfinite(metric) or not -TOL <= metric <= 1 + TOL:
                issues.append(f"{where}: {key}无效")
                valid = False
        if valid and abs(value["query_consistent_single_mask_oracle"] - value["single_candidate_iou"] - value["regret"]) > 1e-7:
            issues.append(f"{where}: regret不等于oracle−单mask质量")
        if value.get("diagnostic_only") is not True or value.get("not_a_multimask_upper_bound") is not True:
            issues.append(f"{where}: 缺少单mask诊断边界标记")
        pair_check(value.get("pair_counts"), expected_units, where)

    for index, fold in enumerate(folds):
        heldout = fold.get("heldout_scene")
        training = fold.get("train_scenes", [])
        heldouts.append(heldout)
        if heldout not in SCENES:
            issues.append(f"fold {index}: 未知 heldout scene {heldout!r}")
        elif sorted(training) != sorted(set(SCENES) - {heldout}):
            issues.append(f"fold {heldout}: train_scenes 必须恰为其他三个场景")
        if len(training) != len(set(training)):
            issues.append(f"fold {heldout}: 重复 train scene")
        test_units = EXPECTED_PROMPTS.get(heldout, 0)
        train_units = 67 - test_units
        if fold.get("train_scene_prompt_units") != train_units or fold.get("heldout_scene_prompt_units") != test_units:
            issues.append(f"fold {heldout}: train/heldout决策计数不符")
        pair_check(fold.get("pair_counts"), train_units, f"fold {heldout} train")
        for head in "SJQ":
            optimizer = fold.get("optimizer", {}).get(head, {})
            if optimizer.get("success") is True:
                successful_optimizers += 1
            else:
                issues.append(f"fold {heldout}/{head}: optimizer未成功")
            if optimizer.get("method") != "L-BFGS-B" or not isinstance(optimizer.get("iterations"), int):
                issues.append(f"fold {heldout}/{head}: optimizer方法/iteration缺失")
            for key in ("objective", "gradient_infinity_norm", "coefficient_l2_norm"):
                value = optimizer.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    issues.append(f"fold {heldout}/{head}: optimizer.{key}缺失或非有限")
            diagnostic_check(fold.get("train_regret", {}).get(head), train_units, f"fold {heldout}/{head} train")
            diagnostic_check(fold.get("oof_ranking", {}).get(head), test_units, f"fold {heldout}/{head} heldout")
        for split, units in (("train", train_units), ("test", test_units)):
            shuffle = fold.get("shuffle", {}).get(split, {})
            if shuffle.get("n_units") != units or shuffle.get("fixed_points") != 0 or shuffle.get("same_normalized_text") != 0:
                issues.append(f"fold {heldout}: {split}错配query控制不符合冻结条件")
    if sorted(str(value) for value in heldouts) != sorted(SCENES):
        issues.append("训练摘要必须包含四个且各一次的 heldout scene")
    if successful_optimizers != 12:
        issues.append(f"optimizer成功数为{successful_optimizers}/12")
    for head in "SJQ":
        diagnostic_check(data.get("out_of_fold", {}).get(head), 67, f"全体OOF/{head}")

    predictions_path = predictions_path or path.parent / "predictions.jsonl"
    units, keys, prediction_counts = [], set(), defaultdict(int)
    if not predictions_path.exists():
        issues.append("predictions.jsonl不存在，无法核实OOF覆盖")
    else:
        with predictions_path.open(encoding="utf-8") as handle:
            for line, text in enumerate(handle, 1):
                if not text.strip():
                    continue
                record = json.loads(text)
                json.dumps(record, allow_nan=False)
                scene, prompt = record.get("scene"), record.get("prompt")
                identity = (scene, prompt)
                where = f"predictions:{line}"
                if identity in keys:
                    issues.append(f"{where}: 重复scene/prompt")
                keys.add(identity)
                prediction_counts[scene] += 1
                if record.get("heldout_scene") != scene or scene not in SCENES or sorted(record.get("train_scenes", [])) != sorted(set(SCENES) - {scene}):
                    issues.append(f"{where}: 非严格OOF划分")
                candidates = [tuple(value) for value in record.get("candidates", [])]
                candidate_set = set(candidates)
                count = record.get("g_A")
                if not isinstance(count, int) or count < 0 or count > len(candidates) or len(candidate_set) != len(candidates):
                    issues.append(f"{where}: 候选或g_A无效")
                for arm in ARMS:
                    selected = [tuple(value) for value in record.get("selections", {}).get(arm, [])]
                    if len(selected) != count or len(set(selected)) != len(selected) or not set(selected) <= candidate_set:
                        issues.append(f"{where}: {arm}选择不满足同池固定数量")
                for head in "SJQ":
                    values = record.get("scores", {}).get(head, [])
                    if len(values) != len(candidates) or not np.isfinite(np.asarray(values, dtype=float)).all():
                        issues.append(f"{where}: {head}分数长度/有限性无效")
                units.append({"scene": scene, "prompt": prompt, "g_A": count})
    if len(units) != 67 or len(keys) != 67 or dict(prediction_counts) != EXPECTED_PROMPTS:
        issues.append("OOF predictions必须覆盖四场67个不同scene/prompt各一次")
    return {"path": str(path), "status": "INVALID" if issues else "RECORDED",
            "issues": issues, "successful_optimizers": successful_optimizers,
            "prediction_inventory": {"path": str(predictions_path), "n": len(units),
                                     "scene_prompt_counts": dict(prediction_counts), "units": units},
            "data": data}


def decisions(summary):
    if summary["status"] != "COMPLETE":
        return {"claim_supported": "not_evaluated", "reason": "输入或协议完整性未通过"}
    overall = summary["overall"]
    comparisons = overall["comparisons"]
    main = comparisons["J_minus_A"]
    scene_deltas = {scene: values["comparisons"]["J_minus_A"]["delta_image_then_scene_macro"]
                    for scene, values in summary["scenes"].items()}
    tests = {
        "J_minus_A_at_least_0.01": main["delta_image_then_scene_macro"] >= .01,
        "J_minus_A_CI_low_positive": main["ci95_image_cluster_bootstrap"][0] > 0,
        "no_scene_J_minus_A_below_minus_0.01": all(value >= -.01 for value in scene_deltas.values()),
        "J_minus_S_CI_low_positive": comparisons["J_minus_S"]["ci95_image_cluster_bootstrap"][0] > 0,
        "J_minus_Q_CI_low_positive": comparisons["J_minus_Q"]["ci95_image_cluster_bootstrap"][0] > 0,
    }
    practical = all(tests[key] for key in tuple(tests)[:3])
    semantic = practical and all(tests.values())
    supported = "yes" if semantic else ("partial" if practical else "no")
    return {"claim_supported": supported, "practical_improvement_gate": "PASS" if practical else "FAIL",
            "semantic_interaction_gate": "PASS" if semantic else "FAIL", "tests": tests,
            "scene_J_minus_A": scene_deltas,
            "scope": "固定候选池与g_A；LOSO四场景pilot；不按结果选择head，不代表新场景总体保证"}


def report_text(summary):
    lines = ["# 同 query 偏好学习：LOSO 语义交互 head", "",
             f"状态：**{summary['status']}**；实际 {summary['n']}/208 条 query-view。", "",
             "A为原main/gap；S为标量RankNet；J额外使用正确q⊙f；Q使用错配q⊙f同容量控制。"
             "协议要求所有输出数量固定为原g_A，核验状态见下方。"
             "已有gateE .6760107为独立anchor，本实验A不是gateE。", ""]
    if not summary["n"]:
        lines.append("尚无实际渲染结果，不报告数值、增益或已训练成功。")
    else:
        overall = summary["overall"]
        lines += ["| 口径 | A | S | J | Q |", "|---|---:|---:|---:|---:|"]
        for label, key in (("图→场景macro（主）", "image_then_scene_macro"),
                           ("逐query均值", "query_mean"),
                           ("场景逐query均值再等权", "scene_query_mean")):
            values = " | ".join(f"{overall['arms'][arm][key]:.6f}" for arm in ARMS)
            lines.append(f"| {label} | {values} |")
        lines += ["", "| 固定比较 | 主口径差值 | 95%图级CI | 胜/负/平 |",
                  "|---|---:|---|---:|"]
        for name, result in overall["comparisons"].items():
            low, high = result["ci95_image_cluster_bootstrap"]
            lines.append(f"| {name} | {result['delta_image_then_scene_macro']:+.6f} | "
                         f"[{low:+.6f}, {high:+.6f}] | {result['wins']}/{result['losses']}/{result['ties']} |")
        decision = summary["decision"]
        lines += ["", f"主张支持：**{decision['claim_supported']}**。"]
        if "tests" in decision:
            lines += ["", "| 冻结门槛 | 通过 |", "|---|---|"]
            lines.extend(f"| {key} | {'是' if value else '否'} |" for key, value in decision["tests"].items())
        lines += ["", "主比较始终为J−A，不按S/J/Q谁最好事后切换。训练regret和top1期望IoU仅作诊断。", "",
                  "| 场景 | n | A | S | J | Q | J−A |", "|---|---:|---:|---:|---:|---:|---:|"]
        for scene, item in summary["scenes"].items():
            values = " | ".join(f"{item['arms'][arm]['image_then_scene_macro']:.6f}" for arm in ARMS)
            delta = item["comparisons"]["J_minus_A"]["delta_image_then_scene_macro"]
            lines.append(f"| {scene} | {item['n']} | {values} | {delta:+.6f} |")
        lines += ["", "分族使用族内逐query均值，不能替代全局主口径。", "",
                  "| 分族 | n | A | S | J | Q | J−A |", "|---|---:|---:|---:|---:|---:|---:|"]
        for family, item in summary["families"].items():
            values = " | ".join(f"{item['arms'][arm]['query_mean']:.6f}" for arm in ARMS)
            delta = item["arms"]["J"]["query_mean"] - item["arms"]["A"]["query_mean"]
            lines.append(f"| {family} | {item['n']} | {values} | {delta:+.6f} |")
        lines += ["", f"Baseline最大误差：{overall['baseline_error_max']:.12g}；"
                  f"不同scene/prompt决策数：{overall['scene_prompt_units']}（不视为独立样本数）。",
                  f"Bootstrap为场景内图像cluster配对重采样，B={BOOTSTRAPS}，seed={SEED}；"
                  "区间描述这四个配对场景，不是跨新场景泛化保证。"]
    training = summary["training_summary"]
    lines += ["", f"训练摘要：**{training['status']}**（{training['path']}）。"
              "完整原始训练诊断保存在SUMMARY.json的training_summary.data，不用于事后选择head。"]
    if "data" in training:
        data = training["data"]
        inventory = training.get("prediction_inventory", {})
        lines += ["", f"收敛记录：{training.get('successful_optimizers', 0)}/12个optimizer成功；"
                  f"OOF预测{inventory.get('n', 0)}/67个scene/prompt。", "",
                  "以下为query内一致的**单mask排序诊断**，不是最终固定g_A多mask mIoU，"
                  "其oracle也不是多mask输出上界。", "",
                  "| OOF head | 单mask期望IoU | 单maskoracle | regret | 加权pair准确率 |",
                  "|---|---:|---:|---:|---:|"]
        for head in "SJQ":
            diag = data.get("out_of_fold", {}).get(head, {})
            values = [diag.get(key) for key in ("single_candidate_iou", "query_consistent_single_mask_oracle",
                                                "regret", "weighted_pair_accuracy_ties_half")]
            if all(isinstance(value, (int, float)) for value in values):
                lines.append(f"| {head} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
        lines += ["", "| 留出场景 | head | train单maskIoU | train regret | heldout单maskIoU | heldout regret | train/test pair准确率 | 非tie训练pair | optimizer/iter |",
                  "|---|---|---:|---:|---:|---:|---|---:|---|"]
        for fold in data.get("folds", []):
            for head in "SJQ":
                train = fold.get("train_regret", {}).get(head, {})
                test = fold.get("oof_ranking", {}).get(head, {})
                optimizer = fold.get("optimizer", {}).get(head, {})
                metric_values = [train.get("single_candidate_iou"), train.get("regret"),
                                 test.get("single_candidate_iou"), test.get("regret"),
                                 train.get("weighted_pair_accuracy_ties_half"),
                                 test.get("weighted_pair_accuracy_ties_half")]
                if not all(isinstance(value, (int, float)) for value in metric_values):
                    continue
                ti, tr, oi, ore, ta, oa = metric_values
                lines.append(f"| {fold.get('heldout_scene')} | {head} | {ti:.6f} | {tr:.6f} | "
                             f"{oi:.6f} | {ore:.6f} | {ta:.4f}/{oa:.4f} | "
                             f"{fold.get('pair_counts', {}).get('non_tied_pairs', 'NA')} | "
                             f"{'成功' if optimizer.get('success') else '失败'}/{optimizer.get('iterations', 'NA')} |")
        lines += ["", "训练场景参与偏好监督，heldout场景仅用于训练完成后的诊断与评价。"
                  "全部head按预定四折保存，不依上述表格选择最优head；相同四场已多轮探索，本次不是独立确认。"]
    issues = summary["integrity_issues"] + training.get("issues", [])
    if issues:
        lines += ["", "尚未通过的输入检查："] + [f"- {issue}" for issue in issues]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--train-summary", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    rows, missing, counts, count_unverified = read_inputs(args.input)
    train_path = args.train_summary or args.input / "train_summary.json"
    if args.train_summary is None and not train_path.exists():
        train_path = args.input.parent / "train_summary.json"
    training = load_training_summary(train_path, args.predictions)
    issues = [f"{scene}: {counts[scene]}/{EXPECTED_COUNTS[scene]} 条"
              for scene in SCENES if counts[scene] != EXPECTED_COUNTS[scene]]
    issues += [f"缺少输入 {path}" for path in missing]
    issues += [f"缺少g_S/g_J/g_Q，无法独立核实固定数量: {path}" for path in count_unverified]
    if rows and max(row["baseline_error"] for row in rows) > BASELINE_TOL:
        issues.append("baseline误差超过1e-6")
    prediction_units = training.get("prediction_inventory", {}).get("units", [])
    prediction_lookup = {(unit["scene"], unit["prompt"]): unit["g_A"] for unit in prediction_units}
    for row in rows:
        identity = (row["scene"], row["prompt"])
        if identity not in prediction_lookup or row["g_A"] != prediction_lookup[identity]:
            issues.append(f"render CSV与OOF预测的身份/g_A不一致: {identity}")
    if len(rows) == 208 and {(row["scene"], row["prompt"]) for row in rows} != set(prediction_lookup):
        issues.append("完整render CSV与67个OOF预测的scene/prompt覆盖不同")
    summary = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "input_root": str(args.input), "n": len(rows),
        "expected_counts": EXPECTED_COUNTS, "actual_counts": counts,
        "missing_files": missing, "integrity_issues": issues,
        "fixed_counts_verified": bool(rows) and not count_unverified,
        "gateE_anchor_is_independent": True, "training_summary": training,
        "status": ("WAITING_FOR_INPUTS" if not rows else
                   "INCOMPLETE_OR_INVALID" if issues or training["status"] != "RECORDED" else "COMPLETE"),
    }
    if rows:
        summary["overall"] = subset_summary(rows)
        summary["scenes"] = {scene: subset_summary([row for row in rows if row["scene"] == scene])
                             for scene in SCENES if counts[scene]}
        summary["families"] = {family: subset_summary([row for row in rows if row["family"] == family])
                               for family in sorted({row["family"] for row in rows})}
    summary["decision"] = decisions(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "REPORT.md").write_text(report_text(summary), encoding="utf-8")
    compact = {"status": summary["status"], "rows": summary["n"],
               "claim_supported": summary["decision"]["claim_supported"],
               "training_summary": training["status"], "integrity_issues": issues + training.get("issues", [])}
    if rows:
        compact["primary_macro"] = {arm: summary["overall"]["arms"][arm]["image_then_scene_macro"] for arm in ARMS}
        compact["comparisons"] = {key: {"delta": value["delta_image_then_scene_macro"],
                                        "ci95": value["ci95_image_cluster_bootstrap"]}
                                  for key, value in summary["overall"]["comparisons"].items()}
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
