"""Summarize existing score decomposition outputs; never run inference.

Requires sibling early_root_top3_20261101/summarize.py for the shared aggregate,
paired image bootstrap and output writer. C-A is the only main comparison;
B/A is a strict positive control and D/E are diagnostic, never selected as best.
Without result rows, main() writes WAITING_FOR_INPUTS, not experimental numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import runpy
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


SHARED_PATH = Path(__file__).resolve().parents[1] / "early_root_top3_20261101" / "summarize.py"
SHARED = runpy.run_path(str(SHARED_PATH), run_name="_score_shared_statistics")
aggregate, paired, write_outputs = (SHARED[k] for k in ("aggregate", "paired", "write_outputs"))
SCENES, EXPECTED_COUNTS = SHARED["SCENES"], SHARED["EXPECTED_COUNTS"]
BOOTSTRAPS, SEED, TOL = SHARED["BOOTSTRAPS"], SHARED["SEED"], SHARED["TOL"]
ARMS = "ABCDE"
PAIR_FIELDS = ("pairs", "raw_correct_pairs", "margin_correct_pairs", "canon_hurts_pairs", "canon_helps_pairs")
FIELDS = ("scene", "image", "prompt", "family", "n", "g_A", "g_D", "g_E",
          "A_iou", "B_iou", "C_iou", "D_iou", "E_iou", "baseline_error",
          "control_set_equal", "AC_set_changed", "AD_set_changed", *PAIR_FIELDS)


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite number: {value!r}")
    return result


def ids(values):
    return [tuple(map(int, value)) for value in values]


def detail_metrics(detail):
    keys = ids(detail["candidates"])
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate IDs")
    arrays = {k: [finite(v) for v in detail[k]] for k in ("r", "b", "m", "s", "candidate_ious")}
    if any(len(v) != len(keys) for v in arrays.values()):
        raise ValueError("detail arrays and candidates have different lengths")
    if any(not -TOL <= v <= 1 + TOL for v in arrays["candidate_ious"]):
        raise ValueError("candidate IoU outside [0,1]")
    if any(abs(r - b - m) > 1e-6 for r, b, m in zip(arrays["r"], arrays["b"], arrays["m"])):
        raise ValueError("recorded m does not equal r-b")
    selections = {a: ids(detail["selections"][a]) for a in ARMS}
    if any(len(v) != len(set(v)) or not set(v) <= set(keys) for v in selections.values()):
        raise ValueError("selection is not a unique subset of candidates")
    pos = [i for i, v in enumerate(arrays["candidate_ious"]) if v >= .5]
    neg = [i for i, v in enumerate(arrays["candidate_ious"]) if v <= .1]
    pairs = [(p, n) for p in pos for n in neg]
    r, m = arrays["r"], arrays["m"]
    # Exact comparisons match run_score.py. Tied scores are not counted correct.
    out = dict(pairs=len(pairs),
        raw_correct_pairs=sum(r[p] > r[n] for p, n in pairs),
        margin_correct_pairs=sum(m[p] > m[n] for p, n in pairs),
        canon_hurts_pairs=sum(r[p] > r[n] and m[p] < m[n] for p, n in pairs),
        canon_helps_pairs=sum(r[p] < r[n] and m[p] > m[n] for p, n in pairs),
        raw_tied_pairs=sum(r[p] == r[n] for p, n in pairs),
        margin_tied_pairs=sum(m[p] == m[n] for p, n in pairs),
        detail_control_set_equal=set(selections["A"]) == set(selections["B"]),
        detail_AC_set_changed=set(selections["A"]) != set(selections["C"]),
        detail_AD_set_changed=set(selections["A"]) != set(selections["D"]),
        detail_fixed_count_equal=len(selections["A"]) == len(selections["B"]) == len(selections["C"]),
        has_positive=bool(pos), n_positives=len(pos), n_negatives=len(neg),
        candidate_pair_ids=[(keys[p], keys[n]) for p, n in pairs])
    ranks = {}
    for name in ("r", "m", "s"):
        order = sorted(range(len(keys)), key=lambda i: (-arrays[name][i], i))
        ranks[name] = {i: rank + 1 for rank, i in enumerate(order)}
    best = max((arrays["candidate_ious"][i] for i in pos), default=None)
    best_ids = [i for i in pos if abs(arrays["candidate_ious"][i] - best) <= TOL]
    for label, indexes in (("first_positive", pos), ("best_positive", best_ids)):
        for name in ("r", "m", "s"):
            out[f"{label}_rank_{name}"] = min((ranks[name][i] for i in indexes), default=None)
        out[f"{label}_rank_improvement_raw_vs_margin"] = (
            out[f"{label}_rank_m"] - out[f"{label}_rank_r"] if indexes else None)
    # Inference state must be fixed across views, while candidate IoU can vary.
    out["inference_signature"] = json.dumps({"candidates": keys, "selections": selections,
        **{k: arrays[k] for k in ("r", "b", "m", "s")}}, sort_keys=True)
    return out, selections


def read_inputs(root):
    rows, missing, counts, detail_counts = [], [], {}, {}
    detail_missing_keys, detail_extra_keys = [], []
    for scene in SCENES:
        csv_path, json_path = root / f"{scene}_score.csv", root / f"{scene}_details.jsonl"
        details = {}
        if json_path.exists():
            with json_path.open(encoding="utf-8-sig") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    key = (d["image"], d["prompt"])
                    if d["scene"] != scene or key in details:
                        raise ValueError(f"detail scene mismatch/duplicate at {json_path}:{line_no}")
                    details[key] = d
        else:
            missing.append(str(json_path))
        detail_counts[scene] = len(details)
        scene_rows, seen = [], set()
        if csv_path.exists():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if set(FIELDS) - set(reader.fieldnames or ()):
                    raise ValueError(f"missing required CSV columns: {csv_path}")
                for line_no, raw in enumerate(reader, 2):
                    row = {k: raw[k] for k in FIELDS}
                    key = (row["image"], row["prompt"])
                    if row["scene"] != scene or key in seen:
                        raise ValueError(f"CSV scene mismatch/duplicate at {csv_path}:{line_no}")
                    seen.add(key)
                    for k in set(FIELDS) - {"scene", "image", "prompt", "family"}:
                        row[k] = finite(row[k])
                    for a in ARMS:
                        if not -TOL <= row[f"{a}_iou"] <= 1 + TOL:
                            raise ValueError(f"invalid IoU at {csv_path}:{line_no}")
                    if key in details:
                        derived, selections = detail_metrics(details[key])
                        for k in PAIR_FIELDS:
                            if row[k] != derived[k]:
                                raise ValueError(f"CSV/detail {k} mismatch: {scene}/{key}")
                        for a in ("A", "D", "E"):
                            if row[f"g_{a}"] != len(selections[a]):
                                raise ValueError(f"CSV/detail selected-count mismatch: {scene}/{key}/{a}")
                        if row["n"] != len(details[key]["candidates"]):
                            raise ValueError(f"CSV/detail pool-size mismatch: {scene}/{key}")
                        for field, derived_key in (("control_set_equal", "detail_control_set_equal"),
                            ("AC_set_changed", "detail_AC_set_changed"), ("AD_set_changed", "detail_AD_set_changed")):
                            if row[field] != int(derived[derived_key]):
                                raise ValueError(f"CSV/detail {field} mismatch: {scene}/{key}")
                        row.update(derived)
                    else:
                        detail_missing_keys.append([scene, *key])
                    scene_rows.append(row)
        else:
            missing.append(str(csv_path))
        detail_extra_keys.extend([scene, *key] for key in sorted(set(details) - seen))
        counts[scene] = len(scene_rows)
        rows.extend(scene_rows)
    return rows, dict(missing_files=missing, actual_counts=counts, detail_counts=detail_counts,
        detail_missing_keys=detail_missing_keys, detail_extra_keys=detail_extra_keys,
        count_mismatches=[f"{s}: CSV {counts[s]}, details {detail_counts[s]}, expected {EXPECTED_COUNTS[s]}"
            for s in SCENES if counts[s] != EXPECTED_COUNTS[s] or detail_counts[s] != EXPECTED_COUNTS[s]])


def mechanism_summary(rows):
    totals = {k: int(sum(r[k] for r in rows)) for k in PAIR_FIELDS}
    eligible = [r for r in rows if r["pairs"] > 0]
    units = defaultdict(list)
    for r in eligible:
        units[(r["scene"], r["prompt"])].append(r)
    totals.update(query_views_with_pairs=len(eligible), scene_prompt_units_with_pairs=len(units),
        query_views=len(rows), scene_prompt_units=len({(r["scene"], r["prompt"]) for r in rows}))
    totals["pair_occurrences_are_not_independent_samples"] = True
    for name in ("raw_correct_pairs", "margin_correct_pairs", "canon_hurts_pairs", "canon_helps_pairs"):
        totals[name + "_occurrence_rate"] = totals[name] / totals["pairs"] if totals["pairs"] else None
        totals[name + "_mean_query_view_rate"] = mean(r[name] / r["pairs"] for r in eligible) if eligible else None
        totals[name + "_mean_scene_prompt_rate"] = mean(
            sum(r[name] for r in rs) / sum(r["pairs"] for r in rs) for rs in units.values()) if units else None
    detailed = [r for r in rows if "has_positive" in r]
    totals["detailed_query_views"] = len(detailed)
    totals["distinct_oriented_scene_prompt_candidate_pairs"] = len({
        (r["scene"], r["prompt"], p, n) for r in detailed for p, n in r["candidate_pair_ids"]})
    totals["raw_tied_pair_occurrences"] = sum(r["raw_tied_pairs"] for r in detailed)
    totals["margin_tied_pair_occurrences"] = sum(r["margin_tied_pairs"] for r in detailed)
    rank_results = {}
    for label in ("first_positive", "best_positive"):
        values = [r for r in detailed if r["has_positive"]]
        changes = [r[f"{label}_rank_improvement_raw_vs_margin"] for r in values]
        rank_results[label] = dict(n_query_views=len(values),
            n_scene_prompt_units=len({(r["scene"], r["prompt"]) for r in values}),
            improved=sum(v > 0 for v in changes), worsened=sum(v < 0 for v in changes),
            tied=sum(v == 0 for v in changes),
            mean_rank_margin=mean(r[f"{label}_rank_m"] for r in values) if values else None,
            mean_rank_raw=mean(r[f"{label}_rank_r"] for r in values) if values else None)
    totals["positive_mask_ranks"] = rank_results
    return totals


def subset_summary(rows):
    out = {"n": len(rows), "scene_prompt_units": len({(r["scene"], r["prompt"]) for r in rows}),
        "arms": {a: aggregate(rows, f"{a}_iou") for a in ARMS},
        "C_minus_A_main": paired(rows, "C_iou", "A_iou"),
        "B_minus_A_control": paired(rows, "B_iou", "A_iou"),
        "D_minus_A_diagnostic": paired(rows, "D_iou", "A_iou"),
        "E_minus_A_diagnostic": paired(rows, "E_iou", "A_iou"),
        "mechanism": mechanism_summary(rows),
        "AC_changed_query_views": sum(r["AC_set_changed"] == 1 for r in rows),
        "AC_changed_scene_prompts": len({(r["scene"], r["prompt"]) for r in rows if r["AC_set_changed"] == 1})}
    return out


def build_summary(rows, inventory, input_root):
    out = dict(generated_at=datetime.now(timezone.utc).isoformat(), input_root=str(input_root),
        shared_statistics_source=str(SHARED_PATH), expected_counts=EXPECTED_COUNTS, n=len(rows), **inventory,
        main_comparison="C_iou - A_iou", diagnostic_only_arms=["D", "E"], gateE_anchor_is_independent=True)
    if not rows:
        out.update(status="WAITING_FOR_INPUTS", main_verdict="NOT_EVALUABLE")
        return out
    signatures = defaultdict(set)
    for r in rows:
        if "inference_signature" in r:
            signatures[(r["scene"], r["prompt"])].add(r["inference_signature"])
    integrity = dict(baseline_error_max=max(abs(r["baseline_error"]) for r in rows),
        baseline_error_exact_zero=all(r["baseline_error"] == 0 for r in rows),
        baseline_within_preregistered_1e_6=all(abs(r["baseline_error"]) <= 1e-6 for r in rows),
        AB_iou_exact_equal=all(r["A_iou"] == r["B_iou"] for r in rows),
        AB_iou_max_absolute_difference=max(abs(r["A_iou"] - r["B_iou"]) for r in rows),
        AB_sets_exact_equal=all(r["control_set_equal"] == 1 for r in rows),
        ABC_fixed_count_equal=all(r.get("detail_fixed_count_equal", False) for r in rows),
        inference_constant_within_scene_prompt=all(len(v) == 1 for v in signatures.values()),
        complete_counts=not inventory["count_mismatches"],
        complete_detail_coverage=not inventory["missing_files"] and not inventory["detail_missing_keys"] and not inventory["detail_extra_keys"])
    valid = all(integrity[k] for k in ("baseline_within_preregistered_1e_6", "AB_iou_exact_equal",
        "AB_sets_exact_equal", "ABC_fixed_count_equal", "inference_constant_within_scene_prompt"))
    out["integrity"] = integrity
    out["status"] = "COMPLETE" if integrity["complete_counts"] and integrity["complete_detail_coverage"] else "PARTIAL_INPUTS"
    if not valid:
        out["status"] = "INTEGRITY_MISMATCH" if integrity["complete_detail_coverage"] else "PARTIAL_INPUTS"
    out["overall"] = subset_summary(rows)
    out["scenes"] = {s: subset_summary([r for r in rows if r["scene"] == s]) for s in SCENES if any(r["scene"] == s for r in rows)}
    out["families"] = {f: subset_summary([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})}
    main = out["overall"]["C_minus_A_main"]
    checks = dict(primary_delta_positive=main["delta_image_then_scene_macro"] > 0,
        bootstrap_lower_positive=main["ci95_image_cluster_bootstrap"][0] > 0,
        no_scene_decreases_more_than_0_01=all(item["C_minus_A_main"]["delta_image_then_scene_macro"] >= -.01 for item in out["scenes"].values()))
    out["frozen_main_checks"] = checks
    out["main_verdict"] = ("PASS" if all(checks.values()) else "FAIL") if out["status"] == "COMPLETE" and valid else "NOT_EVALUABLE"
    return out


def fmt(value, signed=False):
    return "NA" if value is None else format(value, "+.6f" if signed else ".6f")


def render_report(summary):
    lines = ["# Same-candidate score decomposition", "",
        f"Status: **{summary['status']}**; rows {summary['n']}/208; frozen C-A verdict **{summary['main_verdict']}**.", "",
        "A=original sigmoid/gap; B=margin with fixed A count (positive control); C=raw dot product with fixed A count (main). "
        "D=margin/own gap and E=raw/own gap are diagnostics and cannot be selected post hoc as the main result. "
        "No new inference is performed by this summary; gateE .6760107 remains an independent historical anchor.", ""]
    if "overall" not in summary:
        return "\n".join(lines + ["Waiting for results/{scene}_score.csv and results/{scene}_details.jsonl for all four scenes. "
            "No numerical experiment outcome is claimed.", ""])
    overall = summary["overall"]
    main = overall["C_minus_A_main"]
    ci = main["ci95_image_cluster_bootstrap"]
    lines += [f"Main C-A paired 95% CI: [{ci[0]:+.6f}, {ci[1]:+.6f}]; "
        f"{main['wins']} wins / {main['losses']} losses / {main['ties']} ties.", "",
        "| Weighting | A | B control | C main | C-A |", "|---|---:|---:|---:|---:|"]
    for label, key in (("Image → scene macro (primary)", "image_then_scene_macro"),
        ("Query-view mean", "query_mean"), ("Mean of scene query-view means", "scene_query_mean")):
        a, b, c = (overall["arms"][x][key] for x in "ABC")
        lines.append(f"| {label} | {a:.6f} | {b:.6f} | {c:.6f} | {c-a:+.6f} |")
    lines += ["", "| Diagnostic arm | Primary metric | Query-view mean | Scene-query mean | Primary delta to A | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---|"]
    for arm in "DE":
        d = overall[f"{arm}_minus_A_diagnostic"]
        lo, hi = d["ci95_image_cluster_bootstrap"]
        lines.append(f"| {arm} (diagnostic only) | {overall['arms'][arm]['image_then_scene_macro']:.6f} | "
            f"{overall['arms'][arm]['query_mean']:.6f} | {overall['arms'][arm]['scene_query_mean']:.6f} | "
            f"{d['delta_image_then_scene_macro']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "| Scene | Views | Scene/prompts | A | C | C-A | C set changed |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for scene, g in summary["scenes"].items():
        lines.append(f"| {scene} | {g['n']} | {g['scene_prompt_units']} | {g['arms']['A']['image_then_scene_macro']:.6f} | "
            f"{g['arms']['C']['image_then_scene_macro']:.6f} | {g['C_minus_A_main']['delta_image_then_scene_macro']:+.6f} | {g['AC_changed_query_views']} |")
    lines += ["", "Family means below use within-family query-view means, not the global primary weighting.", "",
        "| Family | Views | Scene/prompts | A mean | C mean | C-A |", "|---|---:|---:|---:|---:|---:|"]
    for family, g in summary["families"].items():
        a, c = (g["arms"][x]["query_mean"] for x in "AC")
        lines.append(f"| {family} | {g['n']} | {g['scene_prompt_units']} | {a:.6f} | {c:.6f} | {c-a:+.6f} |")
    lines += ["", "## Same-query candidate pairs", "",
        "Pairs compare IoU≥.5 positive masks with IoU≤.1 negatives. Raw-correct means r_pos>r_neg; "
        "margin-correct means m_pos>m_neg. Canon-hurt is raw-positive/margin-negative; canon-help is the reverse. "
        "Exact score ties are reported separately and are not counted correct. These are repeated pair occurrences, not independent observations.", "",
        "| Group | Pair occurrences | Views with pairs | Scene/prompts with pairs | Raw correct | Margin correct | Canon hurts | Canon helps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in [("Overall", overall)] + list(summary["scenes"].items()) + list(summary["families"].items()):
        p = g["mechanism"]
        lines.append(f"| {name} | {p['pairs']} | {p['query_views_with_pairs']} | {p['scene_prompt_units_with_pairs']} | "
            f"{p['raw_correct_pairs']} | {p['margin_correct_pairs']} | {p['canon_hurts_pairs']} | {p['canon_helps_pairs']} |")
    p = overall["mechanism"]
    lines += ["", f"Distinct oriented scene/prompt/candidate pairs: {p['distinct_oriented_scene_prompt_candidate_pairs']}. "
        f"Raw ties: {p['raw_tied_pair_occurrences']}; margin ties: {p['margin_tied_pair_occurrences']}. "
        "Distinct pairs still share queries/scenes and are not an independent statistical N.", "",
        "| Positive-mask rank diagnostic | Eligible views | Scene/prompts | Raw improved | Raw worsened | Tied | Mean margin rank | Mean raw rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, g in p["positive_mask_ranks"].items():
        lines.append(f"| {label} | {g['n_query_views']} | {g['n_scene_prompt_units']} | {g['improved']} | {g['worsened']} | "
            f"{g['tied']} | {fmt(g['mean_rank_margin'])} | {fmt(g['mean_rank_raw'])} |")
    lines += ["", "First-positive rank is the highest-ranked mask with IoU≥.5. Best-positive rank uses the highest GT IoU "
        "among positive masks, with IoU ties within 1e-8 credited at their earliest rank. Both compare raw r against margin m; "
        "candidate order resolves exact score ties as in the worker.", "", "## Integrity and scope", ""]
    lines += [f"- {k}: {v}" for k, v in summary["integrity"].items()]
    lines += [f"- {overall['n']} query-view evaluations reuse {overall['scene_prompt_units']} scene/prompt decisions; "
        "family scene/prompt counts may overlap. They are not independent new-scene samples.",
        f"- Bootstrap: shared early-root implementation, {BOOTSTRAPS} paired image-cluster draws within scene, seed {SEED}.",
        "- score<.5 is not NAG rejection; r/b may be dot products of legacy preprocessed features rather than unit-vector cosines.",
        "- D/E and ranking diagnostics do not override the frozen C-A endpoint; the same four scenes have been explored repeatedly."]
    if summary["missing_files"] or summary["count_mismatches"]:
        lines += ["", "Incomplete inputs:"] + [f"- {x}" for x in summary["missing_files"] + summary["count_mismatches"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows, inventory = read_inputs(args.input)
    summary = build_summary(rows, inventory, args.input)
    write_outputs(args.output, summary, render_report(summary))
    compact = {k: summary[k] for k in ("status", "n", "main_verdict")}
    if rows:
        compact.update(C_minus_A=summary["overall"]["C_minus_A_main"]["delta_image_then_scene_macro"],
            ci95=summary["overall"]["C_minus_A_main"]["ci95_image_cluster_bootstrap"],
            baseline_error_max=summary["integrity"]["baseline_error_max"],
            AB_exact=summary["integrity"]["AB_iou_exact_equal"] and summary["integrity"]["AB_sets_exact_equal"],
            scene_prompt_units=summary["overall"]["scene_prompt_units"])
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
