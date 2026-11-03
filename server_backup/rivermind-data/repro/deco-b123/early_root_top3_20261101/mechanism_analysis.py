"""Analyze already rendered A/B candidates; never run inference or select a method.

The matched-pool oracle is max single-mask IoU over A_candidates | B_candidates.
It is a GT diagnostic, not the IoU of a union of masks or a deployable selector.
Only MECHANISM.json and MECHANISM.md are written. SUMMARY.json is read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


TOL = 1e-8
SCENES = ("figurines", "ramen", "teatime", "waldo_kitchen")
EXPECTED = dict(zip(SCENES, (56, 71, 59, 22)))
METRICS = ("A_iou", "B_iou", "A_pool_oracle", "B_pool_oracle", "AB_pool_oracle")


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite input")
    return value


def candidate_id(pair):
    if len(pair) != 2:
        raise ValueError(f"invalid candidate ID: {pair}")
    return f"{int(pair[0])}:{int(pair[1])}"


def sign(value):
    return "gain" if value > TOL else "loss" if value < -TOL else "tie"


def candidate_mechanism(detail):
    a = {candidate_id(p) for p in detail["A_candidates"]}
    b = {candidate_id(p) for p in detail["B_candidates"]}
    values = {k: number(v) for k, v in detail["candidate_ious"].items()}
    if set(values) != a | b:
        raise ValueError("candidate_ious must cover exactly the matched candidate union")
    if any(v < -TOL or v > 1 + TOL for v in values.values()):
        raise ValueError("candidate IoU outside [0,1]")
    amax = max((values[k] for k in a), default=0.0)
    bmax = max((values[k] for k in b), default=0.0)
    best = {k for k in a if abs(values[k] - amax) <= TOL}
    retained = best & b
    new = b - a
    better = {k for k in new if values[k] > amax + TOL}
    return {
        "A_candidates": sorted(a), "B_candidates": sorted(b),
        "A_best_candidates": sorted(best), "A_best_retained_candidates": sorted(retained),
        "A_best_lost_candidates": sorted(best - b),
        "A_best_any_retained": bool(retained),
        "A_best_all_retained": bool(best) and best <= b,
        "A_best_all_lost": bool(best) and not retained,
        "A_best_positive_all_lost": amax > TOL and bool(best) and not retained,
        "A_best_zero_all_lost": amax <= TOL and bool(best) and not retained,
        "A_best_empty": not best,
        "B_new_candidates": sorted(new), "B_new_better_candidates": sorted(better),
        "B_has_new_better_than_A_best": bool(better),
        "B_new_better_from_zero_A": bool(better) and amax <= TOL,
        "B_new_better_from_positive_A": bool(better) and amax > TOL,
        "B_new_best_iou": max((values[k] for k in new), default=None),
        "recomputed_A_pool_oracle": amax,
        "recomputed_B_pool_oracle": bmax,
        "AB_pool_oracle": max(amax, bmax),
        "pool_change": sign(bmax - amax),
        "A_roots": detail["A_roots"], "B_roots": detail["B_roots"],
        "A_selected": detail["A_selected"], "B_selected": detail["B_selected"],
    }


def load_rows(root):
    rows = []
    for scene in SCENES:
        choices = {}
        with (root / f"{scene}_choices.jsonl").open(encoding="utf-8-sig") as handle:
            for line in handle:
                d = json.loads(line)
                key = (d["image"], d["prompt"])
                if key in choices:
                    raise ValueError(f"duplicate choices: {scene}, {key}")
                choices[key] = d
        seen = set()
        with (root / f"{scene}_early_root.csv").open(encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                key = (raw["image"], raw["prompt"])
                if raw["scene"] != scene or key in seen:
                    raise ValueError(f"scene mismatch or duplicate CSV: {scene}, {key}")
                seen.add(key)
                row = dict(raw)
                for field in set(raw) - {"scene", "image", "prompt", "family"}:
                    row[field] = number(raw[field])
                row.update(candidate_mechanism(choices[key]))
                for arm in "AB":
                    if abs(row[f"{arm}_pool_oracle"] - row[f"recomputed_{arm}_pool_oracle"]) > TOL:
                        raise ValueError(f"pool-oracle mismatch: {scene}, {key}, {arm}")
                    if len(row[f"{arm}_candidates"]) != row[f"{arm}_pool_n"]:
                        raise ValueError(f"pool-size mismatch: {scene}, {key}, {arm}")
                row["delta_iou"] = row["B_iou"] - row["A_iou"]
                row["delta_pool_oracle"] = row["B_pool_oracle"] - row["A_pool_oracle"]
                row["AB_minus_A_pool_oracle"] = row["AB_pool_oracle"] - row["A_pool_oracle"]
                row["AB_minus_B_pool_oracle"] = row["AB_pool_oracle"] - row["B_pool_oracle"]
                rows.append(row)
        if seen != set(choices):
            raise ValueError(f"CSV/choices keys differ for {scene}")
        if len(seen) != EXPECTED[scene]:
            raise ValueError(f"incomplete {scene}: {len(seen)}/{EXPECTED[scene]}")
    return rows


def aggregate(rows, field):
    images = defaultdict(lambda: defaultdict(list))
    for r in rows:
        images[r["scene"]][r["image"]].append(r[field])
    by_scene = {s: mean(mean(v) for v in ims.values()) for s, ims in images.items()}
    by_unit = defaultdict(list)
    for r in rows:
        by_unit[(r["scene"], r["prompt"])].append(r[field])
    unit_means = {k: mean(v) for k, v in by_unit.items()}
    return {
        "image_then_scene_macro": mean(by_scene.values()) if rows else None,
        "query_view_mean": mean(r[field] for r in rows) if rows else None,
        "scene_query_view_mean": mean(mean(r[field] for r in rows if r["scene"] == s)
                                      for s in images) if rows else None,
        "scene_prompt_mean": mean(unit_means.values()) if rows else None,
        "scene_prompt_then_scene_macro": mean(mean(v for (ss, _), v in unit_means.items() if ss == s)
                                              for s in images) if rows else None,
        "scene_image_macro": by_scene,
        "images_present_by_scene": {s: len(v) for s, v in images.items()},
    }


def group_report(rows):
    flags = ("A_best_any_retained", "A_best_all_retained", "A_best_all_lost",
             "A_best_positive_all_lost", "A_best_zero_all_lost", "A_best_empty",
             "B_has_new_better_than_A_best", "B_new_better_from_zero_A", "B_new_better_from_positive_A")
    out = {"n_query_views": len(rows),
           "n_scene_prompt_units": len({(r["scene"], r["prompt"]) for r in rows}),
           "metrics": {k: aggregate(rows, k) for k in METRICS},
           "deltas": {k: aggregate(rows, k) for k in
                      ("delta_iou", "delta_pool_oracle", "AB_minus_A_pool_oracle", "AB_minus_B_pool_oracle")},
           "query_view_counts": {k: sum(r[k] for r in rows) for k in flags},
           "pool_change_counts": dict(Counter(r["pool_change"] for r in rows)),
           "final_iou_change_counts": dict(Counter(sign(r["delta_iou"]) for r in rows)),
           "candidate_occurrences": {
               "A_best": sum(len(r["A_best_candidates"]) for r in rows),
               "A_best_lost": sum(len(r["A_best_lost_candidates"]) for r in rows),
               "B_new": sum(len(r["B_new_candidates"]) for r in rows),
               "B_new_better_than_A_best": sum(len(r["B_new_better_candidates"]) for r in rows)},
           "cross_counts": dict(Counter(
               ("best_retained" if r["A_best_any_retained"] else "best_all_lost" if r["A_best_all_lost"]
                else "A_empty") + "/" + r["pool_change"] for r in rows)),
           "final_change_by_retention": dict(Counter(
               ("best_retained" if r["A_best_any_retained"] else "best_all_lost" if r["A_best_all_lost"]
                else "A_empty") + "/" + sign(r["delta_iou"]) for r in rows)),
           "global_primary_contributions": {
               k: sum(r["global_primary_weight"] * r[k] for r in rows)
               for k in ("delta_iou", "delta_pool_oracle", "AB_minus_A_pool_oracle", "AB_minus_B_pool_oracle")}}
    for flag in flags:
        units = defaultdict(list)
        for r in rows:
            units[(r["scene"], r["prompt"])].append(bool(r[flag]))
        out.setdefault("scene_prompt_unit_counts", {})[flag] = {
            "at_least_one_evaluated_view": sum(any(v) for v in units.values()),
            "every_evaluated_view": sum(all(v) for v in units.values())}
    return out


def analyze(rows, summary):
    images = defaultdict(Counter)
    units = defaultdict(list)
    for r in rows:
        images[r["scene"]][r["image"]] += 1
        units[(r["scene"], r["prompt"])].append(r)
    unit_records = []
    for (scene, prompt), rs in sorted(units.items()):
        signatures = {json.dumps({k: r[k] for k in
                      ("A_candidates", "B_candidates", "A_roots", "B_roots", "A_selected", "B_selected")},
                                sort_keys=True) for r in rs}
        if len(signatures) != 1:
            raise ValueError(f"query-time choices differ across views: {scene}/{prompt}")
        unit_records.append({"scene": scene, "prompt": prompt, "n_views": len(rs),
            "families": dict(Counter(r["family"] for r in rs)),
            "mean_delta_iou": mean(r["delta_iou"] for r in rs),
            "mean_delta_pool_oracle": mean(r["delta_pool_oracle"] for r in rs),
            "A_best_lost_views": sum(r["A_best_all_lost"] for r in rs),
            "B_new_better_views": sum(r["B_has_new_better_than_A_best"] for r in rs)})
    for r in rows:
        r["global_primary_weight"] = 1 / len(SCENES) / len(images[r["scene"]]) / images[r["scene"]][r["image"]]
    overall = group_report(rows)
    primary = overall["deltas"]["delta_iou"]["image_then_scene_macro"]
    ci = summary["overall"]["B_minus_A"]["ci95_image_cluster_bootstrap"]
    for field, stored in (("A_iou", "A"), ("B_iou", "B"),
                          ("A_pool_oracle", "A_pool_oracle"), ("B_pool_oracle", "B_pool_oracle")):
        if abs(overall["metrics"][field]["image_then_scene_macro"] -
               summary["overall"][stored]["image_then_scene_macro"]) > 1e-10:
            raise ValueError(f"SUMMARY.json does not match CSV: {field}")
    scene_deltas = overall["deltas"]["delta_iou"]["scene_image_macro"]
    checks = {"primary_B_minus_A_positive": primary > 0,
              "bootstrap_lower_strictly_positive": ci[0] > 0,
              "no_scene_decrease_exceeding_0_01": all(v >= -0.01 for v in scene_deltas.values())}
    return {"generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE", "frozen_policy_verdict": "PASS" if all(checks.values()) else "FAIL",
        "frozen_checks": checks, "primary_B_minus_A": primary, "primary_ci95_from_SUMMARY": ci,
        "tolerance": TOL, "data_validation": {"candidate_iou_keys_and_CSV_oracles_matched": True,
        "SUMMARY_primary_metrics_matched": True, "all_scene_prompt_choices_constant_across_views": True,
        "maximum_baseline_error": max(r["baseline_error"] for r in rows)},
        "units": {"query_views": len(rows), "scene_prompts": len(units),
            "images": sum(len(v) for v in images.values()),
            "scene_prompt_counts": dict(Counter(s for s, _ in units)),
            "family_scene_prompt_counts_may_overlap": True},
        "scope": ["Only previously collected CSV and candidate_ious were analyzed; no new inference.",
            "Matched-pool oracle=max(A_pool_oracle,B_pool_oracle): one candidate mask, not a union of masks.",
            "Oracle uses GT for diagnosis only; no post-hoc routing/gating or method score is proposed.",
            "Final gap selections may contain multiple masks; a single-mask oracle is not a bound on final IoU.",
            "208 view-level evaluations reuse 67 scene/prompt retrieval decisions; they are not 208 independent decisions.",
            "67 scene/prompt units share scenes and are not asserted to be independent new-scene samples.",
            "Family labels are inherited per query-view; one scene/prompt can appear in multiple families.",
            "Arm A is original raw-gap main; gateE .6760107 is a separate historical anchor.",
            "Frozen policy failure does not prove missing information or failure of all mask selectors."],
        "overall": overall,
        "by_scene": {s: group_report([r for r in rows if r["scene"] == s]) for s in SCENES},
        "by_family": {f: group_report([r for r in rows if r["family"] == f])
                      for f in sorted({r["family"] for r in rows})},
        "by_family_group": {name: group_report([r for r in rows if r["family"] in fs])
                      for name, fs in {"recall_deep": {"recall", "deep"},
                                       "found_rank2": {"found", "rank2"}}.items()},
        "scene_prompt_units": unit_records, "query_view_details": rows}


def markdown(report):
    overall = report["overall"]
    c = overall["query_view_counts"]
    ci = report["primary_ci95_from_SUMMARY"]
    lines = ["# Early-root top-3: existing-data mechanism analysis", "",
        f"Frozen continuation verdict: **{report['frozen_policy_verdict']}**. "
        f"B-A primary IoU {report['primary_B_minus_A']:+.6f}, existing paired 95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}].", "",
        "No new inference, feature extraction or post-hoc gate was run. This report does not revise the frozen decision rule.", "",
        f"Evaluations: {report['units']['query_views']} query-views, {report['units']['scene_prompts']} distinct scene/prompt retrieval decisions, "
        f"{report['units']['images']} evaluated images, four scenes. Candidate sets and selections are identical across evaluated views of each scene/prompt. "
        "The 208 rows are not 208 independent retrieval decisions; the 67 units also share scenes.", "",
        "## Candidate preservation", "",
        f"At IoU tolerance {TOL:g}, at least one A-best candidate is retained in B on **{c['A_best_any_retained']} / 208** query-views; "
        f"all tied A-best candidates are lost on **{c['A_best_all_lost']} / 208** "
        f"({c['A_best_positive_all_lost']} with positive A-best IoU). "
        f"All tied A-best candidates are retained on {c['A_best_all_retained']} views. "
        f"New B-only nodes strictly exceed A-best on **{c['B_has_new_better_than_A_best']} / 208** views.", "",
        f"The {c['A_best_zero_all_lost']} all-lost views whose A pool has zero IoU count candidate-ID loss, not loss of positive GT coverage. "
        f"Of the 17 new-better views, {c['B_new_better_from_zero_A']} start with zero A-pool IoU and "
        f"{c['B_new_better_from_positive_A']} starts with positive A-pool IoU.", "",
        f"At the scene/prompt level, A-best is lost in at least one evaluated view for "
        f"{overall['scene_prompt_unit_counts']['A_best_all_lost']['at_least_one_evaluated_view']} / 67 units "
        f"({overall['scene_prompt_unit_counts']['A_best_all_lost']['every_evaluated_view']} in every evaluated view). "
        f"Positive A-best is lost in at least one view for "
        f"{overall['scene_prompt_unit_counts']['A_best_positive_all_lost']['at_least_one_evaluated_view']} units. "
        f"New-better B nodes occur for {overall['scene_prompt_unit_counts']['B_has_new_better_than_A_best']['at_least_one_evaluated_view']} / 67 units.", "",
        "Counts below refer to evaluated query-views, not independent prompts. A-best means every A candidate within 1e-8 of the maximum; retaining any tie preserves that value.", "",
        "| Group | Views | Scene/prompts | Any A-best retained | All A-best lost | New B node better | B pool gain/loss/tie | Final gain/loss/tie |",
        "|---|---:|---:|---:|---:|---:|---|---|"]
    for name, g in [("Overall", overall)] + list(report["by_scene"].items()) + list(report["by_family"].items()):
        q, pc, fc = g["query_view_counts"], g["pool_change_counts"], g["final_iou_change_counts"]
        triplet = lambda d: "/".join(str(d.get(k, 0)) for k in ("gain", "loss", "tie"))
        lines.append(f"| {name} | {g['n_query_views']} | {g['n_scene_prompt_units']} | {q['A_best_any_retained']} | "
                     f"{q['A_best_all_lost']} | {q['B_has_new_better_than_A_best']} | {triplet(pc)} | {triplet(fc)} |")
    lines += ["", "## Matched A∪B single-mask capacity", "",
        "For each same query-view, U=max(A_pool_oracle,B_pool_oracle). This is the GT-best **single existing candidate** in A∪B. "
        "It is not a multi-mask union oracle, a new method, or deployable accuracy. "
        "The actual gap-selected prediction can contain multiple candidates and exceed this single-mask diagnostic.", "",
        "All numbers in the next table use query mean within image → image mean within scene → mean across represented scenes. "
        "A family subset uses its represented images/scenes; its conditional mean is not an additive contribution to the global metric.", "",
        "| Group | A final | B final | A pool | B pool | U pool | U-A pool | U-B pool |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in [("Overall", overall)] + list(report["by_scene"].items()) + list(report["by_family"].items()):
        v = [g["metrics"][k]["image_then_scene_macro"] for k in METRICS]
        vals = v + [v[4] - v[2], v[4] - v[3]]
        lines.append("| " + name + " | " + " | ".join(f"{x:.6f}" for x in vals) + " |")
    lines += ["", "## Recall/deep gains versus found/rank2 losses", "",
        "Global contributions retain each row's original four-scene/image weight; the two group contributions sum to the full primary change. "
        "Conditional query-view means describe each group and use different weights.", "",
        "| Group | Views | Final Δ, query-view mean | Pool Δ, query-view mean | Final Δ, global contribution | Pool Δ, global contribution | New-better views | A-best-lost views |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, g in report["by_family_group"].items():
        d, w, q = g["deltas"], g["global_primary_contributions"], g["query_view_counts"]
        lines.append(f"| {name} | {g['n_query_views']} | {d['delta_iou']['query_view_mean']:+.6f} | "
            f"{d['delta_pool_oracle']['query_view_mean']:+.6f} | {w['delta_iou']:+.6f} | "
            f"{w['delta_pool_oracle']:+.6f} | {q['B_has_new_better_than_A_best']} | {q['A_best_all_lost']} |")
    lines += ["", "| Family | Final Δ, query-view mean | Pool Δ, query-view mean | U pool, query-view mean |",
        "|---|---:|---:|---:|"]
    for name, g in report["by_family"].items():
        lines.append(f"| {name} | {g['deltas']['delta_iou']['query_view_mean']:+.6f} | "
            f"{g['deltas']['delta_pool_oracle']['query_view_mean']:+.6f} | "
            f"{g['metrics']['AB_pool_oracle']['query_view_mean']:.6f} |")
    lines += ["", "All 17 strictly better new-node views belong to recall; deep introduces no better single-mask candidate "
        "and its pool oracle decreases despite improved final predictions. The final-IoU change therefore cannot be explained solely "
        "by added candidate capacity. Found/rank2 lose 21 positive A-best candidates and introduce no better candidate. "
        f"Across all families, final IoU still decreases on {overall['final_change_by_retention'].get('best_retained/loss', 0)} views "
        "where an A-best candidate remains available. Candidate preservation alone is insufficient to explain the final selection outcome."]
    lines += ["", "## Unit and interpretation limits", ""]
    for note in report["scope"]:
        lines.append("- " + note)
    lines += ["", "The JSON retains per-query-view candidate IDs, tied-best retention, scene/prompt view counts, "
        "query-view/scene-prompt means and additive family contributions for checking these descriptions. "
        "Any new selection policy would require a separately preregistered experiment.", ""]
    return "\n".join(lines)


def self_test():
    # Tied A-best: retaining either tie must preserve capacity; a strictly better new node is separate.
    d = {"A_candidates": [[3, 1], [3, 2]], "B_candidates": [[3, 2], [2, 4]],
         "candidate_ious": {"3:1": .5, "3:2": .5 + 5e-9, "2:4": .7},
         "A_roots": [], "B_roots": [], "A_selected": [], "B_selected": []}
    r = candidate_mechanism(d)
    assert r["A_best_any_retained"] and not r["A_best_all_lost"] and not r["A_best_all_retained"]
    assert len(r["A_best_candidates"]) == 2 and r["B_has_new_better_than_A_best"]
    assert r["AB_pool_oracle"] == .7
    rs = [{"scene": "s", "image": "i1", "prompt": "p1", "v": 0},
          {"scene": "s", "image": "i1", "prompt": "p2", "v": 0},
          {"scene": "s", "image": "i2", "prompt": "p1", "v": 1}]
    assert aggregate(rs, "v")["image_then_scene_macro"] == .5
    assert aggregate(rs, "v")["query_view_mean"] == 1 / 3
    print("Self-test passed: tied-best retention, matched single-mask oracle, unequal image weighting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    rows = load_rows(args.root / "results")
    summary = json.loads((args.root / "SUMMARY.json").read_text(encoding="utf-8-sig"))
    report = analyze(rows, summary)
    report["input_root"] = str((args.root / "results").resolve())
    (args.root / "MECHANISM.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                        allow_nan=False) + "\n", encoding="utf-8")
    (args.root / "MECHANISM.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"verdict": report["frozen_policy_verdict"], "units": report["units"],
        "counts": report["overall"]["query_view_counts"],
        "AB_pool_oracle_primary": report["overall"]["metrics"]["AB_pool_oracle"]["image_then_scene_macro"]}, indent=2))


if __name__ == "__main__":
    main()
