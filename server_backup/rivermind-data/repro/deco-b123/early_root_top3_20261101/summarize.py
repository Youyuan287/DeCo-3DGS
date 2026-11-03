"""Summarize the paired early-root experiment; no inference or GT selection.

Default inputs: results/<scene>_early_root.csv and optional *choices*.jsonl.
Without inputs this writes a WAITING_FOR_INPUTS status, not numerical results.
Arm A is the experiment's original main/gap baseline, NOT the gateE anchor.
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
BOOTSTRAPS = 2000
SEED = 20261101
TOL = 1e-8
FIELDS = (
    "scene", "image", "prompt", "A_iou", "B_iou", "A_pool_oracle",
    "B_pool_oracle", "A_pool_n", "B_pool_n", "root_changed",
    "B_obs_roots", "B_fallback_slots", "baseline_error", "family",
)
NUMERIC = tuple(k for k in FIELDS if k not in ("scene", "image", "prompt", "family"))


def finite_number(value, key, location):
    value = str(value).strip()
    if value.lower() in ("true", "false"):
        result = float(value.lower() == "true")
    else:
        result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"nonfinite {key} at {location}: {value!r}")
    return result


def read_inputs(input_root):
    rows, missing, counts = [], [], {}
    seen = set()
    for scene in SCENES:
        path = input_root / f"{scene}_early_root.csv"
        if not path.exists():
            missing.append(str(path))
            counts[scene] = 0
            continue
        scene_rows = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            absent = set(FIELDS) - set(reader.fieldnames or ())
            if absent:
                raise ValueError(f"missing columns in {path}: {sorted(absent)}")
            for line, raw in enumerate(reader, 2):
                row = {key: raw[key] for key in FIELDS}
                if row["scene"] != scene:
                    raise ValueError(f"scene mismatch in {path}:{line}")
                for key in NUMERIC:
                    row[key] = finite_number(row[key], key, f"{path}:{line}")
                for key in ("A_iou", "B_iou", "A_pool_oracle", "B_pool_oracle"):
                    if not -TOL <= row[key] <= 1 + TOL:
                        raise ValueError(f"invalid IoU {key} in {path}:{line}")
                identity = (scene, row["image"], row["prompt"])
                if identity in seen:
                    raise ValueError(f"duplicate query-view: {identity}")
                seen.add(identity)
                scene_rows.append(row)
        counts[scene] = len(scene_rows)
        rows.extend(scene_rows)
    return rows, missing, counts


def aggregate(rows, key):
    """Image means within scenes, then equal scene means (available scenes)."""
    by_scene_image = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scene_image[row["scene"]][row["image"]].append(row[key])
    image_means = {
        scene: {image: float(np.mean(values)) for image, values in sorted(images.items())}
        for scene, images in sorted(by_scene_image.items())
    }
    scene_means = {
        scene: float(np.mean(list(images.values()))) for scene, images in image_means.items()
    }
    return {
        "image_then_scene_macro": float(np.mean(list(scene_means.values()))) if rows else None,
        "query_mean": float(np.mean([r[key] for r in rows])) if rows else None,
        "scene_query_mean": float(np.mean([
            np.mean([r[key] for r in rows if r["scene"] == scene])
            for scene in sorted(by_scene_image)
        ])) if rows else None,
        "scene_means": scene_means,
        "image_means": image_means,
        "scenes_present": sorted(by_scene_image),
    }


def paired(rows, first, second):
    """Resample images within each scene; A/B remain paired in every draw."""
    differences = np.asarray([r[first] - r[second] for r in rows], dtype=float)
    by_scene_image = defaultdict(lambda: defaultdict(list))
    for row, difference in zip(rows, differences):
        by_scene_image[row["scene"]][row["image"]].append(float(difference))
    arrays = [
        np.asarray([np.mean(values) for _, values in sorted(images.items())], dtype=float)
        for _, images in sorted(by_scene_image.items())
    ]
    rng = np.random.default_rng(SEED)
    samples = np.zeros(BOOTSTRAPS, dtype=float)
    for values in arrays:
        indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        samples += values[indices].mean(axis=1) / len(arrays)
    return {
        "first": first, "second": second,
        "delta_image_then_scene_macro": float(np.mean([a.mean() for a in arrays])),
        "delta_query_mean": float(differences.mean()),
        "ci95_image_cluster_bootstrap": np.percentile(samples, (2.5, 97.5)).tolist(),
        "wins": int(np.sum(differences > TOL)),
        "losses": int(np.sum(differences < -TOL)),
        "ties": int(np.sum(np.abs(differences) <= TOL)),
        "bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
        "unit": "image clusters resampled within each scene; equal scene weighting",
    }


def subset_summary(rows):
    return {
        "n": len(rows),
        "A": aggregate(rows, "A_iou"), "B": aggregate(rows, "B_iou"),
        "B_minus_A": paired(rows, "B_iou", "A_iou"),
        "A_pool_oracle": aggregate(rows, "A_pool_oracle"),
        "B_pool_oracle": aggregate(rows, "B_pool_oracle"),
        "B_minus_A_pool_oracle": paired(rows, "B_pool_oracle", "A_pool_oracle"),
        "root_changed_query_views": sum(r["root_changed"] > 0.5 for r in rows),
        "root_changed_scene_prompt_units": len({
            (r["scene"], r["prompt"]) for r in rows if r["root_changed"] > 0.5
        }),
        "baseline_error_max": float(max(abs(r["baseline_error"]) for r in rows)),
        "mean_A_pool_n": float(np.mean([r["A_pool_n"] for r in rows])),
        "mean_B_pool_n": float(np.mean([r["B_pool_n"] for r in rows])),
        "mean_B_obs_roots": float(np.mean([r["B_obs_roots"] for r in rows])),
        "mean_B_fallback_slots": float(np.mean([r["B_fallback_slots"] for r in rows])),
        "queries_with_B_fallback_slots": sum(r["B_fallback_slots"] > 0 for r in rows),
    }


def choice_inventory(input_root):
    files = []
    for path in sorted(input_root.glob("*choices*.jsonl")):
        count = 0
        with path.open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid choice JSON at {path}:{line_number}") from exc
                    count += 1
        files.append({"path": str(path), "records": count})
    return files


def write_outputs(output_root, summary, report):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")


def render_report(summary):
    results = summary["overall"]
    delta = results["B_minus_A"]
    ci = delta["ci95_image_cluster_bootstrap"]
    lines = [
        "# Early-root top-3 paired experiment", "",
        f"Status: **{summary['status']}**. Rows: {summary['n']}/208.", "",
        "Arm A is this experiment's original main/gap baseline. The historical gateE 0.6760 "
        "is an independent anchor; neither arm is identified as gateE by this summary.", "",
        "Primary metric: query mean within image, image mean within scene, then equal mean "
        "over the four scenes. Partial inputs are explicitly provisional.", "",
        "| Metric | A | B | B - A |", "|---|---:|---:|---:|",
    ]
    for title, key in (
        ("Image/scene macro (primary)", "image_then_scene_macro"),
        ("All-query mean", "query_mean"),
        ("Mean of scene-query means", "scene_query_mean"),
    ):
        a, b = results["A"][key], results["B"][key]
        lines.append(f"| {title} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    a = results["A_pool_oracle"]["image_then_scene_macro"]
    b = results["B_pool_oracle"]["image_then_scene_macro"]
    lines.extend([
        f"| Pool oracle, image/scene macro | {a:.6f} | {b:.6f} | {b-a:+.6f} |", "",
        f"B-A primary paired 95% CI: [{ci[0]:+.6f}, {ci[1]:+.6f}]; "
        f"{delta['wins']} wins / {delta['losses']} losses / {delta['ties']} ties.",
        "Pool oracle is GT-based diagnostic candidate capacity, not deployable accuracy.", "",
        "## Per scene", "",
        "| Scene | n | A | B | Delta | A pool oracle | B pool oracle | Roots changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for scene, item in summary["scenes"].items():
        a, b = item["A"]["image_then_scene_macro"], item["B"]["image_then_scene_macro"]
        lines.append(f"| {scene} | {item['n']} | {a:.6f} | {b:.6f} | {b-a:+.6f} | "
                     f"{item['A_pool_oracle']['image_then_scene_macro']:.6f} | "
                     f"{item['B_pool_oracle']['image_then_scene_macro']:.6f} | "
                     f"{item['root_changed_query_views']} |")
    lines.extend([
        "", "## Families", "",
        "Family rows below use within-family query means, not the primary global macro. "
        "Families are taken unchanged from the input CSV.", "",
        "| Family | n | A query mean | B query mean | Delta | Wins | Losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for family, item in summary["families"].items():
        a, b = item["A"]["query_mean"], item["B"]["query_mean"]
        lines.append(f"| {family} | {item['n']} | {a:.6f} | {b:.6f} | {b-a:+.6f} | "
                     f"{item['B_minus_A']['wins']} | {item['B_minus_A']['losses']} |")
    lines.extend([
        "", "## Integrity and intervention", "",
        f"- Maximum baseline reproduction error: {results['baseline_error_max']:.12g}.",
        f"- Changed root sets: {results['root_changed_query_views']} query-views, "
        f"{results['root_changed_scene_prompt_units']} distinct scene/prompt units.",
        f"- Mean pool size A/B: {results['mean_A_pool_n']:.3f}/{results['mean_B_pool_n']:.3f}.",
        f"- Mean B observation roots: {results['mean_B_obs_roots']:.3f}.",
        f"- Mean B fallback slots: {results['mean_B_fallback_slots']:.3f}; "
        f"{results['queries_with_B_fallback_slots']} rows have a fallback slot.",
        f"- Bootstrap: {BOOTSTRAPS} paired image-cluster draws within scenes, seed {SEED}.",
        "- Statistical intervals describe these paired scenes, not new-scene generalization.",
    ])
    if summary["missing_files"]:
        lines.extend(["", "Missing inputs:"] + [f"- `{x}`" for x in summary["missing_files"]])
    if summary["count_mismatches"]:
        lines.extend(["", "Count mismatches:"] + [f"- {x}" for x in summary["count_mismatches"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    rows, missing, counts = read_inputs(args.input)
    summary = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "input_root": str(args.input), "n": len(rows),
        "expected_counts": EXPECTED_COUNTS, "actual_counts": counts,
        "missing_files": missing,
        "count_mismatches": [f"{s}: got {counts[s]}, expected {EXPECTED_COUNTS[s]}"
                             for s in SCENES if counts[s] != EXPECTED_COUNTS[s]],
        "choices_files": choice_inventory(args.input),
        "gateE_anchor_is_independent": True,
    }
    if not rows:
        summary["status"] = "WAITING_FOR_INPUTS"
        write_outputs(args.output, summary,
                      "# Early-root top-3 paired experiment\n\n"
                      "Status: **WAITING_FOR_INPUTS**. No experiment rows are available.\n"
                      "Expected results/<scene>_early_root.csv for all four scenes.\n"
                      "No numerical result, improvement, or gateE reproduction is claimed.\n")
    else:
        summary["status"] = "COMPLETE" if not summary["count_mismatches"] else "PARTIAL_INPUTS"
        summary["overall"] = subset_summary(rows)
        if summary["overall"]["baseline_error_max"] > 1e-6:
            summary["status"] = "BASELINE_MISMATCH"
        summary["scenes"] = {s: subset_summary([r for r in rows if r["scene"] == s])
                             for s in SCENES if counts[s]}
        summary["families"] = {f: subset_summary([r for r in rows if r["family"] == f])
                               for f in sorted({r["family"] for r in rows})}
        write_outputs(args.output, summary, render_report(summary))
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
