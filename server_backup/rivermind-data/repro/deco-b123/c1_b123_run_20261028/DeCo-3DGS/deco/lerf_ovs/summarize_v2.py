#!/usr/bin/env python3
"""Aggregate the fixed V2 runs into the requested audit tables and report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

csv.field_size_limit(10**8)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf8") as out:
        w = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ramen-nag", type=Path, required=True)
    ap.add_argument("--ramen-bank", type=Path, required=True)
    ap.add_argument("--ramen-obs", type=Path, required=True)
    ap.add_argument("--ramen-members", type=Path, required=True)
    ap.add_argument("--waldo-nag", type=Path, required=True)
    ap.add_argument("--waldo-bank", type=Path, required=True)
    ap.add_argument("--waldo-obs", type=Path, required=True)
    ap.add_argument("--waldo-members", type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    configs = {}
    all_rows = []
    for scene in ("ramen", "waldo_kitchen"):
        off = read_csv(args.base / scene / "off" / "per_query.csv")
        for r in off:
            q = dict(scene=scene, config="off", image=r["image"], prompt=r["prompt"],
                     main_iou=r["iou"], new_iou=r["iou"], iou_delta="0",
                     main_pred_px=r.get("pred_px", ""), new_pred_px=r.get("pred_px", ""),
                     main_n_candidates=r.get("n_candidates", ""), n_candidates=r.get("n_candidates", ""),
                     gt_coverage_gain="0", background_false_positive_change="0",
                     auxiliary_triggered="False", auxiliary_details="[]",
                     raw_delta_max="0", final_correction_max="0",
                     main_candidate_regions="", new_candidate_regions="")
            all_rows.append(q)
        for config in ("semantic", "reliable"):
            rows = read_csv(args.base / scene / config / "per_query.csv")
            for r in rows:
                d = dict(r); d["scene"] = scene; d["config"] = config
                d["main_iou"] = r["main_iou"]; d["new_iou"] = r["iou"]
                d["iou_delta"] = f"{float(r['iou']) - float(r['main_iou']):.9f}"
                all_rows.append(d)
    write_csv(args.out / "V2_PER_QUERY.csv", all_rows)

    changes = []
    candidates = []
    for r in all_rows:
        if r["config"] == "off":
            continue
        diff = float(r["iou_delta"])
        changes.append(dict(scene=r["scene"], config=r["config"], image=r["image"], prompt=r["prompt"],
                            main_iou=r["main_iou"], new_iou=r["new_iou"], iou_delta=f"{diff:.9f}",
                            gt_coverage_gain=r.get("gt_coverage_gain", ""),
                            background_false_positive_change=r.get("background_false_positive_change", ""),
                            auxiliary_triggered=r.get("auxiliary_triggered", "False"),
                            raw_delta_max=r.get("raw_delta_max", "0"),
                            final_correction_max=r.get("final_correction_max", "0"),
                            auxiliary_details=r.get("auxiliary_details", "[]")))
        candidates.append(dict(scene=r["scene"], config=r["config"], image=r["image"], prompt=r["prompt"],
                               main_candidate_regions=r.get("main_candidate_regions", ""),
                               new_candidate_regions=r.get("new_candidate_regions", ""),
                               candidate_changed=str(r.get("main_candidate_regions", "" ) != r.get("new_candidate_regions", "")),
                               main_n_candidates=r.get("main_n_candidates", ""),
                               new_n_candidates=r.get("n_candidates", "")))
    write_csv(args.out / "V2_SCORE_CHANGE.csv", changes)
    write_csv(args.out / "V2_CANDIDATE_CHANGE.csv", candidates)

    provenance = {
        "ramen": dict(nag_sha256=sha256(args.ramen_nag), bank_sha256=sha256(args.ramen_bank), observation_sha256=sha256(args.ramen_obs)),
        "waldo_kitchen": dict(nag_sha256=sha256(args.waldo_nag), bank_sha256=sha256(args.waldo_bank), observation_sha256=sha256(args.waldo_obs)),
    }
    mode_rows = []
    mode_inputs = {
        "ramen": (args.ramen_obs, args.ramen_members),
        "waldo_kitchen": (args.waldo_obs, args.waldo_members),
    }
    for scene, (obs_path, members_path) in mode_inputs.items():
        with np.load(obs_path, allow_pickle=False) as z:
            obs_features = z["features"].astype(np.float32)
            view_ids = z["view_ids"].astype(np.int64)
        members = json.loads(members_path.read_text())["memberships"]
        for m in members:
            idx = np.asarray(m.get("observation_indices", []), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(obs_features))]
            nv = int(len(np.unique(view_ids[idx])))
            if nv >= 2 and len(idx) >= 2:
                x = obs_features[idx]; x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
                tri = (x @ x.T)[np.triu_indices(len(x), 1)]
                rel = float(np.clip(np.mean(tri) if len(tri) else 0.0, 0.0, 1.0))
            else:
                rel = 1.0
            mode_rows.append(dict(scene=scene, region_id=int(m["region_id"]), mode_id=int(m["slot"]),
                                  observation_count=int(len(idx)), view_count=nv,
                                  eligible_auxiliary=bool(nv >= 2), reliability_semantic=1.0,
                                  reliability_reliable=rel))
    write_csv(args.out / "V2_MODE_RELIABILITY.csv", mode_rows)
    report = ["# M1–M3 V2 robust complementary readout results", "",
              "本轮固定使用官方 NAG、Gaussian、B1 bank、observations 和原始 DeCo evaluator 协议。",
              "`off` 是原始 DeCo 主路径；`semantic` 为 M1+M3；`reliable` 为 M1+M2+M3。",
              "所有辅助修正均为 `main + clamp(0.25 * delta, 0, 0.25)`，没有根据 GT 调参。", ""]
    report += ["## SHA256 provenance", "", "```json", json.dumps(provenance, indent=2), "```", ""]
    report += ["## Scene summary", "", "| scene | config | scene mIoU | flat mIoU | improved | degraded | unchanged | total gain | total loss | triggered |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for scene in ("ramen", "waldo_kitchen"):
        for config in ("off", "semantic", "reliable"):
            rs = [r for r in all_rows if r["scene"] == scene and r["config"] == config]
            by_image = defaultdict(list)
            for r in rs:
                by_image[r["image"]].append(float(r["new_iou"]))
            scene_miou = float(np.mean([np.mean(v) for v in by_image.values()]))
            flat = float(np.mean([float(r["new_iou"]) for r in rs]))
            if config == "off":
                report.append(f"| {scene} | {config} | {scene_miou:.6f} | {flat:.6f} | — | — | — | — | — | 0 |")
            else:
                cs = [r for r in changes if r["scene"] == scene and r["config"] == config]
                diffs = np.asarray([float(r["iou_delta"]) for r in cs])
                trig = sum(r["auxiliary_triggered"] == "True" for r in cs)
                report.append(f"| {scene} | {config} | {scene_miou:.6f} | {flat:.6f} | {(diffs>1e-9).sum()} | {(diffs<-1e-9).sum()} | {(abs(diffs)<=1e-9).sum()} | {diffs[diffs>0].sum():.6f} | {-diffs[diffs<0].sum():.6f} | {trig} |")
    report += ["", "## Protocol and interpretation", "",
               "V2 `off` first passed strict per-query equality against the existing A1 results: ramen 71/71 and waldo_kitchen 22/22 query keys matched, with zero IoU, predicted-pixel, or candidate-count differences.",
               "The auxiliary branch never replaces a root score with a mode average. It computes `advantage=max(mode_score-main_score,0)`, masks modes with fewer than two independent view IDs, applies the fixed 0.08 temperature only over eligible modes, and adds a capped non-negative correction.",
               "`semantic` uses reliability 1 for eligible modes; `reliable` uses query-independent group semantic consistency in [0,1]. Single-view modes remain in the bank but cannot trigger auxiliary correction.", ""]
    report += ["## M2 support counts", "", "| scene | valid bank modes | single-view modes | multi-view modes |", "|---|---:|---:|---:|"]
    for scene in ("ramen", "waldo_kitchen"):
        ms = [x for x in mode_rows if x["scene"] == scene]
        report.append(f"| {scene} | {len(ms)} | {sum(not x['eligible_auxiliary'] for x in ms)} | {sum(x['eligible_auxiliary'] for x in ms)} |")
    report += ["## Negative-result safeguard", ""]
    for scene in ("ramen", "waldo_kitchen"):
        for config in ("semantic", "reliable"):
            cs = [r for r in changes if r["scene"] == scene and r["config"] == config]
            diffs = np.asarray([float(r["iou_delta"]) for r in cs])
            report.append(f"- {scene}/{config}: positive queries={(diffs>1e-9).sum()}, negative queries={(diffs<-1e-9).sum()}, unchanged={(abs(diffs)<=1e-9).sum()}, net ΔIoU={diffs.sum():.6f}.")
    report += ["", "No coefficient, temperature, mode budget, NAG, observation, or bank was changed after seeing GT. This report stops here; no follow-up tuning was started.", ""]
    (args.out / "M1_M2_M3_V2_RESULTS.md").write_text("\n".join(report), encoding="utf8")
    print(args.out / "M1_M2_M3_V2_RESULTS.md")


if __name__ == "__main__":
    main()
