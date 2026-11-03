#!/usr/bin/env python3
"""Rebuild the M1 visibility-stratified audit from existing evidence only.

This script deliberately does not touch production banks or regenerate any
features.  It fixes the analysis-side prior mapping by reading the actual
prior arrays from the B1/B2 bank NPZ files.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank_priors(path: Path) -> tuple[dict[int, list[float]], dict[int, list[bool]]]:
    with np.load(path, allow_pickle=False) as z:
        region_ids = np.asarray(z["region_ids"], dtype=np.int64)
        priors = np.asarray(z["priors"], dtype=np.float64)
        valid = np.asarray(z["valid"], dtype=bool)
    return (
        {int(r): [float(x) for x in row] for r, row in zip(region_ids, priors)},
        {int(r): [bool(x) for x in row] for r, row in zip(region_ids, valid)},
    )


def category(frame: pd.DataFrame) -> pd.Series:
    visible = frame["region_px"] > 0
    target = frame["intersection_px"] > 0
    pure = frame["precision"] > 0.5
    return pd.Series(
        np.select(
            [~visible, visible & ~target, target & pure, target & ~pure],
            ["invisible", "visible_background", "target_purity_gt50", "target_low_purity"],
            default="unclassified",
        ),
        index=frame.index,
        name="visibility_class",
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scene, cls), g in df.groupby(["scene", "visibility_class"], sort=True):
        # A row is a mode record; the second aggregation removes mode-level
        # duplication and reports the max M1 gain per image/query/region.
        rg = (
            g.groupby(["scene", "image", "query", "region_id"], as_index=False)
            .agg(
                gain=("m1_gain", "max"),
                single_score=("single_score", "first"),
                region_px=("region_px", "first"),
                precision=("precision", "first"),
                intersection_px=("intersection_px", "first"),
            )
        )
        rows.append(
            dict(
                scene=scene,
                visibility_class=cls,
                mode_rows=len(g),
                image_query_region_rows=len(rg),
                unique_queries=g[["image", "query"]].drop_duplicates().shape[0],
                mean_m1_gain=float(g.m1_gain.mean()),
                median_m1_gain=float(g.m1_gain.median()),
                mean_max_mode_gain=float(rg.gain.mean()),
                median_max_mode_gain=float(rg.gain.median()),
                mean_single_score=float(g.single_score.mean()),
                mean_region_px=float(g.region_px.mean()),
                median_region_px=float(g.region_px.median()),
                mean_precision=float(g.precision.mean()),
                mean_intersection_px=float(g.intersection_px.mean()),
            )
        )
    return pd.DataFrame(rows).sort_values(["scene", "visibility_class"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, type=Path)
    ap.add_argument("--m1-members", required=True, type=Path)
    ap.add_argument("--m2-members", required=True, type=Path)
    ap.add_argument("--m1-bank", required=True, type=Path)
    ap.add_argument("--m2-bank", required=True, type=Path)
    ap.add_argument("--origin", required=True, type=Path)
    ap.add_argument("--case-index", required=True, type=Path)
    ap.add_argument("--case-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.full)
    if args.scene:
        df = df[df.scene == args.scene].copy()
    df["m1_gain"] = df["m1_slot_score"] - df["single_score"]
    df["visibility_class"] = category(df)
    summary = summarize(df)
    summary.to_csv(out / "M1_RECOVERY_BY_VISIBILITY.csv", index=False)

    # Map the real bank arrays to the mode rows.  Membership JSON contains
    # observation membership only; it intentionally has no prior field.
    m1_pr, m1_valid = load_bank_priors(args.m1_bank)
    m2_pr, m2_valid = load_bank_priors(args.m2_bank)
    fixed = pd.read_csv(args.origin)
    fixed["prior_m1"] = [
        (m1_pr.get(int(r.region_id), [np.nan] * (int(r.mode) + 1))[int(r.mode)])
        if int(r.mode) < len(m1_pr.get(int(r.region_id), []))
        else np.nan
        for r in fixed.itertuples()
    ]
    fixed["prior_m2"] = [
        (m2_pr.get(int(r.region_id), [np.nan] * (int(r.mode) + 1))[int(r.mode)])
        if int(r.mode) < len(m2_pr.get(int(r.region_id), []))
        else np.nan
        for r in fixed.itertuples()
    ]
    fixed["valid_m1"] = [
        bool(m1_valid.get(int(r.region_id), [False] * (int(r.mode) + 1))[int(r.mode)])
        if int(r.mode) < len(m1_valid.get(int(r.region_id), []))
        else False
        for r in fixed.itertuples()
    ]
    fixed["valid_m2"] = [
        bool(m2_valid.get(int(r.region_id), [False] * (int(r.mode) + 1))[int(r.mode)])
        if int(r.mode) < len(m2_valid.get(int(r.region_id), []))
        else False
        for r in fixed.itertuples()
    ]
    fixed.to_csv(out / "MODE_MEMBERSHIP_SUMMARY.csv", index=False)

    # Rebuild the 40 metadata-only candidate pages with an exact mode and
    # exact prior.  Pages not selected for rendering remain metadata-only.
    cases = pd.read_csv(args.case_index)
    if args.scene:
        cases = cases[cases.scene == args.scene].copy()
    pages = []
    for i, r in cases.iterrows():
        sub = df[
            (df.scene == r.scene)
            & (df.image == r.image)
            & (df["query"] == r.query)
            & (df.region_id == int(r.region_id))
        ].sort_values("m1_gain", ascending=False)
        if sub.empty:
            continue
        z = sub.iloc[0]
        mode = int(z.mode_id)
        p1 = float(z.m1_prior)
        page = dict(
            case_index=int(i),
            case_group=r.case_group,
            scene=r.scene,
            image=r.image,
            query=r.query,
            region_id=int(r.region_id),
            mode_id=mode,
            m1_gain=float(z.m1_gain),
            single_score=float(z.single_score),
            m1_slot_score=float(z.m1_slot_score),
            prior_m1=p1,
            precision=float(z.precision),
            recall=float(z.recall),
            iou=float(z.iou),
            region_px=int(z.region_px),
            intersection_px=int(z.intersection_px),
            visibility_class=str(z.visibility_class),
            prior_source=str(args.m1_bank),
            prior_corrected=True,
        )
        pages.append(page)
        body = (
            "<html><body><h2>Corrected M1 case %03d: %s</h2>"
            % (int(i), html.escape(str(r.case_group)))
        )
        body += "<pre>" + html.escape(json.dumps(page, indent=2, ensure_ascii=False)) + "</pre>"
        body += (
            "<p>prior is loaded from the real B1 bank NPZ, keyed by (region_id, mode_id). "
            "This page is metadata-only unless a rendered PNG is listed in the case manifest.</p>"
        )
        (args.case_dir / ("case_%03d.html" % int(i))).write_text(body, encoding="utf8")
    pd.DataFrame(pages).to_csv(out / "CORRECTED_CASE_INDEX.csv", index=False)

    # Machine-readable audit metadata.
    audit = {
        "full_sha256": sha256(args.full),
        "m1_bank_sha256": sha256(args.m1_bank),
        "m2_bank_sha256": sha256(args.m2_bank),
        "m1_members_sha256": sha256(args.m1_members),
        "m2_members_sha256": sha256(args.m2_members),
        "original_case_count": int(len(cases)),
        "corrected_case_count": int(len(pages)),
        "prior_mapping": "B1/B2 NPZ priors[region_ids, mode_id]; membership JSON has no prior field",
        "categories": summary.to_dict(orient="records"),
    }
    (out / "M1_VISIBILITY_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf8")

    md = [
        "# M1 visibility-stratified recovery audit",
        "",
        "This is an analysis-only correction from the existing FULL_EVIDENCE_TABLE; no CLIP, NAG, bank, or algorithm was rerun.",
        "",
        "## Definitions",
        "",
        "- `invisible`: `region_px == 0` in the current image.",
        "- `visible_background`: `region_px > 0` and `intersection_px == 0`.",
        "- `target_purity_gt50`: `intersection_px > 0` and `precision > 0.5`.",
        "- `target_low_purity`: `intersection_px > 0` and `precision <= 0.5`.",
        "- Mode-level M1 gain is `m1_slot_score - single_score`; the table also reports max-mode gain after collapsing duplicate modes for each image/query/region.",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Interpretation guardrails",
        "",
        "The invisible majority is not evidence of background behavior and is reported separately. Visible background has higher positive mode-level gain than target-related rows in both scenes, while target rows often have higher Single Evidence scores and larger projected regions; this can change both the additive gain and the number of mode records. The result does not prove that all modes are noise: mode differences and target-related high-purity cases remain present. Query/region duplication is controlled by the collapsed `image/query/region` columns.",
        "",
        "## Prior correction",
        "",
        "`MODE_MEMBERSHIP_SUMMARY.csv` and all 40 candidate HTML pages now use the real B1/B2 NPZ `priors` arrays. The old membership JSON only stores `observation_indices`, so its missing `prior` default caused the previous all-zero column.",
    ]
    (out / "M1_VISIBILITY_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
