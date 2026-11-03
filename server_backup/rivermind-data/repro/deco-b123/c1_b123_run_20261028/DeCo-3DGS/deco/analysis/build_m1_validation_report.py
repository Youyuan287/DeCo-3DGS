#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, cols=None) -> str:
    if cols:
        df = df[cols]
    return "```text\n" + df.to_string(index=False) + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    args = ap.parse_args()
    root, final = args.root, args.final
    if final.exists():
        shutil.rmtree(final)
    final.mkdir(parents=True)

    old_final = root / "final_review"
    full = pd.read_csv(old_final / "FULL_EVIDENCE_TABLE.csv")
    full["m1_gain"] = full.m1_slot_score - full.single_score
    full["visibility_class"] = "unclassified"
    full.loc[full.region_px == 0, "visibility_class"] = "invisible"
    full.loc[(full.region_px > 0) & (full.intersection_px == 0), "visibility_class"] = "visible_background"
    full.loc[(full.intersection_px > 0) & (full.precision > 0.5), "visibility_class"] = "target_purity_gt50"
    full.loc[(full.intersection_px > 0) & (full.precision <= 0.5), "visibility_class"] = "target_low_purity"

    stats = pd.concat([
        pd.read_csv(root / "revised_ramen" / "M1_RECOVERY_BY_VISIBILITY.csv"),
        pd.read_csv(root / "revised_waldo" / "M1_RECOVERY_BY_VISIBILITY.csv"),
    ], ignore_index=True)
    stats.to_csv(final / "M1_RECOVERY_BY_VISIBILITY.csv", index=False)
    full.to_csv(final / "FULL_EVIDENCE_TABLE_WITH_VISIBILITY.csv", index=False)
    shutil.copy2(old_final / "FULL_EVIDENCE_TABLE.csv", final / "FULL_EVIDENCE_TABLE.csv")
    if (old_final / "M2_REMOVED_DETAIL.csv").exists():
        shutil.copy2(old_final / "M2_REMOVED_DETAIL.csv", final / "M2_REMOVED_DETAIL.csv")

    mode = pd.concat([
        pd.read_csv(root / "revised_ramen" / "MODE_MEMBERSHIP_SUMMARY.csv"),
        pd.read_csv(root / "revised_waldo" / "MODE_MEMBERSHIP_SUMMARY.csv"),
    ], ignore_index=True)
    mode.to_csv(final / "MODE_MEMBERSHIP_SUMMARY.csv", index=False)
    cases = pd.read_csv(Path("/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/analysis/SELECTED_M1_CASES.csv"))
    cases.to_csv(final / "SELECTED_M1_CASES.csv", index=False)

    # Copy corrected 40 metadata pages.
    pages = final / "CORRECTED_CASE_HTML"
    pages.mkdir()
    for scene, d in [("ramen", root / "revised_ramen"), ("waldo_kitchen", root / "revised_waldo")]:
        for p in (d / "semantic_evidence_cases").glob("*.html"):
            shutil.copy2(p, pages / f"{scene}_{p.name}")
    # Copy exactly six rendered figures and manifests.
    figs = final / "CASE_FIGURES"
    figs.mkdir()
    manifests = []
    for scene, d in [("ramen", root / "revised_ramen" / "case_figures"), ("waldo_kitchen", root / "revised_waldo" / "case_figures")]:
        for p in d.glob("*"):
            if p.suffix.lower() in {".png", ".json", ".npz"} or p.name == "CASE_FIGURES_MANIFEST.json":
                shutil.copy2(p, figs / f"{scene}_{p.name}")
        manifests.extend(json.loads((d / "CASE_FIGURES_MANIFEST.json").read_text()))
    (figs / "CASE_FIGURES_MANIFEST.json").write_text(json.dumps(manifests, indent=2, ensure_ascii=False))
    mode_projection_rows = []
    for x in manifests:
        selected = next(m for m in x["mode_support_available"] if int(m["mode_id"]) == int(x["mode_id"]))
        mode_projection_rows.append({
            "scene": x["scene"], "image": x["image"], "query": x["query"],
            "region_id": int(x["region_id"]), "mode_id": int(x["mode_id"]),
            "root_precision": float(x["precision"]),
            "projection_max": float(selected["projection_max"]),
            "projection_px_at_0.5": int(selected["projection_positive_px_at_0.5"]),
            "projection_GT_intersection_at_0.5": int(selected["projection_gt_intersection_at_0.5"] or 0),
            "projection_precision_at_0.5": selected["projection_precision_at_0.5"],
        })
    mode_projection = pd.DataFrame(mode_projection_rows)
    mode_projection.to_csv(final / "SELECTED_MODE_PROJECTION_AUDIT.csv", index=False)

    # Build a concise report with explicit objective semantic labels.
    def stats_rows(scene):
        return stats[stats.scene == scene][["visibility_class", "mode_rows", "image_query_region_rows", "unique_queries", "mean_m1_gain", "mean_max_mode_gain", "mean_single_score", "mean_region_px", "mean_precision"]]

    target = full[full.intersection_px > 0]
    bg = full[(full.region_px > 0) & (full.intersection_px == 0)]
    corr_lines = []
    for scene, g in full[full.region_px > 0].groupby("scene"):
        corr = g[["m1_gain", "single_score", "region_px"]].corr(method="spearman")
        corr_lines.append(
            f"- `{scene}` visible rows: Spearman(gain, Single Evidence) = `{corr.loc['m1_gain','single_score']:.4f}`, Spearman(gain, region area) = `{corr.loc['m1_gain','region_px']:.4f}`."
        )
    report = [
        "# M1_TARGET_SEMANTIC_VALIDATION",
        "",
        "## Scope and stopping rule",
        "",
        "This is an analysis-only correction. It reuses the existing FULL_EVIDENCE_TABLE, frozen observations, existing B1/B2 bank files, GT annotations, camera files, root-region masks, and the same Gaussian renderer used by the support render. It does not rerun CLIP, construct NAG, rebuild banks, change M1/M2/M3, or scan parameters.",
        "",
        "The row-level M1 gain is `m1_slot_score - single_score`. Invisible regions are separated before any interpretation. A second aggregation collapses duplicate mode rows to one max-mode gain per `(scene,image,query,region_id)`.",
        "",
        "## Four-class M1 recovery summary",
        "",
        "The counts below are intentionally shown both as mode rows and as image/query/region records to expose mode and query duplication.",
        "",
        "### ramen",
        md_table(stats_rows("ramen")),
        "",
        "### waldo_kitchen",
        md_table(stats_rows("waldo_kitchen")),
        "",
        "## Why visible background gain can exceed target gain",
        "",
        "1. The invisible majority is not semantic background: it has `region_px == 0` and no current-image support. Including it would mix absence of evidence with visible background.",
        "2. In both scenes, visible background has lower Single Evidence scores than target-related rows, leaving more additive headroom for a mode score. For ramen the visible-background mean Single Evidence is 0.4059 versus 0.4825 over all target-related mode rows; for waldo it is 0.3652 versus 0.5020. This makes a larger raw gain possible without proving stronger target semantics.",
        "3. Projected area differs materially: visible-background mean region area is 13,694.9 px (ramen) and 7,131.1 px (waldo), while target-related mode rows average 66,239.6 px and 66,216.4 px respectively in the original evidence. The low-purity and high-purity target strata are also very different in area, so area is a confounder for raw additive gain.",
        "4. Mode and query repetition is controlled rather than hidden: `mode_rows` can be compared with `image_query_region_rows`, and `unique_queries` is reported. The same region appearing in several query images is not treated as independent scene-level evidence in the collapsed column.",
        "5. Rank-correlation diagnostic on visible rows (not a causal correction):",
        *corr_lines,
        "",
        "Therefore the result is a target-selectivity warning, not proof that all modes are noise. The evidence contains high-purity target-aligned modes and also high-gain background/mixed modes.",
        "",
        "## Prior-field correction",
        "",
        "The previous `MODE_MEMBERSHIP_SUMMARY.csv` used `x.get('prior', 0.0)` on membership JSON entries. Those entries contain observation indices but no prior field, so every exported prior became zero. The corrected file loads `priors` from the real B1 and B2 NPZ arrays using `(region_id, mode_id)`; the backward-compatible `prior` column is now the B1 prior, while `prior_m1` and `prior_m2` preserve both bank values. Production banks are untouched.",
        f"- Corrected ramen rows: `{len(pd.read_csv(root / 'revised_ramen' / 'MODE_MEMBERSHIP_SUMMARY.csv'))}`; corrected waldo rows: `{len(pd.read_csv(root / 'revised_waldo' / 'MODE_MEMBERSHIP_SUMMARY.csv'))}`.",
        f"- In the corrected backward-compatible `prior` column: `{int((mode.prior > 0).sum())}` nonzero values, `{int((mode.prior == 0).sum())}` legitimate exact zeros, and `{int(mode.prior.isna().sum())}` unmapped rows where the corresponding root is absent from that bank; these are no longer silently converted to zero.",
        f"- Corrected HTML pages: `{len(list(pages.glob('*.html')))}`; each page records `prior_m1`, `mode_id`, and its NPZ source.",
        "- The six selected figure manifests also carry the corrected `m1_prior` from FULL_EVIDENCE_TABLE.",
        "",
        "## Six fixed cases",
        "",
        md_table(cases[["case_group", "scene", "image", "query", "region_id", "mode_id", "recover_gain", "precision", "region_px", "intersection_px"]]),
        "",
        "The selection is from the existing FULL_EVIDENCE_TABLE/support evidence universe. The previous 40-page heuristic did not include a waldo high-purity target row among its five `target_recovery` pages; the waldo sink row above is nevertheless an existing support/evidence row, not a newly scored example. This limitation is recorded rather than hidden.",
        "",
        "## Exact current-view mode-support projection",
        "",
        "The following table is from the renderer output for the selected M1 mode, not from the coarse root mask. `projection_precision_at_0.5` uses the exact visible-Gaussian support map thresholded at 0.5; a zero maximum means that this mode's observation-union has no current-view rasterized support under the unmodified renderer.",
        "",
        md_table(mode_projection),
        "",
        "## Semantic case judgments",
        "",
        "- **High-purity root cases:** ramen `(frame_00128, glass of water, region 6, mode 1)` has root precision 0.5320, but its selected mode projection has 11,087 pixels at 0.5 and zero GT intersection. Waldo `(frame_00154, sink, region 31, mode 1)` has root precision 0.9834, but the selected mode has zero current-view projection. Thus coarse-root purity alone does not establish that the high-score mode observes the target.",
        "- **Visible-background cases:** ramen `(frame_00128, glass of water, region 145, mode 1)` and waldo `(frame_00053, yellow desk, region 101, mode 2)` have nonzero root projection but zero GT intersection; the selected mode projections are also empty in the current view. Their scores cannot be interpreted as target observation from these images.",
        "- **Low-purity target cases:** ramen `(frame_00128, glass of water, region 35, mode 1)` has root precision 0.0596 and an empty selected-mode projection. Waldo `(frame_00154, red cup, region 39, mode 2)` has root precision 0.0923, but its mode support has 13,581 pixels, 12,948 GT-intersection pixels, and 0.9534 support precision. This is a concrete example showing that a low-purity coarse root can still contain a target-aligned mode; it is why “all modes are noise” is invalid.",
        "",
        "## Answers to the four required questions",
        "",
        "1. **Does a high-score mode actually observe the query target?** The six-case audit is mixed: direct mode-support/GT alignment is observed for waldo red cup (0.9534 support precision at 0.5), while ramen glass-of-water and waldo sink high-purity roots do not show selected-mode/GT overlap in the current view.",
        "2. **Is a high-score mode mainly background or mixed support?** Several selected modes have no current-view support, and the ramen high-purity selected mode projects away from its GT. The waldo red-cup mode is target-aligned despite a low-purity root. These observations rule out both universal target selectivity and universal noise.",
        "3. **Is the data sufficient to explain mode-score advantage as true complementary semantics?** No. The same evidence contains target-aligned, background-only, and mixed supports, and lacks manual semantic labels for all modes. The score advantage is therefore not semantically identified.",
        "4. **Which cases support or contradict M1?** Waldo red-cup mode 2 supports a target-aligned mode; ramen mode 1 for glass-of-water, waldo sink mode 1, and both visible-background cases contradict a universal target-selectivity claim. The precise causal source remains unresolved.",
        "",
        "## Figure and evidence paths",
        "",
        "- `CASE_FIGURES/`: six RGB + GT + root-region + mode-visible-Gaussian support montages and JSON manifests.",
        "- `SELECTED_MODE_PROJECTION_AUDIT.csv`: exact current-view support/GT overlap for the selected mode in each case.",
        "- `CORRECTED_CASE_HTML/`: all 40 metadata pages with corrected priors; pages not among the six selected cases remain metadata-only.",
        "- `M1_RECOVERY_BY_VISIBILITY.csv`: four-class numeric summary.",
        "- `FULL_EVIDENCE_TABLE.csv`: unchanged source evidence; `FULL_EVIDENCE_TABLE_WITH_VISIBILITY.csv` adds only analysis labels.",
        "",
        "Completed and stopped after this validation; no algorithm or parameter changes were made.",
    ]
    (final / "M1_TARGET_SEMANTIC_VALIDATION.md").write_text("\n".join(report) + "\n", encoding="utf8")
    (final / "CASE_GALLERY.md").write_text(
        "# Case gallery\n\n" + "\n".join(
            f"- `{r.scene}` `{r.image}` query=`{r.query}` region=`{int(r.region_id)}` mode=`{int(r.mode_id)}`: see `CASE_FIGURES/` and manifest."
            for r in cases.itertuples()
        ) + "\n",
        encoding="utf8",
    )
    (final / "FILES_FOR_CHATGPT.md").write_text(
        "# Corrected review package\n\n"
        "- `M1_TARGET_SEMANTIC_VALIDATION.md`\n"
        "- `M1_RECOVERY_BY_VISIBILITY.csv`\n"
        "- `MODE_MEMBERSHIP_SUMMARY.csv`\n"
        "- `FULL_EVIDENCE_TABLE.csv` and `FULL_EVIDENCE_TABLE_WITH_VISIBILITY.csv`\n"
        "- `CASE_FIGURES/` (six real renderer montages)\n"
        "- `CORRECTED_CASE_HTML/` (40 pages with corrected priors)\n",
        encoding="utf8",
    )
    zip_path = root / "CHATGPT_REVIEW_PACKAGE_CORRECTED.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in final.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(final.parent))
    print("wrote", final, "zip", zip_path)


if __name__ == "__main__":
    main()
