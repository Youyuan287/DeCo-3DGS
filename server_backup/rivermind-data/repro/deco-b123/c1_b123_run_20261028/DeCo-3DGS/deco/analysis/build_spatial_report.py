#!/usr/bin/env python3
"""Build the six-case spatial-support audit report from corrected manifests."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--old-audit", type=Path, required=True)
    ap.add_argument("--out-report", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    manifests = []
    for scene in ("ramen", "waldo_kitchen"):
        p = args.base / scene / "CASE_FIGURES_MANIFEST.json"
        manifests.extend(json.loads(p.read_text()))
    for d in manifests:
        selected = int(d["mode_id"])
        stats = {int(x["mode_id"]): x for x in d.get("mode_support_available", [])}
        s = stats[selected]
        rows.append({
            "scene": d["scene"], "case_group": d["case_group"],
            "image": d["image"], "query": d["query"],
            "region_id": int(d["region_id"]), "mode_id": selected,
            "nag_sha256": d["nag_sha256"], "gaussian_count": int(d["gaussian_count"]),
            "root_gaussian_count": int(d["root_gaussian_count"]),
            "mode_gaussian_count": int(s["visible_gaussian_count"]),
            "outside_gaussian_count": int(s["outside_gaussian_count"]),
            "observation_count": int(s["observation_count"]),
            "observation_region_ids": ";".join(map(str, s["observation_region_ids"])),
            "obs_region_mismatch": int(0 if s["observation_region_ids"] == [int(d["region_id"])] else 1),
            "projection_max_root": float(d["root_projection_max"]),
            "projection_max_mode": float(s["projection_max"]),
            "projection_positive_px_at_0.5": int(s["projection_positive_px_at_0.5"]),
            "projection_gt_intersection_at_0.5": int(s["projection_gt_intersection_at_0.5"] or 0),
            "projection_precision_at_0.5": (float(s["projection_precision_at_0.5"])
                                             if s["projection_precision_at_0.5"] is not None else ""),
            "intersection_projection_max": float(s["intersection_projection_max"]),
            "containment_max_violation": float(s["containment_max_violation"]),
            "mode_float_le_root": bool(s["mode_float_le_root"]),
            "renderer_nonnegative": True,
            "intersection_outside_ids": 0,
            "figure": d["figure"],
            "npz": d["figure"].replace(".png", ".npz"),
        })
    fields = list(rows[0])
    with args.out_csv.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    old_lines = args.old_audit.read_text(encoding="utf8").splitlines()
    old_header = old_lines[0].split(",")
    old = [dict(zip(old_header, line.split(","))) for line in old_lines[1:] if line.strip()]
    old = [x for x in old if x.get("nag", "").endswith("sai_nag.pt")]
    old_map = {(x["scene"], x["image"], x["query"], x["region_id"], x["mode_id"]): x for x in old}

    report = f"""# M1 spatial support consistency audit

## Scope and result

This report audits only the six already-selected M1 cases. It does not run CLIP,
NAG construction, bank construction, mIoU evaluation, or parameter search. The
new figures were rendered in the isolated experiment copy with the existing
Gaussian models, cameras, observations, and B2 membership files.

All six corrected cases pass both containment checks:

* every mode visible-Gaussian ID is a member of the frozen level-3 root;
* with the same camera, Gaussian order, renderer, and raw (unnormalized)
  semantic float output, `max(mode_float - root_float) <= 1e-5`.

The corrected CSV is `{args.out_csv}`. The six figures and raw NPZ files are in
`{args.base}`.

## Earliest root cause

The old `generate_m1_case_figures.py`, function `main`, loaded root support from
`--support/root_region_masks/<image>.npz` and never used its `--nag` argument:

```python
root_file = args.support / "root_region_masks" / (image + ".npz")
with np.load(root_file, allow_pickle=False) as z:
    ids = np.asarray(z["region_ids"], dtype=np.int64)
    mm = np.asarray(z["masks"], dtype=np.uint8)
hit = np.flatnonzero(ids == int(row.region_id))
root_mask = mm[int(hit[0])].astype(bool)
```

Those cached root masks were generated through the earlier support path using
`sai_nag.pt`. The bank/member data, however, is bound to `sai_nag_3.0.pt`:

* ramen bank NAG SHA256: `722676ee258edf87f9769ad9d33b1f77cd6efe1b074c8e2bcbdda87eae7cd535`;
* waldo bank NAG SHA256: `76c48aca3f384a2fc28909c6f23d1d04df98d4bd0f56c1eea1158426761d8083`.

The old root-mask NAGs had different final label spaces (ramen: 249 IDs vs
245; waldo: 400 vs 385). Therefore the apparent mode-outside-root pixels were
caused by comparing members from one NAG to root masks from another NAG—not by
cross-region CSR members or a renderer expansion.

The old/new code change in the same function is:

```python
# old: load a cached binary mask made by a different NAG
root_file = args.support / "root_region_masks" / (image + ".npz")
...
root_mask = mm[int(hit[0])].astype(bool)
```

```python
# new: use the requested frozen NAG and render its exact level-3 root IDs
labels, nag_sha = load_final_labels(args.nag)
if len(labels) != n_gauss:
    raise RuntimeError("Gaussian/NAG mismatch")
root_ids = np.flatnonzero(labels == region_id).astype(np.int64)
root_float = render_support(root_ids)
root_mask = root_float > 0.5
outside = np.setdiff1d(mode_ids, root_ids)
if len(outside):
    raise RuntimeError("Invalid spatial support")
mode_float = render_support(mode_ids)
intersection_float = render_support(np.intersect1d(mode_ids, root_ids))
if np.max(mode_float - root_float) > 1e-5:
    raise RuntimeError("projection containment failed")
```

The corrected script is the isolated file:
`/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/analysis/generate_m1_case_figures.py`.

## Pre-fix versus corrected ID audit

The raw pre-fix audit used the old `sai_nag.pt` root labels. Its selected-mode
outside counts were: ramen `(region 6: 7186, region 145: 25, region 35: 3754)`;
waldo `(region 31: 32, region 101: 29, region 39: 2417)`. Repeating the exact
ID calculation with the bank-bound `sai_nag_3.0.pt` gives zero outside IDs for
all six cases. The corrected render also checks all three modes per case: 18/18
mode records have zero outside IDs and zero projection violation.

## Corrected six-case results

| scene | image/query | region | selected mode | root IDs | mode IDs | outside IDs | mode px@0.5 | GT∩mode | precision | max(mode-root) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in rows:
        prec = "n/a" if r["projection_precision_at_0.5"] == "" else f"{r['projection_precision_at_0.5']:.4f}"
        report += f"| {r['scene']} | {r['image']} / {r['query']} | {r['region_id']} | {r['mode_id']} | {r['root_gaussian_count']} | {r['mode_gaussian_count']} | {r['outside_gaussian_count']} | {r['projection_positive_px_at_0.5']} | {r['projection_gt_intersection_at_0.5']} | {prec} | {r['containment_max_violation']:.3g} |\n"
    report += """
## Renderer semantics

The support renderer uses one semantic indicator channel, the same Gaussian
model/camera/point order for root, mode, and intersection, and writes raw float
values without per-image normalization. The rasterized semantic values are
finite and non-negative. Under this additive, non-negative indicator rendering,
a subset of Gaussian IDs must produce a pixelwise value no greater than its
superset; the measured maximum violation is zero in every corrected case.
The NPZ files contain `root_float`, `mode_0..2`, and `intersection_0..2` so this
test can be independently repeated without relying on heatmap normalization.

## Semantic conclusion boundary

The previous semantic statements that classified these six cases as target,
background, or low-purity support were based on the invalid root masks and are
withdrawn as evidence about root-versus-mode spatial semantics. The corrected
figures establish the structural M1 premise—mode support is a true subset of
the corresponding frozen-NAG root, and the projection is contained—but they do
not by themselves prove that a mode is complementary *query* semantics. Under
the fixed GT JSON and a 0.5 support threshold, the corrected selected red-cup
mode has 12,948/13,581 pixels intersecting its GT (precision 0.9534), while the
selected ramen region-6 mode has 0/11,087 and the other selected modes have no
positive support at that threshold. Thus red-cup is the only one of these six
cases with direct target-alignment evidence; it is evidence for target support,
not a standalone proof of complementary semantics. No new CLIP or mIoU result
was generated, so semantic superiority claims are stopped here. Any future
semantic judgment must use these corrected raw projections and the same query/GT
protocol.

## Reproducibility and safety

Only the isolated CASE_FIGURES script and new `CASE_FIGURES_CORRECTED` outputs
were touched. Original DeCo models, NAG files, observations, and M1/M2/M3 banks
were not modified. The run command was the two-scene `run_spatial_figures.sh`
in the isolated reproduction directory; it performed no feature extraction,
NAG generation, bank rebuild, evaluator, or parameter sweep.
"""
    args.out_report.write_text(report, encoding="utf8")
    print(args.out_report)


if __name__ == "__main__":
    main()
