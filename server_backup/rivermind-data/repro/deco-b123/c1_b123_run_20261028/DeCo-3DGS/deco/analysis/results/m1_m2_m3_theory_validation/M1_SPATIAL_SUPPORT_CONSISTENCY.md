# M1 spatial support consistency audit

## Scope and result

This report audits only the six already-selected M1 cases. It does not run CLIP,
NAG construction, bank construction, mIoU evaluation, or parameter search. The
new figures were rendered in the isolated experiment copy with the existing
Gaussian models, cameras, observations, and B2 membership files.

All six corrected cases pass both containment checks:

* every mode visible-Gaussian ID is a member of the frozen level-3 root;
* with the same camera, Gaussian order, renderer, and raw (unnormalized)
  semantic float output, `max(mode_float - root_float) <= 1e-5`.

The corrected CSV is `/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/analysis/results/m1_m2_m3_theory_validation/M1_SPATIAL_SUPPORT_CHECK.csv`. The six figures and raw NPZ files are in
`/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/analysis/results/m1_m2_m3_theory_validation/CASE_FIGURES_CORRECTED`.

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
| ramen | frame_00128 / glass of water | 35 | 1 | 4514 | 3754 | 0 | 0 | 0 | n/a | 0 |
| ramen | frame_00128 / glass of water | 6 | 1 | 7409 | 7186 | 0 | 11087 | 0 | 0.0000 | 0 |
| ramen | frame_00128 / glass of water | 145 | 1 | 25 | 25 | 0 | 0 | 0 | n/a | 0 |
| waldo_kitchen | frame_00154 / red cup | 39 | 2 | 7574 | 2418 | 0 | 13581 | 12948 | 0.9534 | 0 |
| waldo_kitchen | frame_00154 / sink | 31 | 1 | 5176 | 32 | 0 | 0 | 0 | n/a | 0 |
| waldo_kitchen | frame_00053 / yellow desk | 101 | 2 | 1015 | 29 | 0 | 0 | 0 | n/a | 0 |

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
