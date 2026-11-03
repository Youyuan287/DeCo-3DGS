# PREREG — Mask-level aggregation rerank (frozen before computing)

Goal: replace region-mixture observation aggregation with per-view object-level (SAM mask)
feature aggregation, and test whether it fixes the aggregation bottleneck.

Data: `lerf_segidentity_20261030/<scene>_segidentity.csv` (Phase B dump). For each candidate in the
released pool it stores `id_top3` = mean of the top-3 per-view **mask** feature similarities to the
query text (object-level, no GT used in the score), plus `rank` (released order), `level`, `root_id`,
`cand_id`. GT is used only for evaluation IoU and family labels.

Frozen rule (per query):
- k = released gap length on the NAG candidate scores (unchanged; no re-truncation).
- S = z(NAG score) + lambda * z(id_top3), lambda = 1.0 (z within the query's candidate list).
- Select the top-k candidates by S, render their union, evaluate IoU.
No parameter tuning; lambda sensitivities {0.5, 2.0} reported but not used to choose.

Two anchored bases, both evaluated with `eval_lerf_ovs.py --text canon --renorm 0`:
- ReLaGS: `--center none` (macro 64.01 expected on the released readout);
- DeCo B0: `--center image` (macro 63.9263).
The rerank script must use the same center setting as the evaluation.

Gates (per base): scene-mean delta >= +0.01; paired scene-stratified bootstrap (B=2000,
seed=20261030) 95% CI lower > 0; no scene worse than -0.01; found family not worse by >0.005.
If it fails: report negative and stop this rerank form (mask-level aggregation may still be tested
as a feature replacement, a separate prereg).

Notes:
- This tests aggregation **as a reranking signal**; it does not rebuild features.
- The prior observation-level F1b (id_top3 from region observations) failed at 0.6225 vs 0.6396
  (C1-on main). Mask-level features are expected to differ because they are object-level.
- Do not use `agree`/`agree_c` in the score; the diagnostic shows consistency aggregation is the
  weak component.