# PREREG — Modified ROFA inside ReLaGS's own pipeline (frozen before computing)

Scope: modify `merge_proj.py::_filter_feature_outliers` in the ReLaGS repository and evaluate
through ReLaGS's own LERF-OVS harness (`scripts/eval_lerf_relags.py`, topk=5, level_until=3,
filter_small=50, remove_small=True). tau stays at the released deployment value 3.

Why a rebuild is required: the released `sai_nag.pt` stores only aggregated features, not per-view
observations, so ROFA cannot be edited post hoc. The controlled A/B therefore runs the semantic
projection (merge_proj) three times on the same geometry (`nag-l1.pt`, point cloud) and the same
LangSplat language features, differing only in the ROFA mode. Absolute mIoU is therefore not
comparable to the 64.01 released-readout reproduction; only the within-rebuild contrast is
interpretable, and that is the contrast we test.

## Arms (per scene, same inputs, same tau=3)
- `raw`    : released ROFA. q_i on raw per-view features; keep q_i > mu - tau*sigma; weighted mean.
- `center` : consistency q_i computed on unit per-view centered features
             (unit(f) - view mean of unit per-view features, then unit); keep the same rule;
             aggregate the RAW weighted features of kept views. (primary variant)
- `off`    : no outlier filtering; weighted mean of all views (ablation of ROFA itself).

## Primary hypothesis and gate (center vs raw)
Scene-mean mIoU over the 208 queries (per-image mean, then scene mean):
1. delta >= +0.01
2. paired scene-stratified bootstrap (B=2000, seed=20261030) 95% CI lower > 0
3. no scene worse than -0.01
If all pass -> the centered-consistency modification improves ROFA inside ReLaGS (provisional,
4 scenes). If any fail -> report negative and keep released ROFA.

## Secondary (descriptive, no adoption)
- `off` vs `raw`: the actual contribution of released ROFA in this rebuild (does it help at all?).
- Removal statistics from `rofa_stats.json`: obs removed / total obs, units touched.
- Per-query win/loss counts and the identity of flipped queries.

## Notes
- No threshold tuning; tau and all merge/region-growing parameters unchanged from the released run.
- The rebuild's baseline absolute mIoU may differ from the released 64.01 because the available
  language features are LangSplat CLIP ViT-B/16 (the paper likely used openseg ViT-L/14-336);
  this is a known, documented limitation of the reproduction, not of this A/B.
- Do not touch `ref/relags_hf/**`; all outputs go to `work_rofa/<scene>/<mode>/`.