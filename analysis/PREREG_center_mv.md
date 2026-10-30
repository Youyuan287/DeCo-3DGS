# PREREG — Centered multi-view candidate scoring (frozen before computing)

Background (corrected, data-backed): per-view mean subtraction (centering) on mask
features leaves within-view text ranking exactly unchanged (verified: 208/208 ranks
identical) while making the same object's features consistent across views
(same-object cosine 0.63-0.76 -> after centering different-object cosine drops to ~0;
cross-view retrieval top-1 rises 25-42% -> 27-54%). This experiment applies the same
operation to the per-view observation features used by the 3D pipeline and tests
whether candidate scoring improves.

## Frozen signals (per candidate, released top-5 pool)
Covering observations per candidate as in the Phase-A identity dump
(root: region observations; leaf: support-coverage >= 0.4, deduped by view,
fallback to all region observations if <3, flagged).
Let F_o = unit observation feature, v(o) the view id, m_v = mean of F_o over all
observations of view v (then unit). Centered c_o = unit(F_o - m_v).
- S1_center = cos( unit( sum_o a_o c_o ), t ),  a_o = visibility weight * coverage
- S2_center = mean of top-3 cos(c_o, t)
- Raw counterparts S1_raw / S2_raw computed identically from F_o.

## Stage 1 (ranking, no rendering; GT only to label the best candidate)
Evaluate single-choice ranking (argmax score) on all 208 queries:
  scene-mean IoU, found retention, deep rescue, rank2 rescue, recall (reported).
Gate to proceed to Stage 2: S2_center satisfies ALL:
  1. scene-mean >= raw S2 + 0.02 (raw S2 measured in the same pass; reference ~0.5456)
  2. found retention >= 0.95
  3. deep rescue >= 0.60
  4. rank2 rescue >= 0.35
If Stage 1 fails: close this line (centering at candidate scoring does not improve ranking).

## Stage 2 (end-to-end, only if Stage 1 passes; frozen rule)
Reorder the released top-5 pool by S2_center and run the released gap rule
(selection length from the reordered score list); render exact selection.
Gate: scene-mean delta vs released main >= +0.01, paired scene-stratified bootstrap
(B=2000, seed 20261030) CI lower > 0; no scene worse than -0.01; found not worse by
>0.005. Also report module C (0.6590) for information.

## Notes
- Same 4 scenes; provisional if PASS. No parameter tuning; all constants inherited.
- The previous F1 id_top3 experiment (uncentered, end-to-end 0.6225) is the reference
  for what a cross-view signal without centering can and cannot do.