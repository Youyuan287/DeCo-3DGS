# PREREG — ROFA-style view aggregation and its centered-consistency variant (frozen before computing)

ReLaGS reference: within a unit, per-view features are aggregated with visibility weights after
removing outlier views by cross-view consistency (ROFA: keep q_i >= mu - tau*sigma, tau=2), and the
text is used only at scoring time. This experiment applies the same mechanism to the released
DeCo candidate observations and tests two variants end-to-end.

## Frozen variants (per candidate in the released top-5 pool)
Covering observations: root candidates use their region observations; leaf candidates use the
support-coverage >= 0.4 subset (deduped by view, fallback to all region observations if <3).
Weights a_o = visibility weight * coverage (normalized).
- AGG_raw  : weighted mean of raw observation features, score = cos(unit(mean), t)
- AGG_rofa : ROFA on raw features (q_i = mean pairwise cosine to other views; keep q_i >= mu-2sigma;
             if empty keep the argmax q_i), weighted mean of kept raw features, same scoring.
- AGG_rofa_center: the improved variant. Consistency is computed on per-view centered features
             (each view's features minus that view's mean over observations, unit); the same ROFA
             rule selects kept views; the kept views' RAW features are aggregated; same scoring.
End-to-end selection for each variant: reorder the released top-5 pool by that score and run the
released gap rule on the reordered scores; render the exact selection. main@5 rendered for parity.

## Stage 1 diagnostics (GT only to evaluate)
Per variant: single-choice scene-mean, found retention, deep rescue, rank2 rescue, recall rescue;
ROFA statistics: mean fraction of views removed, share of candidates whose query view is removed;
also agree/raw score correlation.

## Gate (strict), primary AGG_rofa_center
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
Secondary: AGG_raw and AGG_rofa reported; if any beats the primary, it is NOT adopted without a
new pre-registration.

## Notes
- Known risk: the observation-based signal historically helps deep but hurts rank2 (F1). The
  end-to-end gate protects against adopting a signal that only wins at ranking level.
- Same 4 scenes; provisional if PASS.