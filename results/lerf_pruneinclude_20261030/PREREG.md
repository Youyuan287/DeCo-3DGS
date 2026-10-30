# PREREG — Prune-and-include module (frozen before computing)

Context: learned prefix pruning P1 (+0.98 scene-mean, CI spans 0, harms deep) and its guarded form
(max_pred >= 0.5: +0.71, CI [+0.0007,+0.0135], no scene harmed) fix cut-late queries (P1: +13.3 on
34 queries). The released R1 rule (scale-gated challenger inclusion, PREG rules_20261030) fixes
cut-early queries (+0.0129 scene-mean). The two act on different error families, so a combined
module may be additive.

## Frozen module C (all constants inherited; no new tuning)
Per query, released top-5 pool:
1. Prune: with the frozen LOSO IoU regressor predictions (preds.csv), drop selected non-top-1
   candidates with pred < 0.5 * max_pred (top-1 always kept).
2. Guard: apply step 1 only if max_pred >= 0.5; otherwise keep the released gap prefix.
3. Include (R1): if the scale gate fires (top-1 area < 0.001 and max area > 2x top-1 area), union
   the challenger with the highest scale evidence A = z(area)+z(mean_vis) (additive epsilon ties ->
   unchanged).
Render the final set exactly. GT only for IoU.

Diagnostic variant C0: same but without the max_pred >= 0.5 guard.

## Gate (strict), primary C
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
If FAIL: stop on these scenes and report the full picture.

## Reporting
C and C0 vs main; per-scene; per-family; cut_early/cut_late breakdown; include-rate and prune-rate;
wins/losses. Learned component disclosed (regressor trained LOSO on other scenes). Same 4 scenes;
provisional if PASS.