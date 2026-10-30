# PREREG — Module v2: prune-threshold and learner-inclusion grid (frozen before computing)

Current module C (PREEG_prune_include) passes its gate: LERF main 0.6396 -> 0.6590 (+0.0240,
CI [+0.0061,+0.0452]). Two frozen constants have never been selected under LOSO and one channel is
unused:
  (i) prune threshold alpha (currently 0.5 * max_pred);
  (ii) the learned regressor is used only for pruning, not for including unselected candidates
       (the standalone S3 inclusion rule was +0.0117 with a CI that included 0, but it was never
       combined with pruning).

## Frozen grid (9 configurations; no other values)
- alpha in {0.3, 0.4, 0.5, 0.6}: prune selected non-top-1 candidates with pred < alpha * max_pred
  (top-1 always kept; guard: pruning applied only if max_pred >= 0.5, as in C).
- beta in {0.8, 1.0, off}: after pruning, include any unselected candidate with
  pred >= beta * max_pred (learner-inclusion channel); off = no learner inclusion.
- The scale-gated challenger inclusion (R1 constants: area < 0.001 and max area > 2x top-1 area) is
  applied in ALL configurations.
- Configs: (alpha, beta) in {0.3,0.4,0.5,0.6} x {0.8,1.0} = 8, plus (0.5, off) = 9.
Predictions: the frozen LOSO IoU regressor (preds.csv, GBM, features from PREG_subset_selector).

## Protocol (LOSO)
For each held-out scene: choose the configuration with the best mean IoU over the other three scenes
(ties -> the configuration closest to current C: smaller beta, then alpha closer to 0.5); evaluate on
the held-out scene. Report the full grid, per-scene, per-family, and wins/losses.

## Gate (strict)
PASS requires ALL:
1. LOSO held-out scene-mean delta vs released main >= +0.03 with paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
4. LOSO held-out scene-mean strictly better than current C (0.6590) by >= +0.005
Secondary: per-config table vs C.

## Discipline
Grid and gate frozen now; no values added after seeing results. If FAIL, keep module C as the
reported method and stop tuning on these scenes. Same 4 scenes; provisional if PASS.