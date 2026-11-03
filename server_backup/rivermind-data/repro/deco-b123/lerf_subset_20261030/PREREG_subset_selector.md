# PREREG — Learned subset selector (frozen before computing; learned probe, LOSO-disclosed)

Context: zero-parameter decision rules and candidate-generation thresholds both failed their gates.
A LOSO IoU regressor on released-pool candidates reaches found-retention 0.984 and hard rescue
0.418 (single-choice scene-mean 0.604, below multi-label main 0.6396). This experiment tests whether
the regressor helps at the SET level (union with the released gap prefix), where the oracle headroom
is large (greedy5 0.751 vs main 0.6396).

Narrative note: this component is LEARNED (GT IoU labels from training scenes). If it passes, adopting
it changes the paper's claim from training-free evidence to a learned selection layer; it is treated
as a method candidate only after the gate below passes, and always reported with LOSO disclosure.

## Frozen predictor
- LOSO over the 4 scenes: per fold, GBM regression (max_depth=3, max_iter=300, lr=0.05,
  min_samples_leaf=30) trained on the OTHER three scenes' released-pool candidates.
- Features (GT-free, identical to the diagnostic): score, deficit, is_leaf, n_points, area_ratio,
  cover_views, id_mean, id_top3, agree, agree_top, id_mean_c, agree_c, mwp_conf, mean_vis.
- Label: candidate rendered IoU.

## Frozen selection rules (applied per query to the released top-5 pool)
Let S0 = released gap prefix; max_pred = max predicted IoU over the pool.
- S1 (primary): S0 ∪ {c : pred(c) >= max_pred AND pred(c) >= 0.5}
- S2 (diagnostic): S0 ∪ {c : pred(c) >= max_pred AND pred(c) >= 0.4}
- S3 (diagnostic): S0 ∪ {c : pred(c) >= max_pred}
Union rendered exactly; GT only for IoU.

## Gate (strict)
PASS requires ALL for S1:
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found-family mean not worse than main by more than 0.005
If FAIL: stop; no further selector variants on these scenes.

## Reporting
Full S1/S2/S3 table, per-scene, per-family (found/rank2/deep/recall_fail), selection-size stats,
per-query wins/losses. Same 4 scenes; any PASS is provisional.