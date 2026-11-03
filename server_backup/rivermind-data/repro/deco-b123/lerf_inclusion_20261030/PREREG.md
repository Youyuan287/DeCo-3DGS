# PREREG — Learned inclusion v1: candidate-membership classifier (frozen before computing)

Context: the remaining headroom is set-level. Oracle references on the released top-5 pool:
main 0.6396, prefix_best 0.704, single_best 0.708, greedy5 0.751. Module C (prune + scale-include)
reaches 0.6590. Previous set-level attempts used threshold rules on a regressed IoU and failed.
This experiment trains the selector on the set-level target directly.

## Training signal (LOSO; GT used only for labels on training scenes)
For each training query, compute the GT-greedy subset of its top-5 pool:
- render each candidate's mask; start from the empty union; iteratively add the candidate whose
  union with the current set maximizes IoU against GT; stop when no improvement or 5 additions.
- Label: 1 if the candidate is in the chosen subset, else 0.
Classifier: GBM binary classification (max_depth=3, max_iter=300, lr=0.05, min_samples_leaf=30,
class_weight balanced) on GT-free candidate features: score, deficit, rank, is_leaf, n_points,
area_ratio, mean_vis, cover_views, id_mean, id_top3, agree, agree_top, id_mean_c, agree_c, mwp_conf.
Per fold, train on the OTHER three scenes only.

## Frozen test rules (no tuning)
- R_inc (primary): S = released gap prefix (S0) ∪ {c : P(c) >= 0.5 and c not in S0}.
- R_inc_prune (secondary): R_inc followed by module C's guarded pruning (alpha=0.5, max_pred>=0.5
  guard using the IoU regressor predictions), then the scale-gated inclusion (R1 constants).
- Top-1 is always kept. Render exact selections; GT only for IoU.

## Gate (strict), primary R_inc
1. scene-mean mIoU delta vs released main >= +0.02, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
4. strictly better than module C (0.6590) by >= +0.005
If FAIL: stop the set-selection line on these scenes; report and keep C.

## Reporting
R_inc and R_inc_prune vs main and C; full per-scene, per-family (found/rank2/deep/recall_fail),
greedy-membership statistics (how often the greedy subset is a singleton, mean subset size),
selection-size stats, wins/losses. Same 4 scenes; provisional if PASS.