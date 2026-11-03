# PREREG — Learned prefix pruning (frozen before computing; learned probe, LOSO-disclosed)

Context: cut-late errors (34/208) mean the released gap prefix includes false positives. Probability
filtering (D3) was near-neutral (19 wins / 10 losses, delta -0.0017). The LOSO IoU regressor has
found-retention 0.984 and can estimate which selected candidates are low-quality. Hypothesis:
pruning the released gap prefix with predicted-IoU thresholds improves scene-mean without harming
found queries.

Narrative note: learned component (GT IoU labels from training scenes); LOSO-disclosed. Treated as a
method candidate only if the gate passes.

## Frozen rules (released gap prefix S0, per query; max_pred over the pool)
- P1 (primary): S = {c in S0 : pred(c) >= 0.5 * max_pred} ∪ {top-1}; i.e., drop selected non-top-1
  candidates whose predicted IoU is below half of the pool maximum.
- P2 (diagnostic): threshold 0.3 * max_pred.
- P3 (diagnostic): absolute pred(c) >= 0.3 (top-1 always kept).
Predictions come from the frozen LOSO regressor of PREREG_subset_selector.md
(preds.csv; GBM, features frozen there).

## Gate (strict), primary P1
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
If FAIL: stop (no further pruning variants on these scenes).

## Reporting
P1/P2/P3 table, per-scene, per-family (found/rank2/deep/recall_fail), cut_early/cut_late effects,
selection sizes, wins/losses. Same 4 scenes; provisional if PASS.