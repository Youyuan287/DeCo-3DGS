# PREREG — Pool-depth method delta (frozen before computing)

Motivation: the decision-layer (candidate switching) is closed (identity v1/v2/v3/Phase B all failed
the pre-registered gates; asymmetric switch cost −0.85 vs +0.47). The remaining headroom from the
step-0 oracle is at the set/recall level: optimal truncation 71.23, greedy5 75.12, greedy20 77.91
(scene-mean mIoU; released main 63.96 / attribution recomputation 0.6396).

Hypothesis (single, frozen): applying the released gap rule to a deeper visual pool
(top-10 and top-20 roots, same leaf promotion, same gap rule) improves scene-mean IoU without any
parameter or learned component. This is a pure method-delta test.

## Protocol (frozen)
- Pools: pool(sims, k) for k in {5, 10, 20} exactly as in lerf_oracle.py (root cut k, leaves promoted,
  sorted by score). Selection: released gap_prefix over each pool; render the exact union; IoU vs GT.
- No GT in selection. No parameters. No training.
- Primary endpoint: scene-mean mIoU of gap@20 vs released main@5.
- Secondary: gap@10; per-family (found / rank2 / deep / recall_fail from the step-0 families);
  per-query win/loss; bootstrap CIs (scene-stratified, B=2000, seed 20261030).

## Gate (strict)
PASS requires:
1. gap@20 scene-mean − main@5 >= +0.01 with bootstrap CI lower > 0
2. no scene worse than −0.01
3. per-family report shows the gain is not hiding a deep regression (deep family not worse than main by >0.02)
If gap@20 passes, adopt it as the new base for a follow-up evidence-gated inclusion test.
If it fails, do not tune pool size further on these scenes (fishing); move to candidate-generation work.

## Notes
- Oracle references (not method results): prefix_best(top5) 0.704, greedy5 0.751, greedy20 0.774
  per-query mean; the released main is multi-label gap. gap@20 is NOT greedy; no assumption it
  approaches greedy20.
- Same 4 scenes; any PASS is provisional.