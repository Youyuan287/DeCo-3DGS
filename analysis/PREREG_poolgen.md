# PREREG — Candidate-generation thresholds (frozen before computing)

Context: pool-depth alone failed (PREREG_pool_depth, gap@20 0.6292 vs main 0.6396). The remaining
unreachable family is recall: 29/208 queries (13.9%) have NO correct candidate in the released
top-5 pool (best candidate IoU ~0; oracle greedy20 only 0.196). Hypothesis: candidate generation
itself discards small objects through three frozen thresholds:
  (i)  root cut: keep roots with > 50 Gaussians   (nag_data/pool keep rule)
  (ii) leaf promotion: leaves >= 1% of the root's points (nag_data._analyze_leaf_candidates)
  (iii) pool depth: top-5 roots.

## Frozen grid (no other values may be tried on these scenes)
min_points ∈ {50, 20, 10}
leaf_ratio ∈ {0.01, 0.005}
n_roots    ∈ {5, 10}
=> 12 configurations. Selection rule is always the released gap rule (no other selector);
every selection rendered exactly; GT only for IoU.

## Protocol (LOSO)
- For each held-out scene s: pick the configuration with the best scene-mean IoU over the other
  three scenes (ties -> smaller deviation from released thresholds: n_roots=5, leaf_ratio=0.01,
  min_points=50).
- Evaluate the chosen configuration on s; aggregate held-out scene-mean. Also report the full grid
  (transparency), per-scene, per-family (recall_fail / found / rank2 / deep), and pool-size stats.
- Statistics: scene-stratified paired bootstrap (B=2000, seed 20261030) of the held-out delta vs
  released main@5.

## Gate (strict)
PASS requires ALL:
1. LOSO held-out scene-mean delta >= +0.01 with bootstrap CI lower > 0
2. no scene worse than -0.01
3. recall_fail family mean IoU strictly improves, and deep family is not worse by > 0.02
If FAIL: candidate-generation tuning on these scenes stops; move to the set-level selector or
analysis positioning (no further threshold fishing).

## Notes
- Threshold changes alter the candidate pool definition itself; the released main is no longer a
  strict subset of the new pool, so per-query wins/losses are reported alongside the mean.
- Same 4 scenes; any PASS is provisional and must be cross-validated on ScanNet later.