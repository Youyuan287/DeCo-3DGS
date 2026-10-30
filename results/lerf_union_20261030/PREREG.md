# PREREG — Union candidate generation v1 (frozen before computing)

Context: level-1 proposals alone cannot be merged with the released level-2 pool by raw score
(different feature spaces). This experiment fixes one cross-level fusion and tests whether it helps.

## Frozen candidate set and fusion
- P2: released top-5 pool at level-2 (as in all prior experiments).
- P1: top-10 pool at level-1 (same root cut >50 points, leaf promotion at level-1).
- Per-level min-max normalization within the query: s' = (s - min) / (max - min); if max == min,
  s' = 0.5 for that level.
- Merged list = P2 ∪ P1 candidates sorted by s' descending.

## Frozen rules
- U_gap (primary): released gap rule applied to the merged list (largest s' drop).
- U_add (secondary): released main selection (P2 gap prefix) union all P1 candidates with
  s' >= 0.8.
Render exact selections; GT only for IoU.

## Gate (strict), primary U_gap
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
If FAIL: candidate-generation line closed on these scenes; keep module C and report.

## Reporting
U_gap and U_add vs main and C; per-scene; per-family; recall-fail family specifically; selection
sizes; wins/losses. Same 4 scenes; provisional if PASS.