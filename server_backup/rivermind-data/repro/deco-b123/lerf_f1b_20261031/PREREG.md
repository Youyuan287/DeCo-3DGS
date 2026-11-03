# PREREG — F1b: fused ranking with released truncation length (frozen before computing)

F1 (reorder by S + gap on S) failed overall (-0.0096) but decomposed cleanly: deep +0.167, found
-0.001, rank2 -0.142. Mechanism: moving the gap rule into the fused score space changed the
truncation length and wrecked rank-2 queries. F1b isolates ranking from truncation:
keep the released gap prefix LENGTH k (computed on the original NAG scores), but choose WHICH k
candidates by the fused score S = z(NAG score) + z(id_top3).

## Frozen rule R_F1b
- k = released gap prefix length on the original top-5 pool (as in main).
- Selection = the k candidates with the highest S (ties -> higher NAG score).
- Render exact selection; GT only for IoU.
Secondary (reported): same but k+1 and k+2 lengths (diagnostic only).

## Gate (strict), primary R_F1b
1. scene-mean delta vs released main >= +0.01, paired scene-stratified bootstrap (B=2000,
   seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found not worse by >0.005
Compare against main, F1 (0.6225) and module C (0.6590). Same 4 scenes; provisional if PASS.
If FAIL: feature-fusion line at top-5 granularity is closed; next lever is F2 (teacher swap).