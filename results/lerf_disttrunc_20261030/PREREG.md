# PREREG — Distribution-informed truncation module (frozen before computing)

Motivation: the released gap rule ignores the shape of the score distribution and cuts at the
largest raw score drop; optimal truncation is worth +7.3 scene-mean (71.23 vs 63.96) and 28% of
queries cut at the wrong length (24 early / 34 late). Distribution-informed candidates are the
one class of module not yet tested for LERF.

## Frozen rules (zero learned parameters; only fixed probabilities)
Per query, take the released top-5 pool and its candidate scores s_i (sorted desc).
Compute within-query z = (s - mean)/std and p_i = softmax(z)_i (temperature 1; affine-invariant
rescaling of s does not change p).

- D1 (primary): credible prefix — smallest k with sum_{i<=k} p_i >= 0.8; select top-k.
- D2: same with p >= 0.9.
- D3: drop candidates with p_i < 0.05 (always keep the top-1 if all are dropped), then run the
  released gap rule on the filtered list.
- D4: same as D3 with threshold 0.10.
All selections rendered exactly; GT only for IoU.

## Gate (strict), primary D1
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
D2/D3/D4 are diagnostic reports only. If D1 fails: stop; no further truncation variants on these
scenes (report and move on).

## Notes
- Offline k-distribution check (before this prereg): mean k at p=0.8 is 2.44 vs gap 2.11;
  50% identical, 44% larger, 7% smaller. No IoU has been computed for D1-D4 yet.
- Same 4 scenes; provisional if PASS.