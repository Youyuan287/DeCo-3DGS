# PREREG — F1: purified multi-view feature signal fused into candidate scores (frozen before computing)

Context: the user's critique (features, not selection) is supported by our audits: representability is
not the bottleneck; deep-family correct candidates exist with IoU 0.8-0.96 but CLIP ranks them low.
A zero-cost probe on existing observations showed: single-choice ranking with the top-3 per-view
similarity (id_top3, a visibility-aware multi-view feature) rescues 73.7% of deep queries (NAG score: 0)
but loses some found queries. Score-level fusion S = z(NAG score) + z(id_top3) raises single-choice
scene-mean 0.5327 -> 0.5651 with found retention 0.992 in the probe (lambda=1).

## Frozen signal and rule
- id_top3 per candidate: mean of the top-3 per-view cosine similarities between the candidate's
  covering observations and the prompt (already computed in lerf_identity_20261030; observations are
  visibility-weighted per-view features; GT-free).
- Fused score, per query: S = z(NAG candidate score) + z(id_top3), z within the top-5 pool, lambda=1
  (equal weights; no tuning).
- Primary rule R_F1: reorder the released top-5 pool by S, then apply the released gap rule to the
  reordered list; render the exact selection.
- Secondary (reported, not gated): C module applied with pruning/scale-inclusion but ranking by S.

## Gate (strict), primary R_F1
1. scene-mean mIoU delta vs released main >= +0.01, paired scene-stratified bootstrap
   (B=2000, seed 20261030) CI lower > 0
2. no scene worse than -0.01
3. found family not worse than main by more than 0.005
Report: vs module C (0.6590) as information; per-family; selection sizes; wins/losses.
If FAIL: feature-fusion line at this granularity is closed; next would be teacher swap (F2).

## Notes
- The probe (above) was computed on the same 4 scenes from existing CSVs; the end-to-end numbers have
  not been computed yet. Lambda is frozen at 1; no sweep in the gated result.
- Same 4 scenes; provisional if PASS.