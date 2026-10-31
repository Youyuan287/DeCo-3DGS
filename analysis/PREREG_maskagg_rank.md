# PREREG — Rank fusion for mask-level aggregation (frozen before computing)

Motivation: the z-score fusion `S = z(NAG) + 1.0 z(id_top3_mask)` passed the gate
(ReLaGS 0.6401->0.6583; B0 0.6393->0.6619) but is lambda-sensitive (0.5:+0.37, 2.0:-2.00).
Rank fusion removes the scale dependency.

Frozen rule per query (released pool order and k unchanged):
- r1 = rank of NAG score (0 = best), r2 = rank of id_top3_mask (0 = best).
- S = -(r1 + r2) (Borda, equal weights).
- Select top-k by S, render union, evaluate.
No parameters; no tuning. Two bases: ReLaGS (--center none) and B0 (--center image),
evaluator `eval_lerf_ovs.py --text canon --renorm 0`.

Gates (same as PREREG_maskagg): Delta >= +0.01; paired scene-stratified bootstrap
(B=2000, seed=20261030) CI lower > 0; no scene worse than -0.01; found not worse by >0.005.
Secondary: compare with z-fusion lambda=1.0; report per-scene and per-family.

If rank fusion passes and is at least as good as z-fusion, adopt it as the robust default;
if it fails while z-fusion passes, keep z-fusion and report the instability.