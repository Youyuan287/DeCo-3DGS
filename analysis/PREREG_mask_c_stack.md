# PREREG — Module C + mask-order stack (frozen before computing)

Motivation: module C (prune+include) and mask-level rerank change almost disjoint query sets
(B0: 32 vs 12 changed, 5 overlap; teatime all C, ramen mostly disjoint). Test whether applying
module C's selection rules to the MASK order stacks their gains.

Exact rule per query (released pool p5, NAG order; k = released gap length g):
- S0 = indices 0..g-1 (NAG order) = main.
- S0m = first g candidates by id_top3_mask (ties by NAG order) = mask order.
- pv = per-candidate NAG pred; mp = max(pv); area/mean_vis as in module C.
- prune(X) = [i in X if pv[i] >= 0.5*mp], fallback to [X[0]].
- C: S_C = prune(S0) if mp>=0.5 else S0; include area champion if R1 condition (unchanged).
- stack: S_stack = prune(S0m) if mp>=0.5 else S0m; include the same area champion.
- Render each selection and evaluate IoU.

Primary base: B0 (`--center image`), because module C was defined on C1-on. Secondary: ReLaGS
(`--center none`). Evaluator `eval_lerf_ovs.py --text canon --renorm 0`.

Gates (stack vs module C):
1. scene-mean delta >= +0.01;
2. paired scene-stratified bootstrap (B=2000, seed=20261030) 95% CI lower > 0;
3. no scene worse than -0.01;
4. found family not worse than C by >0.005.
Also report stack vs main and C vs main. If stack fails vs C but beats main by more than either
alone, report as a partial/orthogonality result, not adopted.

No tuning: g, thresholds, R1 condition, and mask signal (id_top3) are inherited from prior preregs.