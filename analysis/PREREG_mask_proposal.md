# PREREG — Text-selected 2D mask lifted as a 3D proposal (mask expansion)

Motivation: recall-family queries (29/208) have no correct candidate in the released pool
(best-candidate IoU 0.027). Mask-level reranking cannot fix them. Instead, use the 2D side to
propose a new candidate: in the query view, score every level-3 SAM mask by text similarity and
lift the top mask to 3D.

## Stage 1 (frozen): proposal alone
For each query, in the query view only:
- labels = level-3 SAM masks; feature = dataset CLIP feature of the mask (absolute label indexing);
- score(L) = cos(feature(L), text); order descending; take top-1 (and top-3 sensitivity);
- lift: select Gaussians with render_point weight > 1e-4 whose projected center lies inside the
  (union of) chosen mask(s); render the selection; IoU vs GT.
No NAG pool, no hierarchy, no GT in selection. Report per-scene and per-family (found/rank2/deep/recall).

Stage-1 diagnostic gate to proceed: recall-family mean IoU >= 0.10 (vs 0.027 in the pool) OR
deep-family mean IoU >= 0.60. No requirement to beat the full method alone.

## Stage 2 (frozen combination): union with gateB
If Stage 1 passes, for each query take the gateB selection (current best) and union it with the
proposal top-1, but only when the proposal's top-1 text margin (sim_top1 - sim_top2) > 0.
Render the union and evaluate.

Gates vs gateB: scene-mean delta >= +0.01; paired scene-stratified bootstrap (B=2000, seed=20261030)
CI lower > 0; no scene worse than -0.01; found family not worse by >0.005. If it fails, report
negative; do not tune the margin threshold.

## Notes
- The proposal uses the same query view as evaluation (2D evidence is part of the pipeline input),
  but never uses GT for selection.
- Absolute numbers compare to B0 (C1 on, 0.6393 main) and gateB (0.6757).
- Top-3 is reported as sensitivity only.