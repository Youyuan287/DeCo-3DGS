# PREREG — Geometry-constrained 2D mask selection (frozen before computing)

Motivation: the GT-best 2D mask lifted to 3D reaches 0.8247 on deep and 0.2352 on recall, vs
0.6607/0.0198 for the 3D selection; the bottleneck is choosing the right 2D mask. Cross-view
feature consistency failed (it selects shared/background masks). Here we use **3D geometry**:
a mask should overlap the projected support of the object candidates.

## Frozen rule
Per query, in the query view:
- Candidate pool p5, NAG/mask scores, module C and gateE selection as in PREREG_c_mask_gate.
- Support set = Gaussians of (gateE selection) UNION (the candidate with the highest id_top3).
  This uses only the pool and mask evidence; no GT.
- Project visible support Gaussians (render_point weight > 1e-4, front-facing) to the query view;
  support_px = pixels covered by them.
- For every level-3 SAM mask L in the query view:
  text(L) = cos(mask feature, text);
  geo(L) = IoU(mask L, support_px);
  z-score both within the query's masks.
- Selection variants: text, geo, text+0.5*geo, text+1.0*geo (primary). Top-1 mask lifted to 3D
  (visible Gaussians inside the mask), rendered, IoU vs GT.

## Stage-1 diagnostic gate
Proceed if the primary (text+1.0*geo) satisfies BOTH: deep >= 0.66 (3D gateE level) and found >= 0.60;
and improves either deep or recall over text-only. Report all variants per family.

## Stage 2 (if passes)
Use the selected mask as a proposal in the evidence-competitive expansion: replace gateE only when
the proposal's top-3 mask similarity exceeds the selected candidates' id_top3. Gates vs gateE:
delta >= +0.01, bootstrap CI lower > 0, no scene worse than -0.01, found not worse by >0.005.

No tuned thresholds; lambda in {0.5, 1.0} pre-registered, 1.0 primary.