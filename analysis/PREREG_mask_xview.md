# PREREG — Cross-view mask selection (frozen before computing)

Motivation: the 2D upper bound (GT-best mask lifted) is much higher than the 3D selection for
deep (0.8247 vs 0.6607) and recall (0.2352 vs 0.0198), while 3D is better for found/rank2.
Text-only 2D mask selection is weak (deep 0.529, recall 0.108). We test whether cross-view
mask-feature support improves **which** query-view mask is selected.

## Frozen rule
For each query (query view = annotated frame), for every level-3 SAM mask L in the query view:
- text(L) = cos(feature(L), text);
- support(L) = mean over the other annotated frames of the max cosine between feature(L) and
  that frame's masks;
- z-score text and support within the query's masks;
- score(L) = z(text) + lam * z(support), lam in {0.5, 1.0} (1.0 primary); also support-only.
Select top-1; lift visible Gaussians inside it (weight>1e-4, front-facing) and render; IoU vs GT.
No GT in selection; support uses only other annotated frames (4-7 per scene).

## Stage-1 diagnostic gate
Proceed if xview (lam=1) improves either: deep mean IoU >= 0.66 (3D gateE level) or recall mean
IoU >= 0.10 (text-only proposal level), AND found mean IoU does not drop below 0.60.
Report text/xview05/xview10/support-only and per-family.

## Stage 2 (if passes)
Use xview top-1 as a 2D proposal in the same evidence-competitive expansion as before:
replace the gate selection only when the proposal's mask evidence (top-3 mask similarity) exceeds
the selected candidates' id_top3. Gates vs gateE: delta >= +0.01, bootstrap CI lower > 0, no scene
worse than -0.01, found not worse by >0.005.