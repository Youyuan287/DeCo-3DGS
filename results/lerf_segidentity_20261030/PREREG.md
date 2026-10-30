# PREREG — LERF-OVS identity pilot Phase B (segment/mask-level evidence), frozen before computing

Motivation (known): Phase A region-level evidence is real (centered agreement size-independent,
fused hard AUC 0.902) but cannot drive switching: v2/v3 gates failed, no positive operating point
(false-switch cost ≈ −0.85 IoU). Root cause hypothesis: region-level observation features are
mixtures over the whole root region; 52% of queries fell back to region level, and big wrong regions
look "consistent". Phase B computes identity evidence at the leaf/segment level using per-view
2D-mask assignments of the candidate's own Gaussians.

## Segment-level evidence (frozen)
For a candidate with Gaussian set G and root region r:
- views = views in which region r is observed (observations_level3 region_ids == r); view weight
  w_v = observation weights (visibility fraction).
- For each such view v, per-Gaussian 2D mask labels lab_v (uint8; 0 = background) come from
  merge_proj.SAI3D.extract_gaussian_features at level 3 (same mechanism as the frozen export,
  sim threshold 0.85).
- ratio_{v,L} = |{g in G : lab_v[g]=L}| / |G| for L>=1; keep masks with ratio >= 0.3 (frozen).
- Per-view segment feature f_v = L2norm( sum_L ratio_{v,L} * vf_v[L-1] ), vf_v = view.semantic["sem"]
  rows smin..smax (per-2D-mask CLIP features).
- masked_frac_v = |{g in G : lab_v[g]>=1}| / |G|; seg_conf_v = max_L ratio_{v,L}.

## Identity features (same names as Phase A)
- id_mean = cos(L2norm(sum_v w_v f_v), t); id_top3 = mean of top-3 cos(f_v,t)
- agree = mean pairwise cosine among f_v (n>600 deterministic subsample); agree_c on (f_v − raw_mean)
- id_mean_c = (sum_v w_v (f_v − raw_mean)) · t   (production-style, no renormalization)
- cover_frac = weighted mean of masked_frac_v; cover_views = #usable views; mwp_conf = weighted mean
  of seg_conf_v
- Fallback flag: <3 usable views → features NaN, fallback_seg=1 (reported; excluded from primary).

## Fusion and decision rules (frozen)
- Scale stream A = z(area_ratio)+z(mean_vis) (from evidence CSV, as before).
- Segment identity stream Vc_seg = z(id_mean_c)+z(agree_c) (within query).
- Primary fused F_seg = z(A)+z(Vc_seg); primary rule R_Fseg = argmax F_seg (ties → top-1).
- Secondary (reported): consensus R_cons_seg, min-fusion, margin sweep (diagnostic only).

## Gate (strict; PASS requires ALL)
1. hard (55): rescue(R_Fseg) >= 0.50 AND CI lower > 0.30
2. found (124): retention >= 0.85 AND CI lower > 0.75
3. cross_root_hard (46): rescue >= 0.45 AND CI lower > 0.25
4. hard: AUC(Vc_seg) residualized on log n_points, CI lower > 0.60
Secondary: deep (19) rescue >= 0.55 (report only).
Statistics unchanged: within-query pairwise AUC / rescue / retention; scene-stratified bootstrap
B=2000, seed 20261030.

## End-to-end (only if PASS)
Union gap prefix with the F_seg challenger when switching; render once per query; no parameters.
Ship bar: scene-mean delta > +0.01, CI lower > 0, no scene worse than −0.01 (vs main 0.6396 / R1 0.6525).

## Stop rule (agreed)
If Phase B fails the gate, the evidence-layer search ends: pivot to analysis-oriented positioning
(diagnosis + evidence findings + truncation rule) or candidate-generation work. No further
un-preregistered decision-rule variants on these four scenes.

## Caveats
- Same 4 scenes; any PASS is provisional.
- Per-Gaussian mask labels inherit the frozen export's 0.85 similarity threshold and level-3 semantics;
  differences versus VALA (α·T significance) remain as documented in Phase A.