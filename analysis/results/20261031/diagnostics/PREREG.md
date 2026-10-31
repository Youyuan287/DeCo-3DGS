# PREREG — LERF-OVS identity-evidence pilot (Phase A)

Frozen before any identity feature is computed. Do not edit after results are seen.

## Scope
- Dataset: LERF-OVS 4 scenes, 208 queries.
- Candidate pool: released top-5 pool (roots level-3 expanded to leaves level-2), same as lerf_evidence_dump.py.
- GT (rendered IoU per candidate) used ONLY for evaluation, never in features.
- Training-free: no learned weights; the only pre-registered combination is V = z(id_mean) + z(agree) (within-query z-scores, equal weights).

## Observation selection (frozen)
- Root candidate: obs = all observations with region_ids == root_id.
- Leaf candidate: obs from the same root region with support-coverage
  c_o = sum(support_weights over G ∩ support(o)) / sum(support_weights(o)),
  keep c_o >= 0.4 (primary). Sensitivity: 0.3 and 0.5 recorded per row (n_cov30/40/50, cover_frac30/40/50).
- One observation per view: dedupe by view id keeping the max c_o.
- Fallback (pre-registered): if n_cov < 3, use all root-region observations and set fallback_root=1.

## Features (frozen definitions)
- id_mean = cos(L2norm(Σ_o a_o·F_o), t), a_o ∝ weights[o]·c_o (normalized)
- id_top3 = mean of top-3 per-observation cosines cos(F_o, t)
- agree = mean pairwise cosine among selected observation features (n>600 → deterministic subsample to 600)
- agree_top = mean pairwise cosine among the top-3 observations by similarity
- cover_frac40 = fraction of candidate Gaussians appearing in any selected observation's support
- cover_views = number of distinct views among selected observations
- mwp_conf = mean over candidate Gaussians of max support beta over selected observations (ReLaGS MWP analogue)
- Centered variants (production style): id_mean_c = (Σ a_o·(F_o − raw_mean))·t (no renormalization),
  agree_c = mean pairwise cosine among (F_o − raw_mean). raw_mean = snag.feat[-1].mean(0).
- Query-level combination: V = z(id_mean) + z(agree); Vc = z(id_mean_c) + z(agree_c).

## Families (frozen; from candidate IoU)
- recall_fail: best candidate IoU < 0.3
- found: best candidate at rank 1
- rank2: best at rank 2
- deep: best at rank >= 3
- hard = rank2 ∪ deep; cross_root_hard = hard with best.root_id != rank1.root_id

## Statistics (frozen)
- within-query pairwise AUC (ties 0.5); rescue rate = argmax(feature) == best candidate (ties broken by CLIP score);
- bootstrap: 2000 resamples, seed 20261030, stratified by scene (resample queries within each scene, concatenate);
- size/context control: residualize the feature on [1, log1p(n_points)] (and separately log1p(cover_views)) within query, recompute AUC;
- report zero-variance share, tie rate, NaN share, fallback_root share, same-root share.

## Gate (strict; pre-registered)
PASS requires ALL of:
1. hard (n=55): AUC(V) >= 0.75 AND bootstrap 95% CI lower > 0.60
2. hard: rescue(V) >= 0.45 AND CI lower > 0.30
3. cross_root_hard (n=46): AUC CI lower > 0.60
4. residualized on log n_points: AUC CI lower > 0.55
5. found (n=124): AUC(V) >= 0.60 (harm check)
6. deep (n=19) secondary: AUC >= 0.70
If any fails → negative result, stop, no end-to-end reranker.

## End-to-end (only after PASS)
- final = CLIP score + λ·z(V); λ ∈ {0.25, 0.5, 1.0} chosen by leave-one-scene-out only.
- Rerank candidate order by final, re-run the released gap rule, render, report scene-mean mIoU vs main (0.6396) and R1 rule (0.6525).
- Ship bar: ΔmIoU > +0.01 with paired stratified bootstrap CI lower > 0 and no scene worse than −0.01.