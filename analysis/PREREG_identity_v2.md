# PREREG — LERF-OVS identity pilot v2 (frozen before any v2 metric is computed)

Follow-up to the Phase A pilot (verdict FAIL). This v2 uses only pre-existing dumps:
identity CSV (lerf_identity_20261030), evidence CSV (lerf_evidence_20261030), candidate/attribution tables.

Motivation (established, not re-derived here): raw agreement is an inverted signal; the centered
identity evidence Vc = z(id_mean_c)+z(agree_c) reached hard-query AUC 0.807 (size-residualized 0.758)
but rescue 0.382, below the scale prior area_ratio 0.881. The pilot's failure is attributed to the
decision rule, not the signal.

## Frozen signals (all GT-free, zero parameters)
- Scale stream: A = z(area_ratio) + z(mean_vis)   (within-query z; mean_vis from evidence CSV).
- Identity stream: Vc = z(id_mean_c) + z(agree_c) (within-query z; from identity CSV).
- Fused evidence: F = z(A) + z(Vc)                (equal weight between the two pre-registered streams).

## Frozen decision rules (single-candidate scope, stage 1)
- R_F: pick argmax(F); ties resolved to the released top-1 (rows sorted by CLIP rank).
- R_F_gate (secondary): apply R_F only if area(top1) < 0.001 and area.max() > 2*area(top1); else keep top-1.
- R_Vc, R_A: argmax of each single stream (reported for decomposition only).

## Families (unchanged)
recall_fail (best IoU<0.3) / found (best at rank1) / rank2 / deep (rank>=3); hard = rank2∪deep;
cross_root_hard = hard with best.root_id != top1.root_id.

## Statistics (unchanged)
within-query pairwise AUC; rescue = argmax==best; scene-stratified bootstrap B=2000, seed=20261030;
residualization within query on log1p(n_points) (and log1p(cover_views) for reporting).

## v2 gate (strict; PASS requires ALL)
1. hard (n=55): rescue(R_F) >= 0.50 AND CI lower > 0.30
2. found (n=124): retention(R_F) >= 0.85 AND CI lower > 0.75   (retention = rule keeps the rank-1 candidate)
3. cross_root_hard (n=46): rescue(R_F) >= 0.45 AND CI lower > 0.25
4. hard: residualized-on-n_points AUC(F) CI lower > 0.60
Secondary (reported, not gating): deep (n=19) rescue(R_F) >= 0.55; R_F_gate metrics.

## End-to-end (only if gate PASSES)
- Multi-label inclusion rule: if argmax(F) != top-1, union the released gap prefix with the F-challenger
  (gap recomputed only over the original order); render exact selection.
- No tunable parameter. Report scene-mean mIoU vs main (0.6396) and R1 rule (0.6525); paired
  scene-stratified bootstrap; per-scene and per-query win/loss.
- Ship bar: delta mIoU > +0.01 with CI lower > 0 and no scene worse than −0.01.

## Discipline
- No threshold tuning after seeing v2 results. R_F_gate and single-stream rules are secondary reports only.
- All claims are on the same 4 scenes as the pilot; external validity still requires new scenes/Phase B.