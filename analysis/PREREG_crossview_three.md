# PREREG — Three training-free cross-view attraction/repulsion schemes (frozen before computing)

Context. In the editable ReLaGS pipeline the released ROFA is neutral (ramen: off 0.4944,
raw 0.4962, centered-consistency 0.4850) and in the 64.01-anchored bank surrogate all ROFA
variants are bit-identical. The diagnosed bottleneck is not outlier views but that per-view
features are region/mask mixtures: averaging them mixes different instances and shared scene
semantics. The three schemes below attack that directly: same-instance views should attract,
different-instance views should repel. All are training-free, use no GT, no prompts, no labels.

Shared data: per-unit per-view observations (unit features f_i, weights a_i, view ids v_i,
visibility CSR). Evaluation only through rendered query mIoU on the 208 LERF queries.

## Scheme 1 — Mutual-nearest-neighbour consensus (MNN)
For a unit's observations:
  S_ij = cos(f_i, f_j), masked to v_i != v_j.
  nn(i) = argmax_j S_ij.
  keep i iff nn(nn(i)) == i (mutual nearest neighbour).
Aggregate the kept views' raw features with the released visibility weights.
Fallback: if |kept| < max(2, 0.1n) or |kept| == n, use released ROFA on the same set (flagged).
Rationale: same-instance views match each other mutually across views; views of another
instance or background are not mutual best matches. No labels, O(n^2), same cost as ROFA.
Diagnostics: kept fraction, query-view retained, mean pairwise cosine of kept vs raw.

## Scheme 2 — Dominant-mode aggregation
Per unit, affinity A = (1-gamma) * cos + gamma * (2*IoU_vis - 1), gamma=0.25 frozen.
Connected components at threshold theta=0.50 (frozen; no sweep). For each mode m:
  mass_m = sum of visibility weights, cons_m = mean pairwise cosine inside m,
  score_m = mass_m * max(cons_m, 0).
Keep the single highest-score mode and aggregate its raw features (visibility weighted).
If the best mode covers < 0.5 of the unit's total mass, keep the top 2 modes by score and
aggregate both (pre-registered, no per-query choice).
Rationale: different instances form different modes; the dominant consistent mode is the object.
Cost: connected components on <= ~150 nodes per unit.

## Scheme 3 — Candidate repulsion at readout (hard-negative margin)
No feature rebuild. Given the top-k candidate unit features F (k x D, unit) and text t:
  s_c = <F_c, t>
  m_c = normalize(mean_{j != c} F_j)          # competitor prototype
  s'_c = s_c - lambda * <F_c, m_c>,  lambda = 0.5 (frozen).
Selection remains the released gap rule on s'. Sensitivities lambda in {0.25, 1.0} reported,
never used to choose the result.
Rationale: candidates in the pool are usually different instances or background; subtracting
their shared direction suppresses the generic component and keeps the query-specific object.
Note: subtracting a single global scene prototype is rank-invariant and therefore not used.

## Evaluation and gates (identical for all three)
Stage 1 (diagnostic, no GT): kept/mode fraction in [0.20, 0.90], query view retained >= 0.95,
mean pairwise cosine of the retained set exceeds raw by >= 0.02. Fail -> report and stop that scheme.
Stage 2 (end-to-end, per scheme):
  base for 1/2 = rebuilt merge_proj readout with the same inputs, raw mode (same session);
  base for 3   = the 64.01-anchored readout (`--center none --text canon`, evidence off).
  Gates: scene-mean delta >= +0.01; paired scene-stratified bootstrap (B=2000, seed=20261030)
  95% CI lower > 0; no scene worse than -0.01; found family not worse by > 0.005.
  All 208 queries and all 4 scenes reported; no per-query or per-scene selection.
Multiple testing: three schemes are reported together; a scheme is adopted only if it passes
its gate; a scheme that fails is kept as a negative result.

## Implementation map
1. `merge_proj_rofa.py --rofa_mode mnn` (add branch in `_filter_feature_outliers`).
   Same branch in `evidence_modes_rcs.rofa_keep` for the bank surrogate.
2. `merge_proj_rofa.py --rofa_mode mode`; bank: `build_bank` dominant-mode aggregation.
3. `eval_lerf_ovs_repulse.py` = copy of `eval_lerf_ovs.py` + `--repulse-lambda` applied to the
   candidate score list before the gap selection (readout-only, no feature change).

## Cost
1/2: 4 scenes x (raw + scheme) merge_proj ~ 30-50 min each scheme; evaluation ~4 min.
3: evaluation only, 4 scenes x 2 configurations ~ 10 min.
Runs are independent; each writes its own CSVs and logs; no overwriting of `ref/relags_hf/**`.