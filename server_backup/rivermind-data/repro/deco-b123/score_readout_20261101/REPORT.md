# Same-candidate score decomposition

Status: **COMPLETE**; rows 208/208; frozen C-A verdict **FAIL**.

A=original sigmoid/gap; B=margin with fixed A count (positive control); C=raw dot product with fixed A count (main). D=margin/own gap and E=raw/own gap are diagnostics and cannot be selected post hoc as the main result. No new inference is performed by this summary; gateE .6760107 remains an independent historical anchor.

Main C-A paired 95% CI: [-0.060288, +0.008362]; 14 wins / 9 losses / 185 ties.

| Weighting | A | B control | C main | C-A |
|---|---:|---:|---:|---:|
| Image → scene macro (primary) | 0.639262 | 0.639262 | 0.615477 | -0.023786 |
| Query-view mean | 0.625692 | 0.625692 | 0.626981 | +0.001290 |
| Mean of scene query-view means | 0.639623 | 0.639623 | 0.616049 | -0.023574 |

| Diagnostic arm | Primary metric | Query-view mean | Scene-query mean | Primary delta to A | Paired 95% CI |
|---|---:|---:|---:|---:|---|
| D (diagnostic only) | 0.639262 | 0.625692 | 0.639623 | +0.000000 | [+0.000000, +0.000000] |
| E (diagnostic only) | 0.594130 | 0.610704 | 0.593777 | -0.045133 | [-0.064966, -0.025004] |

| Scene | Views | Scene/prompts | A | C | C-A | C set changed |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 56 | 21 | 0.585869 | 0.616977 | +0.031108 | 7 |
| ramen | 71 | 14 | 0.518076 | 0.550233 | +0.032157 | 24 |
| teatime | 59 | 14 | 0.793896 | 0.793896 | +0.000000 | 2 |
| waldo_kitchen | 22 | 18 | 0.659209 | 0.500801 | -0.158408 | 7 |

Family means below use within-family query-view means, not the global primary weighting.

| Family | Views | Scene/prompts | A mean | C mean | C-A |
|---|---:|---:|---:|---:|---:|
| deep | 19 | 7 | 0.226267 | 0.525491 | +0.299224 |
| found | 124 | 43 | 0.816010 | 0.788954 | -0.027056 |
| rank2 | 36 | 13 | 0.663525 | 0.606349 | -0.057176 |
| recall | 29 | 11 | 0.026642 | 0.026516 | -0.000127 |

## Same-query candidate pairs

Pairs compare IoU≥.5 positive masks with IoU≤.1 negatives. Raw-correct means r_pos>r_neg; margin-correct means m_pos>m_neg. Canon-hurt is raw-positive/margin-negative; canon-help is the reverse. Exact score ties are reported separately and are not counted correct. These are repeated pair occurrences, not independent observations.

| Group | Pair occurrences | Views with pairs | Scene/prompts with pairs | Raw correct | Margin correct | Canon hurts | Canon helps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 695 | 170 | 56 | 614 | 614 | 42 | 42 |
| figurines | 225 | 48 | 19 | 216 | 197 | 24 | 5 |
| ramen | 213 | 53 | 11 | 163 | 166 | 13 | 16 |
| teatime | 180 | 53 | 12 | 175 | 180 | 0 | 5 |
| waldo_kitchen | 77 | 16 | 14 | 60 | 71 | 5 | 16 |
| deep | 81 | 18 | 7 | 56 | 27 | 37 | 8 |
| found | 476 | 119 | 40 | 453 | 476 | 0 | 23 |
| rank2 | 138 | 33 | 13 | 105 | 111 | 5 | 11 |
| recall | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Distinct oriented scene/prompt/candidate pairs: 264. Raw ties: 0; margin ties: 0. Distinct pairs still share queries/scenes and are not an independent statistical N.

| Positive-mask rank diagnostic | Eligible views | Scene/prompts | Raw improved | Raw worsened | Tied | Mean margin rank | Mean raw rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_positive | 170 | 56 | 16 | 22 | 132 | 1.547059 | 1.547059 |
| best_positive | 170 | 56 | 16 | 22 | 132 | 1.547059 | 1.547059 |

First-positive rank is the highest-ranked mask with IoU≥.5. Best-positive rank uses the highest GT IoU among positive masks, with IoU ties within 1e-8 credited at their earliest rank. Both compare raw r against margin m; candidate order resolves exact score ties as in the worker.

## Integrity and scope

- baseline_error_max: 0.0
- baseline_error_exact_zero: True
- baseline_within_preregistered_1e_6: True
- AB_iou_exact_equal: True
- AB_iou_max_absolute_difference: 0.0
- AB_sets_exact_equal: True
- ABC_fixed_count_equal: True
- inference_constant_within_scene_prompt: True
- complete_counts: True
- complete_detail_coverage: True
- 208 query-view evaluations reuse 67 scene/prompt decisions; family scene/prompt counts may overlap. They are not independent new-scene samples.
- Bootstrap: shared early-root implementation, 2000 paired image-cluster draws within scene, seed 20261101.
- score<.5 is not NAG rejection; r/b may be dot products of legacy preprocessed features rather than unit-vector cosines.
- D/E and ranking diagnostics do not override the frozen C-A endpoint; the same four scenes have been explored repeatedly.
