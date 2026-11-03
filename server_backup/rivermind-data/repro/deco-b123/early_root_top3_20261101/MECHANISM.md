# Early-root top-3: existing-data mechanism analysis

Frozen continuation verdict: **FAIL**. B-A primary IoU -0.092354, existing paired 95% CI [-0.152159, -0.040303].

No new inference, feature extraction or post-hoc gate was run. This report does not revise the frozen decision rule.

Evaluations: 208 query-views, 67 distinct scene/prompt retrieval decisions, 22 evaluated images, four scenes. Candidate sets and selections are identical across evaluated views of each scene/prompt. The 208 rows are not 208 independent retrieval decisions; the 67 units also share scenes.

## Candidate preservation

At IoU tolerance 1e-08, at least one A-best candidate is retained in B on **173 / 208** query-views; all tied A-best candidates are lost on **35 / 208** (25 with positive A-best IoU). All tied A-best candidates are retained on 163 views. New B-only nodes strictly exceed A-best on **17 / 208** views.

The 10 all-lost views whose A pool has zero IoU count candidate-ID loss, not loss of positive GT coverage. Of the 17 new-better views, 16 start with zero A-pool IoU and 1 starts with positive A-pool IoU.

At the scene/prompt level, A-best is lost in at least one evaluated view for 11 / 67 units (9 in every evaluated view). Positive A-best is lost in at least one view for 8 units. New-better B nodes occur for 5 / 67 units.

Counts below refer to evaluated query-views, not independent prompts. A-best means every A candidate within 1e-8 of the maximum; retaining any tie preserves that value.

| Group | Views | Scene/prompts | Any A-best retained | All A-best lost | New B node better | B pool gain/loss/tie | Final gain/loss/tie |
|---|---:|---:|---:|---:|---:|---|---|
| Overall | 208 | 67 | 173 | 35 | 17 | 17/25/166 | 35/46/127 |
| figurines | 56 | 21 | 51 | 5 | 2 | 2/3/51 | 14/12/30 |
| ramen | 71 | 14 | 45 | 26 | 12 | 12/18/41 | 18/24/29 |
| teatime | 59 | 14 | 59 | 0 | 2 | 2/0/57 | 1/0/58 |
| waldo_kitchen | 22 | 18 | 18 | 4 | 1 | 1/4/17 | 2/10/10 |
| deep | 19 | 7 | 15 | 4 | 0 | 0/4/15 | 9/0/10 |
| found | 124 | 43 | 116 | 8 | 0 | 0/8/116 | 6/27/91 |
| rank2 | 36 | 13 | 23 | 13 | 0 | 0/13/23 | 5/18/13 |
| recall | 29 | 11 | 19 | 10 | 17 | 17/0/12 | 15/1/13 |

## Matched A∪B single-mask capacity

For each same query-view, U=max(A_pool_oracle,B_pool_oracle). This is the GT-best **single existing candidate** in A∪B. It is not a multi-mask union oracle, a new method, or deployable accuracy. The actual gap-selected prediction can contain multiple candidates and exceed this single-mask diagnostic.

All numbers in the next table use query mean within image → image mean within scene → mean across represented scenes. A family subset uses its represented images/scenes; its conditional mean is not an additive contribution to the global metric.

| Group | A final | B final | A pool | B pool | U pool | U-A pool | U-B pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 0.639262 | 0.546909 | 0.707397 | 0.661103 | 0.766503 | 0.059106 | 0.105400 |
| figurines | 0.585869 | 0.526451 | 0.734317 | 0.699030 | 0.748287 | 0.013971 | 0.049257 |
| ramen | 0.518076 | 0.407993 | 0.627519 | 0.580864 | 0.763650 | 0.136132 | 0.182786 |
| teatime | 0.793896 | 0.799475 | 0.801153 | 0.822173 | 0.822173 | 0.021020 | 0.000000 |
| waldo_kitchen | 0.659209 | 0.453716 | 0.666598 | 0.542344 | 0.731901 | 0.065303 | 0.189557 |
| deep | 0.458850 | 0.680060 | 0.851230 | 0.761588 | 0.851230 | 0.000000 | 0.089642 |
| found | 0.818404 | 0.670892 | 0.838293 | 0.767294 | 0.838293 | 0.000000 | 0.070999 |
| rank2 | 0.689876 | 0.407345 | 0.790744 | 0.475712 | 0.790744 | 0.000000 | 0.315032 |
| recall | 0.028051 | 0.113549 | 0.028297 | 0.325667 | 0.325667 | 0.297370 | 0.000000 |

## Recall/deep gains versus found/rank2 losses

Global contributions retain each row's original four-scene/image weight; the two group contributions sum to the full primary change. Conditional query-view means describe each group and use different weights.

| Group | Views | Final Δ, query-view mean | Pool Δ, query-view mean | Final Δ, global contribution | Pool Δ, global contribution | New-better views | A-best-lost views |
|---|---:|---:|---:|---:|---:|---:|---:|
| recall_deep | 48 | +0.169458 | +0.197895 | +0.045339 | +0.048766 | 17 | 14 |
| found_rank2 | 160 | -0.141669 | -0.098688 | -0.137693 | -0.095060 | 0 | 21 |

| Family | Final Δ, query-view mean | Pool Δ, query-view mean | U pool, query-view mean |
|---|---:|---:|---:|
| deep | +0.315384 | -0.148676 | 0.799797 |
| found | -0.107422 | -0.049339 | 0.840147 |
| rank2 | -0.259629 | -0.268667 | 0.751242 |
| recall | +0.073852 | +0.424960 | 0.451930 |

All 17 strictly better new-node views belong to recall; deep introduces no better single-mask candidate and its pool oracle decreases despite improved final predictions. The final-IoU change therefore cannot be explained solely by added candidate capacity. Found/rank2 lose 21 positive A-best candidates and introduce no better candidate. Across all families, final IoU still decreases on 25 views where an A-best candidate remains available. Candidate preservation alone is insufficient to explain the final selection outcome.

## Unit and interpretation limits

- Only previously collected CSV and candidate_ious were analyzed; no new inference.
- Matched-pool oracle=max(A_pool_oracle,B_pool_oracle): one candidate mask, not a union of masks.
- Oracle uses GT for diagnosis only; no post-hoc routing/gating or method score is proposed.
- Final gap selections may contain multiple masks; a single-mask oracle is not a bound on final IoU.
- 208 view-level evaluations reuse 67 scene/prompt retrieval decisions; they are not 208 independent decisions.
- 67 scene/prompt units share scenes and are not asserted to be independent new-scene samples.
- Family labels are inherited per query-view; one scene/prompt can appear in multiple families.
- Arm A is original raw-gap main; gateE .6760107 is a separate historical anchor.
- Frozen policy failure does not prove missing information or failure of all mask selectors.

The JSON retains per-query-view candidate IDs, tied-best retention, scene/prompt view counts, query-view/scene-prompt means and additive family contributions for checking these descriptions. Any new selection policy would require a separately preregistered experiment.
