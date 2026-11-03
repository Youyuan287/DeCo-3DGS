# M1_TARGET_SEMANTIC_VALIDATION

## Scope and stopping rule

This is an analysis-only correction. It reuses the existing FULL_EVIDENCE_TABLE, frozen observations, existing B1/B2 bank files, GT annotations, camera files, root-region masks, and the same Gaussian renderer used by the support render. It does not rerun CLIP, construct NAG, rebuild banks, change M1/M2/M3, or scan parameters.

The row-level M1 gain is `m1_slot_score - single_score`. Invisible regions are separated before any interpretation. A second aggregation collapses duplicate mode rows to one max-mode gain per `(scene,image,query,region_id)`.

## Four-class M1 recovery summary

The counts below are intentionally shown both as mode rows and as image/query/region records to expose mode and query duplication.

### ramen
```text
  visibility_class  mode_rows  image_query_region_rows  unique_queries  mean_m1_gain  mean_max_mode_gain  mean_single_score  mean_region_px  mean_precision
         invisible      43294                    14952              71      0.018108            0.058189           0.399050        0.000000        0.000000
 target_low_purity        645                      215              71     -0.002591            0.051511           0.461006    79805.693023        0.041313
target_purity_gt50        186                       62              38     -0.058595            0.026805           0.556831    19196.048387        0.878671
visible_background       7137                     2379              71      0.020585            0.067522           0.405907    13694.900799        0.000000
```

### waldo_kitchen
```text
  visibility_class  mode_rows  image_query_region_rows  unique_queries  mean_m1_gain  mean_max_mode_gain  mean_single_score  mean_region_px  mean_precision
         invisible      21579                     7471              22      0.012993            0.061877           0.380730        0.000000        0.000000
 target_low_purity        219                       73              22      0.006916            0.077727           0.450679    77483.739726        0.054971
target_purity_gt50        114                       38              20     -0.087782            0.014872           0.600581    44571.157895        0.881126
visible_background       3432                     1152              22      0.028567            0.082788           0.365226     7131.074301        0.000000
```

## Why visible background gain can exceed target gain

1. The invisible majority is not semantic background: it has `region_px == 0` and no current-image support. Including it would mix absence of evidence with visible background.
2. In both scenes, visible background has lower Single Evidence scores than target-related rows, leaving more additive headroom for a mode score. For ramen the visible-background mean Single Evidence is 0.4059 versus 0.4825 over all target-related mode rows; for waldo it is 0.3652 versus 0.5020. This makes a larger raw gain possible without proving stronger target semantics.
3. Projected area differs materially: visible-background mean region area is 13,694.9 px (ramen) and 7,131.1 px (waldo), while target-related mode rows average 66,239.6 px and 66,216.4 px respectively in the original evidence. The low-purity and high-purity target strata are also very different in area, so area is a confounder for raw additive gain.
4. Mode and query repetition is controlled rather than hidden: `mode_rows` can be compared with `image_query_region_rows`, and `unique_queries` is reported. The same region appearing in several query images is not treated as independent scene-level evidence in the collapsed column.
5. Rank-correlation diagnostic on visible rows (not a causal correction):
- `ramen` visible rows: Spearman(gain, Single Evidence) = `-0.2646`, Spearman(gain, region area) = `0.0031`.
- `waldo_kitchen` visible rows: Spearman(gain, Single Evidence) = `-0.3971`, Spearman(gain, region area) = `-0.0133`.

Therefore the result is a target-selectivity warning, not proof that all modes are noise. The evidence contains high-purity target-aligned modes and also high-gain background/mixed modes.

## Prior-field correction

The previous `MODE_MEMBERSHIP_SUMMARY.csv` used `x.get('prior', 0.0)` on membership JSON entries. Those entries contain observation indices but no prior field, so every exported prior became zero. The corrected file loads `priors` from the real B1 and B2 NPZ arrays using `(region_id, mode_id)`; the backward-compatible `prior` column is now the B1 prior, while `prior_m1` and `prior_m2` preserve both bank values. Production banks are untouched.
- Corrected ramen rows: `722`; corrected waldo rows: `1152`.
- In the corrected backward-compatible `prior` column: `1765` nonzero values, `50` legitimate exact zeros, and `59` unmapped rows where the corresponding root is absent from that bank; these are no longer silently converted to zero.
- Corrected HTML pages: `40`; each page records `prior_m1`, `mode_id`, and its NPZ source.
- The six selected figure manifests also carry the corrected `m1_prior` from FULL_EVIDENCE_TABLE.

## Six fixed cases

```text
        case_group         scene       image          query  region_id  mode_id  recover_gain  precision  region_px  intersection_px
target_purity_gt50         ramen frame_00128 glass of water          6        1      0.272883   0.532045       8020             4267
visible_background         ramen frame_00128 glass of water        145        1      0.370243   0.000000        405                0
 target_low_purity         ramen frame_00128 glass of water         35        1      0.315535   0.059567      31158             1856
target_purity_gt50 waldo_kitchen frame_00154           sink         31        1      0.138236   0.983355      32983            32434
visible_background waldo_kitchen frame_00053    yellow desk        101        2      0.377847   0.000000        251                0
 target_low_purity waldo_kitchen frame_00154        red cup         39        2      0.323959   0.092327       4040              373
```

The selection is from the existing FULL_EVIDENCE_TABLE/support evidence universe. The previous 40-page heuristic did not include a waldo high-purity target row among its five `target_recovery` pages; the waldo sink row above is nevertheless an existing support/evidence row, not a newly scored example. This limitation is recorded rather than hidden.

## Exact current-view mode-support projection

The following table is from the renderer output for the selected M1 mode, not from the coarse root mask. `projection_precision_at_0.5` uses the exact visible-Gaussian support map thresholded at 0.5; a zero maximum means that this mode's observation-union has no current-view rasterized support under the unmodified renderer.

```text
        scene       image          query  region_id  mode_id  root_precision  projection_max  projection_px_at_0.5  projection_GT_intersection_at_0.5  projection_precision_at_0.5
        ramen frame_00128 glass of water         35        1        0.059567          0.0000                     0                                  0                          NaN
        ramen frame_00128 glass of water          6        1        0.532045          0.9999                 11087                                  0                     0.000000
        ramen frame_00128 glass of water        145        1        0.000000          0.0000                     0                                  0                          NaN
waldo_kitchen frame_00154        red cup         39        2        0.092327          0.9999                 13581                              12948                     0.953391
waldo_kitchen frame_00154           sink         31        1        0.983355          0.0000                     0                                  0                          NaN
waldo_kitchen frame_00053    yellow desk        101        2        0.000000          0.0000                     0                                  0                          NaN
```

## Semantic case judgments

- **High-purity root cases:** ramen `(frame_00128, glass of water, region 6, mode 1)` has root precision 0.5320, but its selected mode projection has 11,087 pixels at 0.5 and zero GT intersection. Waldo `(frame_00154, sink, region 31, mode 1)` has root precision 0.9834, but the selected mode has zero current-view projection. Thus coarse-root purity alone does not establish that the high-score mode observes the target.
- **Visible-background cases:** ramen `(frame_00128, glass of water, region 145, mode 1)` and waldo `(frame_00053, yellow desk, region 101, mode 2)` have nonzero root projection but zero GT intersection; the selected mode projections are also empty in the current view. Their scores cannot be interpreted as target observation from these images.
- **Low-purity target cases:** ramen `(frame_00128, glass of water, region 35, mode 1)` has root precision 0.0596 and an empty selected-mode projection. Waldo `(frame_00154, red cup, region 39, mode 2)` has root precision 0.0923, but its mode support has 13,581 pixels, 12,948 GT-intersection pixels, and 0.9534 support precision. This is a concrete example showing that a low-purity coarse root can still contain a target-aligned mode; it is why “all modes are noise” is invalid.

## Answers to the four required questions

1. **Does a high-score mode actually observe the query target?** The six-case audit is mixed: direct mode-support/GT alignment is observed for waldo red cup (0.9534 support precision at 0.5), while ramen glass-of-water and waldo sink high-purity roots do not show selected-mode/GT overlap in the current view.
2. **Is a high-score mode mainly background or mixed support?** Several selected modes have no current-view support, and the ramen high-purity selected mode projects away from its GT. The waldo red-cup mode is target-aligned despite a low-purity root. These observations rule out both universal target selectivity and universal noise.
3. **Is the data sufficient to explain mode-score advantage as true complementary semantics?** No. The same evidence contains target-aligned, background-only, and mixed supports, and lacks manual semantic labels for all modes. The score advantage is therefore not semantically identified.
4. **Which cases support or contradict M1?** Waldo red-cup mode 2 supports a target-aligned mode; ramen mode 1 for glass-of-water, waldo sink mode 1, and both visible-background cases contradict a universal target-selectivity claim. The precise causal source remains unresolved.

## Figure and evidence paths

- `CASE_FIGURES/`: six RGB + GT + root-region + mode-visible-Gaussian support montages and JSON manifests.
- `SELECTED_MODE_PROJECTION_AUDIT.csv`: exact current-view support/GT overlap for the selected mode in each case.
- `CORRECTED_CASE_HTML/`: all 40 metadata pages with corrected priors; pages not among the six selected cases remain metadata-only.
- `M1_RECOVERY_BY_VISIBILITY.csv`: four-class numeric summary.
- `FULL_EVIDENCE_TABLE.csv`: unchanged source evidence; `FULL_EVIDENCE_TABLE_WITH_VISIBILITY.csv` adds only analysis labels.

Completed and stopped after this validation; no algorithm or parameter changes were made.
