# M1 visibility-stratified recovery audit

This is an analysis-only correction from the existing FULL_EVIDENCE_TABLE; no CLIP, NAG, bank, or algorithm was rerun.

## Definitions

- `invisible`: `region_px == 0` in the current image.
- `visible_background`: `region_px > 0` and `intersection_px == 0`.
- `target_purity_gt50`: `intersection_px > 0` and `precision > 0.5`.
- `target_low_purity`: `intersection_px > 0` and `precision <= 0.5`.
- Mode-level M1 gain is `m1_slot_score - single_score`; the table also reports max-mode gain after collapsing duplicate modes for each image/query/region.

## Summary

```text
scene   visibility_class  mode_rows  image_query_region_rows  unique_queries  mean_m1_gain  median_m1_gain  mean_max_mode_gain  median_max_mode_gain  mean_single_score  mean_region_px  median_region_px  mean_precision  mean_intersection_px
ramen          invisible      43294                    14952              71      0.018108        0.015927            0.058189              0.050347           0.399050        0.000000               0.0        0.000000              0.000000
ramen  target_low_purity        645                      215              71     -0.002591        0.001176            0.051511              0.029934           0.461006    79805.693023           40381.0        0.041313           1944.297674
ramen target_purity_gt50        186                       62              38     -0.058595       -0.040789            0.026805              0.010917           0.556831    19196.048387            9307.0        0.878671          17766.338710
ramen visible_background       7137                     2379              71      0.020585        0.016759            0.067522              0.059370           0.405907    13694.900799            1803.0        0.000000              0.000000
```

## Interpretation guardrails

The invisible majority is not evidence of background behavior and is reported separately. Visible background has higher positive mode-level gain than target-related rows in both scenes, while target rows often have higher Single Evidence scores and larger projected regions; this can change both the additive gain and the number of mode records. The result does not prove that all modes are noise: mode differences and target-related high-purity cases remain present. Query/region duplication is controlled by the collapsed `image/query/region` columns.

## Prior correction

`MODE_MEMBERSHIP_SUMMARY.csv` and all 40 candidate HTML pages now use the real B1/B2 NPZ `priors` arrays. The old membership JSON only stores `observation_indices`, so its missing `prior` default caused the previous all-zero column.
