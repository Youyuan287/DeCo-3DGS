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
waldo_kitchen          invisible      21579                     7471              22      0.012993        0.009455            0.061877              0.055305           0.380730        0.000000               0.0        0.000000              0.000000
waldo_kitchen  target_low_purity        219                       73              22      0.006916        0.006934            0.077727              0.062357           0.450679    77483.739726            9210.0        0.054971            411.123288
waldo_kitchen target_purity_gt50        114                       38              20     -0.087782       -0.080006            0.014872              0.011976           0.600581    44571.157895            6030.5        0.881126          38427.078947
waldo_kitchen visible_background       3432                     1152              22      0.028567        0.020887            0.082788              0.071032           0.365226     7131.074301            1141.0        0.000000              0.000000
```

## Interpretation guardrails

The invisible majority is not evidence of background behavior and is reported separately. Visible background has higher positive mode-level gain than target-related rows in both scenes, while target rows often have higher Single Evidence scores and larger projected regions; this can change both the additive gain and the number of mode records. The result does not prove that all modes are noise: mode differences and target-related high-purity cases remain present. Query/region duplication is controlled by the collapsed `image/query/region` columns.

## Prior correction

`MODE_MEMBERSHIP_SUMMARY.csv` and all 40 candidate HTML pages now use the real B1/B2 NPZ `priors` arrays. The old membership JSON only stores `observation_indices`, so its missing `prior` default caused the previous all-zero column.
