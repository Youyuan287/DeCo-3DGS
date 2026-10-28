# Evidence artifacts behind the audit numbers

Small JSON/CSV files copied from the audit host. They are the primary evidence
for the failure claim; the reports quote them.

| file | source | what it says |
|------|--------|--------------|
| `summary.json` | `observations/ramen/prototype_eval/summary.json` | the headline failure: `k1 0.0066732`, `b2_raw 0.0065405`, `b3_same_scene 0.0057221`, 71 queries / 7 images, status `PROTOTYPE_UNVERIFIED_SAME_SCENE_DIRECTION_FINAL_REGION_TOP5`. |
| `structural_check_level3.json` | `observations/ramen/structural_check_level3.json` | 11961 observations, 300 regions, 131 views, weight range 3.669e-05…1.0, `k1_max_abs_error=0.0` (self-consistency only), group counts 1–3. |
| `b3_same_scene_prototype_level3.json` | `observations/ramen/b3_same_scene_prototype_level3.json` | `raw_avg_groups 2.1533` vs `debiased_avg_groups 2.49`, 110 regions change group count, `u_norm ≈ 1`. |
| `ramen_summary.json` | `results/ramen_summary.json` | the certified baseline: `miou 0.5114937943613442`, 71 queries, protocol `topk=5 level_until=3 filter_small=50 remove_small=true`. |
| `hf_only/ramen_summary.json` | `results/hf_only/ramen_summary.json` | the flat-top-5 arm: 0.31463847662012323 with `level_until=-1`. |
| `official_base_metrics.csv` | `results/official_base_metrics.csv` | ramen: official field mIoU 40.40, rebuild 49.48, paper 51.2. |

Three different level-3 partitions: the certified baseline that produced
`ramen_summary.json` came from the official `sai_nag.pt` (249 level-3 regions),
the flat arm from the same file, and the prototype from a regenerated
`sai_nag_3.0.pt` with 304. See `../../MULTIEVIDENCE_BUG_REPORT.md` P1.