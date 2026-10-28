# The scripts the audit actually read

Everything here was pulled verbatim from `/root/rivermind-data/ReLaGS_Reproduction`
on 2026-10-28 and is byte-identical to what produced the numbers in `../../`.
Kept as plain files rather than a patch because these are small and a reader
should be able to open them directly from the repo.

| file | what the audit used it for |
|------|---------------------------|
| `reproduce_ramen_multievidence.sh` | the entry point; the 7-step chain in RAMEN_PIPELINE_AUDIT.md is traced through it. Step 3 is the only GPU step and passes the same flags as the certified launcher; there is no step that runs the certified evaluator. |
| `eval_ramen_evidence_prototype.py` | the script that produced 0.67 / 0.65 / 0.57. Bug P0-1 lives at lines 44, 46 and 49. |
| `analyze_ramen_observations.py` | produces the `k1_max_abs_error=0.0` that looks like an equivalence proof but is only a self-consistency check. |
| `prototype_b3_ramen.py` | defines the "shared direction" as `u = l2n(l2n(f).mean(0))`, i.e. the mean of all observations of this scene. |
| `vlm_utils.py` | `ClipSimMeasure.compute_similarity` — the canon-contrastive scoring the prototype replaced. **Differs from the repo's `utils/vlm_utils.py`**; this is the version that ran. |
| `eval_lerf_relags.py` | the certified evaluator (`topk`, `level_until`, `filter_small`, `remove_small`). **Differs from `reproduction/scripts/eval_lerf_relags.py`**; this is the version that ran. |
| `run_scene.sh`, `run_official_scene.sh` | the launcher comparison behind task item 6 (non-target changes). |

Not duplicated here because they are byte-identical to upstream
(`sha256` prefix `175c9e78d58bbaf4` on both sides): `nag_data.py`, i.e. the
hierarchical candidate search the audit analyses in
`search_matched_superpoint_in_mhtree` / `_analyze_leaf_candidates` /
`_filter_by_similarity_gap`.

## Version pinning

To reproduce the audit you need this exact set, not the repo's current copies:
two of the files above (`vlm_utils.py`, `eval_lerf_relags.py`) drifted from the
committed versions, and the region partition itself is not reproducible without
`--seed` (see `../code/README.md`).