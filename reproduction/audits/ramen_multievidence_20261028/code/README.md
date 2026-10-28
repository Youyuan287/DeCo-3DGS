# The projection module as it was actually run

The audit in `../` is about `merge_proj.py` — the module that turns per-view
2D CLIP/SAM features into per-superpoint 3D features. The working tree that
produced `observations_level3.npz` is **not** upstream DeCo: it carries 73
insertions and 2 deletions against `merge_proj.py` at `38620a7`. They are
reproduced here so every claim in the audit can be traced back to code.

| file | role | relation to upstream |
|------|------|----------------------|
| `merge_proj_observation_export.diff` | the exact delta | applies cleanly to `merge_proj.py` @ `38620a7` (`git apply --check` passes) |
| `merge_proj.relags_observation_export.py` | the full resulting file | upstream + the delta; byte-identical to the server copy after CRLF normalisation |
| `multi_evidence.py` | new module, 173 lines | not in upstream |
| `production_observation_adapter.py` | new module, 40 lines | not in upstream |

To reconstruct:

```
git apply reproduction/audits/ramen_multievidence_20261028/code/merge_proj_observation_export.diff
cp reproduction/audits/ramen_multievidence_20261028/code/{multi_evidence,production_observation_adapter}.py .
```

## What the delta actually does

1. **Observation export** (`SAI3D.proj_gaussian_features`, inside the per-view
   loop). For every visible superpoint in every view it records the raw
   `sp_mask_mat @ view_level_feature` vector, the `visibility_portion`, the
   member and visible Gaussian ids, and the level. This is the ONLY way the
   multi-evidence module gets its input — there is no CLI flag for it, the path
   comes from `RELAGS_OBSERVATION_OUT` / `--observation_out`, defaulting to empty
   (export off). The hook sits **before** ROFA filtering, before L2
   normalisation and before the `* visibility_portion` scaling at the end of the
   loop, so a reader must apply the weight exactly once. The audit's T2 test
   re-applies it once and reproduces the shipped `nag_feat[2]` at cos 0.999591.
2. **`--mix {soft,hard,top2}`** — how one superpoint's feature is composed from
   the SAM masks it overlaps in a view. Default `soft` = the upstream ratio
   mixture; `hard` keeps only the dominant mask; `top2` renormalises the two
   largest. Unused in the audited runs (default stayed in force).
3. **`--agg {mean,medoid,trimmed}`** — how the kept per-view features are
   aggregated after ROFA. Default `mean` = upstream. `medoid` returns a real
   observation, `trimmed` drops the most isolated quartile. Also unused.
4. **`.float()`** on `view.semantic["sem"]` — the fp16 view feature is widened
   before the matmul, avoiding an fp16 accumulation.
5. **`--seed`** — the per-mask random encodings that drive region growing are
   drawn from a seeded `torch.Generator` instead of unseeded `torch.normal`.
   **This is the flag the audit needed and did not have.** Without it the
   level-1/2/3 partitions differ between two runs with identical arguments; the
   audit measured 304 / 245 / 249 level-3 regions across three `sai_nag*.pt`
   files whose level-0 partitions are all 7936. Any A/B that compares two runs
   built separately is therefore comparing two different region sets.
6. **`saifine_labels.npy`** — dumps the finest label map next to the checkpoint
   so an evaluator can check id spaces without re-running the pipeline.

## What is NOT here

No GPU was available during the audit, so `T5` (re-running the certified
renderer) was verified by line-by-line comparison rather than execution. The
reports say this where it matters; nothing in them claims the algorithm itself
is ineffective.