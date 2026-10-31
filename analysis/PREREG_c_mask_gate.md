# PREREG — Per-query gating between module C and mask rerank (frozen before computing)

Motivation: module C and mask-level rerank change almost disjoint query sets and their simple
stack (mask order before C's prune) failed. Instead, choose per query between the two already
gate-passing selections using a training-free signal: whether C changes the released selection.

Frozen rule per query (released pool, k = released gap length g):
- S0 = released prefix (indices 0..g-1).
- S_C = module C's prune+include selection on S0 (constants unchanged).
- S_M = top-g by S = z(NAG score) + 1.0 * z(id_top3_mask) (z within the candidate list).
- if set(S_C) != set(S0): select S_C (C is "active" on this query);
  else: select S_M.
No GT, no prompts, no per-query tuning. Primary base: DeCo B0 (`--center image`), because the
module C preds/evidence artefacts were generated in that configuration. ReLaGS (`--center none`)
requires regenerating preds/evidence; deferred.

Gates:
1. vs module C: delta >= +0.01, paired scene-stratified bootstrap (B=2000, seed=20261030) CI lower > 0;
2. vs mask rerank: delta >= +0.01, CI lower > 0;
3. no scene worse than -0.01 (relative to main);
4. found family not worse than module C by >0.005.
If it fails, report negative; do not tune the gate signal.

Secondary (reported, not adopted): the symmetric rule (if mask changes selection use mask, else C).