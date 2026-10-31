# PREREG — SREM readout on Phase A token field (frozen before computing)

## Signal
For each candidate, fixed per-view observation tokens X={f_o} (query-agnostic field).
Background B = tokens of the other candidates in the released top-5 pool (query-relative,
text-free). Similarities are z-scored within the candidate:
  s_o = <f_o,t>; b_o = max_{b in B} <f_o,b>; base_o = z(s)_o - lambda * z(b)_o.
Greedy select tokens up to k<=8 with marginal gain base_o - mu * max_{j in S} <f_o,f_j>;
stop when marginal <= 0. Score = mean base over selected - mu * mean pairwise cos(selected).
Primary: lambda=1, mu=1. Ablations: (1,0), (0,1).

## Stage 1 (candidate hit rates, no end-to-end)
For each query, pick argmax signal over candidates; hit if it equals the GT-best candidate.
Compare with id_top3 (Phase A reference: found 65.3%, rank2 27.8%, deep 73.7%, recall 24.1%).
Gate to proceed: SREM improves found or deep by >= 3 points versus id_top3, and does not drop
found by more than 2 points.
Report all ablations and adaptive-k distribution.

## Stage 2 (if Stage 1 passes)
End-to-end mask rerank with `lerf_maskagg_rerank.py --column srem` (S = z(NAG)+z(srem)),
then gateE with the same column. Gates as before (delta >= +0.01 vs main, bootstrap CI > 0,
no scene worse than -0.01, found not worse than C by >0.005).

No GT in selection; GT only for IoU/family. No tuning beyond the pre-registered lambda/mu.
Note: Phase A tokens are region-level mixtures; if SREM fails here, the next test is Phase B
mask tokens, not a different lambda.