# PREREG — Margin-based principled gating (frozen before computing)

Problem with the current gate: it uses "the mask pathway changed the released selection" as a proxy
for reliability. That is heuristic. We replace it with a stability/margin criterion.

## Theory (two equivalent justifications)

1. **Distribution-free stability.** Let a pathway produce scores s over the candidate pool and select
   S = top-k. If every score is perturbed by at most delta in L-infinity, membership of S is unchanged
   whenever  Delta(S) := min_{i in S} s_i - max_{j not in S} s_j > 2 delta.
   Hence a selection with larger Delta tolerates a larger score perturbation; choosing the pathway with
   the larger Delta maximizes the guaranteed stability radius.
2. **Gaussian noise model.** If s_c = mu_c + eps_c, eps ~ N(0, sigma^2), and the true object has the
   largest mu, then P(top-1 correct) = Phi(Delta_1 / (sigma sqrt(2))) (Delta_1 = top1 - top2 margin),
   which is monotone in the standardized margin. For two pathways with comparable sigma, the larger
   standardized margin has the higher probability of a correct top-1.

Neither justification uses labels; both lead to the same rule: **prefer the more decisive pathway.**

## Frozen rule (per query)
- S0 = released prefix; S_C = module C selection; S_M = mask rerank selection.
- zn = z(NAG scores), zm = z(S) with S = z(NAG) + 1.0 z(id_top3_mask) (z within the candidate list).
- margin(X, s) = min_{i in X} s_i - max_{j not in X} s_j; empty -> -inf, full -> +inf.
- m_nag = margin(S0, zn); m_mask = margin(S_M, zm).
- gateC: if set(S_M) != set(S0) and m_mask > m_nag: select S_M; else select S_C.
Rationale for the fallback: when the mask evidence is not more decisive than NAG, do not change the
released identity; use module C's conservative correction.

Primary base: DeCo B0 (`--center image`). Gate same as before: vs gateB (current best) delta >= +0.005
with bootstrap CI lower > 0; vs C and vs mask positive; found not worse than C by >0.005; no scene
worse than -0.01. If it does not beat gateB but is within noise, it is reported as the principled
alternative, not adopted.

No tuned thresholds: the comparison is parameter-free (m_mask > m_nag).