# M1-RCS: Cross-View Repeated-Support Complementary Semantic Selection

This experiment keeps the frozen DeCo V2 main path, NAG, Gaussian model, observations, canon scoring, ROFA/gap search, and V2 Softmax readout unchanged. It changes only the M1 bank seed-selection step.

For each level-3 region, residual observation features are L2-normalized. For observation `i`, each *other* view contributes at most one positive semantic match, `b_{iv}=max(0,max_{j:view(j)=v}<h_i,h_j>)`. View masses are the sums of positive observation weights in that view. The cross-view support score is the mass-weighted mean

`r_i = sum_v mass_v b_iv / sum_v mass_v`,

and `c_i` is the number of other views with `b_iv>0`. RCS requires `c_i>=1` for a seed. The first seed maximizes `(r_i, stable_index)`; subsequent seeds use the original affinity novelty `1-max_{s in S} A[i,s]`, restricted to supported observations, with support-score and stable-index tie breaks. Assignment, ROFA, prototype construction, priors, and V2 evaluation are otherwise identical to the original bank.

No GT, new CLIP extraction, NAG construction, observation export, M2, or M3 modification is used. Original mode selection remains available with `--mode-selection original`; RCS uses `--mode-selection rcs`.

The run script is `run_m1_rcs.sh`. It writes only to `evidence_m1_rcs_20261029/` and uses the four frozen scene inputs.

## Verification and command

CPU unit tests: `PYTHONPATH=deco/lerf_ovs python tests/test_m1_rcs.py -v` (8/8 passed on fj01). The production evaluator command was unchanged apart from the bank path, e.g. `python deco/lerf_ovs/eval_lerf_ovs_v3.py --scene ramen ... --evidence-v2 semantic --evidence-aggregation softmax --evidence-temperature 0.08 --evidence-coefficient 0.25 --sel gap`.

The first teatime bank exposed a loader invariant for regions with no supported auxiliary mode (`priors.sum()==1` is required even when `valid=False`). The isolated output was repaired by writing a finite zero feature/residual dummy slot with prior 1 and `valid=False`; this does not participate in readout. No production bank was changed.

RCS is therefore a bank-construction ablation only. It does not claim a new M3 effect: M2 remains OFF and the certified V2 Softmax evaluator is reused exactly.
