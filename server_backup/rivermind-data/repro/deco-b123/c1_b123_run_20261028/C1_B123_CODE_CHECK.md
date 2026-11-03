# C1_B123_CODE_CHECK

- isolated checkout: `/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS`
- upstream commit: `fbbb7f9c1a971f71d971db9e76aba5ce77336f07`
- generator SHA256: `62838a541af68e32dbb9f88c7a450b74d2f720928047fe39db06bbac6266662f`
- generated evaluator SHA256: `ca873060a65ebaae744ee2509dd38a4aba8fd509a3f2b7cbb11d68a4afd50621`
- original evaluator unchanged in isolated checkout: `4704416e535c5f933e8bffe163d05cbc29615575e9718efd4c46f10cb381119d`
- CPU unittest: `4/4 passed`
- py_compile: passed
- C1 refactor: raw root mean/count/dim captured immediately after CUDA feature load; duplicate post-C1 cache removed; B1 shared direction untouched.
- center=none/image uses the evaluator switch; no custom C1 subtraction was added.
- C1 ON/OFF bank content: identical matching bank SHA per scene.
- B2 ROFA: existing bank metadata reports zero deleted observations; no tau or other constant was changed.
- No merge_proj.py, observation export, NAG clustering or training was run in this experiment.

## Input SHA256
- ramen NAG: `10e3f12717a7b960358d42d862287dd94de3ba2ccb8f256f3d643197868f2324`
- ramen observations: `6ae30f6e03b01177b14e900a447d71ee788ef57a31cac9909b0abf0d33358232`
- waldo_kitchen NAG: `6cd39db45d60031ab04eac18630a918c311890de22220e07678118a6da903098`
- waldo_kitchen observations: `2586d87fb86eb4dee8885889b6e057363450ef710e06d4e2ca8d9fa72702fbf8`
