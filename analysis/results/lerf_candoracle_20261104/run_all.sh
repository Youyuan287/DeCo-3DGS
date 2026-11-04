#!/usr/bin/env bash
set -uo pipefail
RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
R=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
export RELAGS_REPO HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$R:$R/deco/lerf_ovs:$RELAGS_REPO:$RELAGS_REPO/ext:$RELAGS_REPO/ext/spt/dependencies/grid_graph/python/bin:$RELAGS_REPO/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS_REPO/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
cd /root/rivermind-data/repro/deco-b123/lerf_candoracle_20261104
mkdir -p out
for sc in figurines ramen teatime waldo_kitchen; do
  echo "START $sc $(date -Is)"
  $PY lerf_ovs_cand_oracle.py --scene $sc --out out > out/$sc.log 2>&1
  echo "END $sc rc=$? $(date -Is)"
  grep -E "SCENE_DONE|Traceback|Error" out/$sc.log | tail -3
done
$PY lerf_ovs_cand_oracle_analyze.py --oracle_dir out --out out > out/analysis.log 2>&1
echo "ANALYZE rc=$?"
echo ALL_DONE
