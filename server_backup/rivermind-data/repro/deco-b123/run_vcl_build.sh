#!/bin/bash
R=/root/rivermind-data/ReLaGS_Reproduction; B=/root/rivermind-data/repro/deco-b123
V=$R/venv/bin/python; DEP=$R/ReLaGS/ext/spt/dependencies
export PYTHONPATH=$R/ReLaGS:$R/ReLaGS/ext:$DEP/grid_graph/python/bin:$DEP/parallel_cut_pursuit/python/bin:$DEP/parallel_cut_pursuit/python/wrappers
export HF_HUB_OFFLINE=1
cd $R/ReLaGS
for S in scene0000_00; do
  echo "== VCL merge $S $(date -Is)"
  $V merge_proj_vcl.py -s /root/rivermind-data/datasets/scannet/$S -m $B/vcl_models/$S --thres_connect 0.9,0.7,0.7 --thres_merge 100 --tau 3 --iteration 0 > $B/vcl_models/$S.log 2>&1
  echo "exit=$? $(date -Is)"; tail -3 $B/vcl_models/$S.log
done
echo VCL_BUILD_DONE
