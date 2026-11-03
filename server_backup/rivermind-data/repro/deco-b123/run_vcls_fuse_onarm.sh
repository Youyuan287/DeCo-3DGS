#!/bin/bash
R=/root/rivermind-data/ReLaGS_Reproduction; B=/root/rivermind-data/repro/deco-b123
V=$R/venv/bin/python; DEP=$R/ReLaGS/ext/spt/dependencies
export PYTHONPATH=$B/c1_b123_run_20261028/DeCo-3DGS/deco/scannet:$R/ReLaGS:$R/ReLaGS/ext:$DEP/grid_graph/python/bin:$DEP/parallel_cut_pursuit/python/bin:$DEP/parallel_cut_pursuit/python/wrappers
export HF_HUB_OFFLINE=1
OUT=$B/vcls_fuse_onarm_20261102; mkdir -p $OUT
for S in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  GT=$B/scannet10_full_20261029/$S
  [ -d "$GT" ] || GT=$B/scannet_v2_certified_20261029/$S/export
  echo "===== $S $(date -Is)"
  $V $B/scannet_vcls_fuse.py --scene $S --model /root/verify58_20261029/full10/$S/model \
    --input $B/scannet_v2_certified_20261029/$S/export --gt_root $GT \
    --released_npz $B/scannet_v2_certified_20261029/ablation/v2_on_on/$S/predictions_and_scores.npz \
    --alphas 0,0.5 --betas 0,0.25,0.5,0.75 > $OUT/$S.log 2>&1
  grep RESULT $OUT/$S.log | tail -1
done
echo ALL_DONE
