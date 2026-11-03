#!/bin/bash
R=/root/rivermind-data/ReLaGS_Reproduction; B=/root/rivermind-data/repro/deco-b123; D=$B/c1_b123_run_20261028/DeCo-3DGS; V=$R/venv/bin/python; DEP=$R/ReLaGS/ext/spt/dependencies
export RELAGS_REPO=$R/ReLaGS HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$D:$D/deco/lerf_ovs:$R/ReLaGS:$R/ReLaGS/ext:$DEP/grid_graph/python/bin:$DEP/parallel_cut_pursuit/python/bin:$DEP/parallel_cut_pursuit/python/wrappers"
OUT=$B/vcn_adj_20261102
for sc in figurines ramen teatime waldo_kitchen; do
  b=$B/m1_sas_20261029/${sc}_new; [ -d "$b" ] || b=$B/m1_sas_20261029/waldo_new
  $V $D/deco/lerf_ovs/lerf_vcn_adjacent.py --scene $sc --source /root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$sc --model $R/ref/relags_hf/$sc --nag $R/ref/relags_hf/$sc/sai_nag.pt --gt_root $R/ref/hf_lerf/lerf_ovs/label --sas $b --out $OUT > $OUT/$sc.log 2>&1
  grep RESULT $OUT/$sc.log | tail -1
done
echo ALL_DONE
