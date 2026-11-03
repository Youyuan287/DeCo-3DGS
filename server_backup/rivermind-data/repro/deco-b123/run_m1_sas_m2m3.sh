#!/usr/bin/env bash
set -euo pipefail

PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODELROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
NAGROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GTROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
BASE=/root/rivermind-data/repro/deco-b123/m1_sas_m2m3_20261029
SASROOT=/root/rivermind-data/repro/deco-b123/m1_sas_20261029

mkdir -p "$BASE"
cd "$REPO"
export RELAGS_REPO="$RELAGS"
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

for scene in figurines ramen teatime waldo_kitchen; do
  bank="$SASROOT/${scene}_new/SAS.npz"
  members="$SASROOT/${scene}_new/SAS.members.json"
  obs="$SASROOT/${scene}_new/observations_level3.npz"
  out="$BASE/${scene}_m1_sas_m2m3"
  csv="$out/per_query.csv"
  mkdir -p "$out"
  "$PY" deco/lerf_ovs/eval_lerf_ovs_v2.py \
    --scene "$scene" \
    --source "$SRCROOT/$scene" \
    --model "$MODELROOT/$scene" \
    --nag "$NAGROOT/$scene/sai_nag.pt" \
    --gt_root "$GTROOT" \
    --out "$out" \
    --csv "$csv" \
    --topk 5 --level_until 3 --filter_small 50 \
    --center image --text canon --renorm 0 --sel gap \
    --tag m1_sas_m2m3 --evidence-v2 reliable \
    --evidence-bank "$bank" \
    --evidence-observations "$obs" \
    --evidence-members "$members" \
    --evidence-temperature 0.08 \
    2>&1 | tee "$BASE/${scene}.log"
done

echo "M1_SAS_M2M3_DONE $(date -Is)"
