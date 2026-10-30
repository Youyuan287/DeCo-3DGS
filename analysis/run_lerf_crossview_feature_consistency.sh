#!/usr/bin/env bash
set -uo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODELROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GTROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
OUT=/root/rivermind-data/repro/deco-b123/lerf_crossview_20261031
mkdir -p "$OUT"
cd "$REPO"
export RELAGS_REPO="$RELAGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
for scene in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
  echo "START $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_crossview_feature_consistency.py --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --gt_root "$GTROOT" --out "$OUT/${scene}_crossview.csv" > "$OUT/$scene.log" 2>&1
  echo "END $scene rc=$? $(date -Is)"; grep -E "SCENE_DONE|Error|error|Traceback|same-object|different-object|retrieval" "$OUT/$scene.log" | tail -5
done
echo ALL_DONE