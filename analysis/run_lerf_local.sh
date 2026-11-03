#!/usr/bin/env bash
set -uo pipefail
R=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODELROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GTROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
SAS=/root/rivermind-data/repro/deco-b123/m1_sas_20261029
OUT=/root/rivermind-data/repro/deco-b123/lerf_local_20261103
mkdir -p "$OUT"
export RELAGS_REPO="$RELAGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$R:$R/deco/lerf_ovs:$RELAGS:$RELAGS/ext"
cd "$R"
for scene in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
  b="$SAS/${scene}_new"; [ -d "$b" ] || b="$SAS/waldo_new"
  echo "START $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_local_contrast.py \
    --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --nag "$MODELROOT/$scene/sai_nag.pt" --sas "$b" --gt_root "$GTROOT" \
    --out "$OUT" > "$OUT/$scene.log" 2>&1
  echo "END $scene rc=$? $(date -Is)"
  grep -h "^RESULT" "$OUT/$scene.log" | tail -1
done
echo ALL_DONE