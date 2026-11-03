#!/usr/bin/env bash
set -euo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
BASE=/root/rivermind-data/repro/deco-b123/scannet10_full_20261029
TEMPLATE=/root/rivermind-data/repro/deco-b123/scannet_m3_evaluate_generic.py
SCENES="scene0000_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00"
for scene in $SCENES; do
  ROOT="$BASE/$scene"
  mkdir -p "$ROOT/code"
  cp /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/eval_scannet_3d.py "$ROOT/code/"
  cp /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/evidence_readout_v2.py "$ROOT/code/"
  cp /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/evidence_readout_weight_ablation.py "$ROOT/code/"
  cp /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/score_scannet.py "$ROOT/code/"
  cp /root/rivermind-data/repro/deco-b123/scannet0062_m3_weights_20261029/code/deco_ops.py "$ROOT/code/"
  cp /root/rivermind-data/repro/deco-b123/scannet_m3_evaluate_generic.py "$ROOT/code/scannet_m1m2m3_evaluate.py"
  cp "$BASE/gt/${scene}_vh_clean_2.labels.ply" "$ROOT/"
  touch "$ROOT/BANK_READY"
  SCANNET_SCENE="$scene" SCANNET_ABLATION_ROOT="$ROOT" \
    SCANNET_MODEL="/root/rivermind-data/output/scannet/$scene" \
    SCANNET_SRC="/root/rivermind-data/datasets/scannet/$scene" \
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$ROOT/code/scannet_m1m2m3_evaluate.py" \
    2>&1 | tee "$BASE/${scene}_ablation.log"
done
echo SCANNET_NINE_ABLATION_DONE $(date -Is)
