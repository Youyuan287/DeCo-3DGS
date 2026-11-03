#!/usr/bin/env bash
set -euo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
BASE=/root/rivermind-data/repro/deco-b123/scannet10_full_20261029
SCRIPT=$BASE/scannet_c1c2_m1m2m3_evaluate.py
CODE=/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code
MODELBASE=/root/rivermind-data/output/scannet
for mode in off_off c1_on_c2_off c1_off_c2_on c1_on_c2_on; do
  case "$mode" in
    off_off) C1=off; C2=off ;;
    c1_on_c2_off) C1=on; C2=off ;;
    c1_off_c2_on) C1=off; C2=on ;;
    c1_on_c2_on) C1=on; C2=on ;;
  esac
  for scene in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
    if [ "$scene" = "scene0062_00" ]; then
      INP=/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029
    else
      INP=$BASE/$scene
    fi
    OUT=$BASE/c1c2_ablation_20261029/$mode/$scene
    mkdir -p "$OUT"
    if [ -f "$OUT/COMPLETE" ]; then
      echo "SKIP $mode $scene"
      continue
    fi
    SCANNET_SCENE="$scene" SCANNET_C1="$C1" SCANNET_C2="$C2" SCANNET_INPUT_ROOT="$INP" SCANNET_ABLATION_ROOT="$OUT" SCANNET_CODE="$CODE" SCANNET_MODEL="$MODELBASE/$scene" \
      "$PY" "$SCRIPT" > "$OUT/run.log" 2>&1
    echo "DONE $mode $scene"
  done
done
echo COMPLETE
