#!/usr/bin/env bash
# Same as run_certified_v2_eval.sh but on the DeCo main arm (C1 mean-subtract on, C2 both_lse on).
# Parity against the unchanged published scorer is asserted inside the evaluator; a mismatch aborts the batch.
set -euo pipefail
BASE=/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
CODE=/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code
EVAL=$BASE/scannet_c1c2_m1m2m3_evaluate_certified.py
for s in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  OUT=$BASE/$s
  E=$OUT/export
  [ -f "$E/BANK_READY" ] && [ -f "$E/EXPORT_MANIFEST.json" ] || { echo "MISSING_INPUT $s"; exit 2; }
  R=$BASE/ablation/v2_on_on/$s
  if [ -f "$R/COMPLETE" ]; then echo "SKIP_EVAL $s"; continue; fi
  mkdir -p "$R"
  echo "START_EVAL $s $(date -Is)"
  SCANNET_SCENE="$s" SCANNET_C1=on SCANNET_C2=on SCANNET_INPUT_ROOT="$E" \
  SCANNET_ABLATION_ROOT="$R" SCANNET_CODE="$CODE" SCANNET_MODEL="$OUT/model" \
    "$PY" "$EVAL" 2>&1 | tee "$R/run.log"
  echo "DONE_EVAL $s $(date -Is)"
done
echo "V2_ONON_EVAL_DONE $(date -Is)"
