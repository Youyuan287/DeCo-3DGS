#!/usr/bin/env bash
set -euo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
BASE=/root/rivermind-data/repro/deco-b123/scannet10_full_20261029
ROOT=/root/rivermind-data/repro/deco-b123
for scene in scene0000_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  out="$BASE/$scene"
  if [ -e "$out/observations_level3.npz" ]; then echo "SKIP existing $scene"; continue; fi
  if [ -d "$out" ] && [ ! -e "$out/observations_level3.npz" ]; then rm -rf "$out"; fi
  "$PY" "$ROOT/m1_sas_export.py" \
    --repo "$REPO" \
    --source "/root/rivermind-data/datasets/scannet/$scene" \
    --model "/root/rivermind-data/output/scannet/$scene" \
    --nag "/root/rivermind-data/output/scannet/$scene/sai_nag_3.0.pt" \
    --outdir "$out" --seed 1234 --iteration 0 \
    2>&1 | tee "$BASE/${scene}_export.log"
  "$PY" "$ROOT/build_m1_sas_bank.py" \
    --observations "$out/observations_level3.npz" \
    --sidecar "$out/observation_semantic_sidecar.npz" \
    --out "$out/B1.npz" --mode semantic --max-modes 3 --geo-weight .25 \
    2>&1 | tee "$BASE/${scene}_bank.log"
  touch "$out/BANK_READY"
done
echo SCANNET_NINE_EXPORT_DONE $(date -Is)
