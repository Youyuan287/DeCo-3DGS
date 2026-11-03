#!/usr/bin/env bash
set -euo pipefail
BASE=/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
for s in scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  OUT="$BASE/$s"
  E="$OUT/export"
  if [ -f "$E/BANK_READY" ]; then echo "SKIP $s"; continue; fi
  mkdir -p "$OUT"
  if [ -d "$E" ] && [ -z "$(find "$E" -mindepth 1 -maxdepth 1 -print -quit)" ]; then rmdir "$E"; fi
  echo "START_EXPORT $s $(date -Is)"
  "$PY" /root/rivermind-data/repro/deco-b123/m1_sas_export.py \
    --repo "$REPO" --source "/root/rivermind-data/datasets/scannet/$s" \
    --model "$OUT/model" --nag "$OUT/model/sai_nag_3.0.pt" \
    --outdir "$E" --seed 1234 --iteration 0 \
    2>&1 | tee "$BASE/${s}_export.log"
  "$PY" /root/rivermind-data/repro/deco-b123/build_m1_sas_bank.py \
    --observations "$E/observations_level3.npz" \
    --sidecar "$E/observation_semantic_sidecar.npz" \
    --out "$E/B1.npz" --mode semantic --max-modes 3 --geo-weight .25 \
    2>&1 | tee "$BASE/${s}_bank.log"
  ln -sfn "/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/gt/${s}_vh_clean_2.labels.ply" "$E/${s}_vh_clean_2.labels.ply"
  touch "$E/BANK_READY"
  echo "DONE $s $(date -Is)"
done
echo "BATCH_EXPORT_DONE $(date -Is)"
