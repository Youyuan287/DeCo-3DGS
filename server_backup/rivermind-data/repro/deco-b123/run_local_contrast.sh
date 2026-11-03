#!/usr/bin/env bash
set -uo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
B=/root/rivermind-data/repro/deco-b123
D=$B/scannet_v2_certified_20261029
MODEL=/root/verify58_20261029/full10
OUT=$B/local_contrast_20261103
mkdir -p "$OUT"
for S in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 \
         scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  echo "START $S $(date -Is)"
  "$PY" "$B/scannet_local_contrast.py" \
    --scene "$S" \
    --input "$D/$S/export" \
    --gt_root "$D/$S/export" \
    --model "$MODEL/$S/model" \
    --released_npz "$D/ablation/v2_off_off/$S/predictions_and_scores.npz" \
    --out "$OUT" > "$OUT/$S.log" 2>&1
  echo "END $S rc=$? $(date -Is)"
  grep -h "^RESULT" "$OUT/$S.log" | tail -1
done
echo ALL_DONE