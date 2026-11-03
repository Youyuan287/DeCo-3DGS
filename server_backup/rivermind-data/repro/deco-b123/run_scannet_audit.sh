#!/usr/bin/env bash
set -uo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
CODE=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS/deco/scannet
B=/root/rivermind-data/repro/deco-b123
DATA=$B/scannet_v2_certified_20261029
MODEL=/root/verify58_20261029/full10
OUT=$B/scannet_audit_20261101
mkdir -p "$OUT"
for S in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  echo "START $S $(date -Is)"
  "$PY" $CODE/scannet_audit_eval.py --scene $S --input $DATA/$S/export --model $MODEL/$S/model --out $OUT > "$OUT/$S.log" 2>&1
  echo "END $S rc=$? $(date -Is)"; tail -1 "$OUT/$S.log"
done
echo ALL_DONE
