#!/usr/bin/env bash
set -euo pipefail
BASE=/root/rivermind-data/repro/deco-b123/scannet_v2_certified_20261029
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
CODE=/root/rivermind-data/repro/deco-b123/scannet0062_m1m2m3_20261029/code
EVAL=$BASE/scannet_c1c2_m1m2m3_evaluate_certified.py
for s in scene0000_00 scene0062_00 scene0070_00 scene0097_00 scene0140_00 scene0200_00 scene0347_00 scene0400_00 scene0590_00 scene0645_00; do
  OUT=$BASE/$s
  E=$OUT/export
  if [ ! -f "$E/BANK_READY" ]; then echo "WAITING $s"; exit 2; fi
  ln -sfn "/root/rivermind-data/repro/deco-b123/scannet10_full_20261029/gt/${s}_vh_clean_2.labels.ply" "$E/${s}_vh_clean_2.labels.ply"
  "$PY" - "$E" "$OUT/model/sai_nag_3.0.pt" "$OUT/model/point_cloud/iteration_0/point_cloud.ply" "$E/B1.npz" "$E/B1.members.json" "$E/observations_level3.npz" "$E/${s}_vh_clean_2.labels.ply" <<'PY'
import sys, json, hashlib
from pathlib import Path
e=Path(sys.argv[1])
ps=[Path(x) for x in sys.argv[2:]]
e.joinpath('EXPORT_MANIFEST.json').write_text(json.dumps({'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}},indent=2))
PY
  R=$BASE/ablation/v2_off_off/$s
  if [ -f "$R/COMPLETE" ]; then echo "SKIP_EVAL $s"; continue; fi
  mkdir -p "$R"
  echo "START_EVAL $s $(date -Is)"
  SCANNET_SCENE="$s" SCANNET_C1=off SCANNET_C2=off SCANNET_INPUT_ROOT="$E" \
  SCANNET_ABLATION_ROOT="$R" SCANNET_CODE="$CODE" SCANNET_MODEL="$OUT/model" \
    "$PY" "$EVAL" 2>&1 | tee "$R/run.log"
  echo "DONE_EVAL $s $(date -Is)"
done
echo "V2_EVAL_DONE $(date -Is)"
