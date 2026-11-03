#!/usr/bin/env bash
set -euo pipefail
DECO=/root/rivermind-data/repro/deco-b123/DeCo-3DGS
RELAGS=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen
MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen
NAG=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen/sai_nag_3.0.pt
OBS=/root/rivermind-data/repro/deco-b123/waldo_b123_20261028/observations_level3.npz
GT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label
OUT=/root/rivermind-data/repro/deco-b123/waldo_b123_20261028/eval
export PYTHONPATH="$DECO:$DECO/deco:$RELAGS:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
sha256sum "$NAG" "$OBS" > "$OUT/INPUT_SHA256.txt"
for arm in K0 B1 B2; do
  case "$arm" in
    K0) b1=0; b2=0; modes=1; gw=0 ;;
    B1) b1=1; b2=0; modes=3; gw=0.25 ;;
    B2) b1=1; b2=1; modes=3; gw=0.25 ;;
  esac
  "$PY" "$DECO/deco/lerf_ovs/build_bank_b123.py" --observations "$OBS" --nag "$NAG" --out "$OUT/$arm.npz" --b1 "$b1" --b2 "$b2" --max-modes "$modes" --geo-weight "$gw"
done
"$PY" - "$OUT" "$OBS" <<'PY'
import sys,hashlib,json,numpy as np
from pathlib import Path
out=Path(sys.argv[1]); obs=Path(sys.argv[2]); h=hashlib.sha256(obs.read_bytes()).hexdigest()
for arm in ('K0','B1','B2'):
 with np.load(out/(arm+'.npz'),allow_pickle=False) as z:
  meta=json.loads(str(z['metadata'].item()))
  assert meta.get('observation_sha256')==h, (arm,meta.get('observation_sha256'),h)
print('BANKS_OK',h)
PY
for arm in B0 K0 B1 B2 B3; do
  case "$arm" in
    B0) b1=0; b2=0; b3=0; bank=""; extra=() ;;
    K0) b1=0; b2=0; b3=0; bank="$OUT/K0.npz"; extra=(--control-k1 1) ;;
    B1) b1=1; b2=0; b3=0; bank="$OUT/B1.npz"; extra=() ;;
    B2) b1=1; b2=1; b3=0; bank="$OUT/B2.npz"; extra=() ;;
    B3) b1=1; b2=1; b3=1; bank="$OUT/B2.npz"; extra=() ;;
  esac
  folder="$OUT/$arm"; mkdir -p "$folder"
  args=(--scene waldo_kitchen --source "$SRC" --model "$MODEL" --nag "$NAG" --gt_root "$GT" --out "$folder" --csv "$folder/per_query.csv" --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap --b1 "$b1" --b2 "$b2" --b3 "$b3" --evidence-temperature 0.08 --tag "$arm")
  if [[ -n "$bank" ]]; then args+=(--evidence-bank "$bank"); fi
  if [[ "$arm" == K0 ]]; then args+=(--control-k1 1); fi
  "$PY" "$DECO/deco/lerf_ovs/eval_lerf_ovs_b123.py" "${args[@]}"
done