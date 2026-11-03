#!/usr/bin/env bash
set -euo pipefail
: "${DECO_ROOT:?Path to FULL DeCo-3DGS clone}"
: "${RELAGS_REPO:?Path to existing ReLaGS code imported by DeCo}"
: "${NAG:?Frozen sai_nag_3.0.pt}"
: "${OBS:?Existing observations_level3.npz}"
: "${SRC:?Existing ramen source}"
: "${MODEL:?Existing ramen 3DGS model}"
: "${GT:?GT root whose child is ramen}"
PY="${PY:-python}"
OUT="${OUT:-$DECO_ROOT/_b123_eval}"
DECO_CENTER="${DECO_CENTER:-image}"
DECO_TEXT="${DECO_TEXT:-canon}"
DECO_RENORM="${DECO_RENORM:-0}"
DECO_SEL="${DECO_SEL:-gap}"
GEO_WEIGHT="${GEO_WEIGHT:-0.25}"
TOPK="${TOPK:-5}"
MAX_MODES="${MAX_MODES:-3}"
TEMP="${TEMP:-0.08}"
for f in "$NAG" "$OBS" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_b123.py"; do test -f "$f" || { echo "Missing $f" >&2; exit 1; }; done
export PYTHONPATH="$DECO_ROOT/deco:$RELAGS_REPO:$RELAGS_REPO/ext:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"
# Freeze source hashes once, not inferred from matching region counts.
sha256sum "$NAG" "$OBS" > "$OUT/INPUT_SHA256.txt"
for arm in K0 B1 B2; do
  if [[ "$arm" == B2 ]]; then b2=1; else b2=0; fi
  if [[ "$arm" == K0 ]]; then b1=0; K=1; gw=0; else b1=1; K="$MAX_MODES"; gw="$GEO_WEIGHT"; fi
  target="$OUT/${arm}.npz"
  if [[ ! -f "$target" ]]; then
    "$PY" "$DECO_ROOT/deco/lerf_ovs/build_bank_b123.py" --observations "$OBS" --nag "$NAG" --out "$target" --b1 "$b1" --b2 "$b2" --max-modes "$K" --geo-weight "$gw"
  else echo "Existing bank: $target (hash validated by evaluator)"; fi
done
# Check the three banks all come from the same frozen observation file.
"$PY" - "$OUT" "$OBS" <<'PYCHECK'
import sys,hashlib,json,numpy as np
from pathlib import Path
out=Path(sys.argv[1]);obs=Path(sys.argv[2]);h=hashlib.sha256(obs.read_bytes()).hexdigest()
for arm in ('K0','B1','B2'):
 with np.load(out/(arm+'.npz'),allow_pickle=False) as z:
  meta=json.loads(str(z['metadata'].item()))
  if meta.get('observation_sha256')!=h:raise SystemExit(f'{arm}: source observation SHA mismatch; STOP')
print('All three banks verified against one observation SHA:',h)
PYCHECK
# This is a 5-arm controlled ablation. B3 uses same bank as B2: only the query router changes.
for arm in B0 K0 B1 B2 B3; do
  case "$arm" in
    B0) B1=0; B2=0; B3=0; bank="" ;;
    K0) B1=0; B2=0; B3=0; bank="$OUT/K0.npz" ;;
    B1) B1=1; B2=0; B3=0; bank="$OUT/B1.npz" ;;
    B2) B1=1; B2=1; B3=0; bank="$OUT/B2.npz" ;;
    B3) B1=1; B2=1; B3=1; bank="$OUT/B2.npz" ;;
  esac
  folder="$OUT/$arm"; mkdir -p "$folder"
  summary="$folder/ramen_${arm}_summary.json"
  if [[ -e "$summary" ]]; then echo "Existing output $summary: stop rather than overwrite" >&2; exit 1; fi
  extra=()
  if [[ "$arm" == K0 ]]; then extra+=(--control-k1 1); fi
  if [[ -n "$bank" ]]; then extra+=(--evidence-bank "$bank"); fi
  "$PY" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_b123.py" \
    --scene ramen --source "$SRC" --model "$MODEL" --nag "$NAG" \
    --gt_root "$GT" --out "$folder" --csv "$folder/per_query.csv" \
    --topk "$TOPK" --level_until 3 --filter_small 50 \
    --center "$DECO_CENTER" --text "$DECO_TEXT" --renorm "$DECO_RENORM" --sel "$DECO_SEL" \
    --b1 "$B1" --b2 "$B2" --b3 "$B3" \
    --evidence-temperature "$TEMP" --tag "$arm" "${extra[@]}"
done
