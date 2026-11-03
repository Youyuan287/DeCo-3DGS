#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
CODE="$ROOT/DeCo-3DGS"
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REL=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
GT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label
OUT="$ROOT/evidence_m3_query_adaptive_20261029"
export RELAGS_REPO="$REL" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$CODE/deco/lerf_ovs:$REL:$REL/ext${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

case_inputs() {
  case "$1" in
    ramen)
      OBS=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz
      BANK=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.npz
      MEMBERS=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.members.json ;;
    waldo_kitchen)
      OBS=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz
      BANK=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.npz
      MEMBERS=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.members.json ;;
    figurines|teatime)
      OBS="$ROOT/evidence_v2_four_20261028/generated_inputs/$1/observations_level3.npz"
      BANK="$ROOT/evidence_v2_four_20261028/generated_inputs/$1/B1.npz"
      MEMBERS="$ROOT/evidence_v2_four_20261028/generated_inputs/$1/B1.members.json" ;;
    *) exit 2 ;;
  esac
  SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$1
  MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$1
  NAG="$MODEL/sai_nag.pt"
}

for scene in figurines ramen teatime waldo_kitchen; do
  case_inputs "$scene"
  DIR="$OUT/$scene/fixed_prior"
  mkdir -p "$DIR"
  "$PY" - "$scene" "$DIR" "$BANK" "$OBS" "$NAG" "$MEMBERS" "$CODE/deco/lerf_ovs/eval_lerf_ovs_fixed.py" <<'PY'
import hashlib,json,sys
from pathlib import Path
scene,d,b,o,n,m,e=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
Path(d,'manifest.json').write_text(json.dumps({'scene':scene,'arm':'fixed_prior','bank':b,'observation':o,'nag':n,'members':m,'sha256':{k:sha(v) for k,v in [('BANK',b),('OBS',o),('NAG',n),('MEMBERS',m),('EVAL',e)]},'protocol':{'M2':'semantic','center':'image','text':'canon','renorm':0,'sel':'gap','topk':5,'level_until':3,'filter_small':50,'temperature':.08,'coefficient':.25,'weight_mode':'normalized_priors'}},indent=2))
PY
  echo "RUN $scene fixed_prior"
  "$PY" "$CODE/deco/lerf_ovs/eval_lerf_ovs_fixed.py" --scene "$scene" --source "$SOURCE" --model "$MODEL" --nag "$NAG" --gt_root "$GT" --out "$DIR" --csv "$DIR/per_query.csv" --tag fixed_prior --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap --evidence-v2 semantic --evidence-bank "$BANK" --evidence-observations "$OBS" --evidence-members "$MEMBERS" --evidence-temperature .08 >"$DIR/run.log" 2>&1
done
touch "$OUT/COMPLETE"
