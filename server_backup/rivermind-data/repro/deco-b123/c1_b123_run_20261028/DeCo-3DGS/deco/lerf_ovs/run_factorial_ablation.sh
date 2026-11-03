#!/usr/bin/env bash
# Fixed V3 two-factor ablation plus isolated M1 raw-vs-debiased grouping check.
set -euo pipefail

ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
CODE="$ROOT/DeCo-3DGS"
EVAL="$CODE/deco/lerf_ovs/eval_lerf_ovs_factorial.py"
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REL=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
GT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label
OUT="$ROOT/evidence_v3_factorial_20261029"
OLD="$ROOT/evidence_v3_relative_20261029"
export RELAGS_REPO="$REL"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
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
    *) echo "unknown scene: $1" >&2; exit 2 ;;
  esac
  SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$1
  MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$1
  NAG="$MODEL/sai_nag.pt"
}

write_manifest() {
  local scene="$1" arm="$2" dir="$3" bank="$4" obs="$5" nag="$6" members="$7" evidence_mode="$8"
  SCENE="$scene" ARM="$arm" RUN_DIR="$dir" BANK="$bank" OBS="$obs" NAG="$nag" MEMBERS="$members" EVIDENCE_MODE="$evidence_mode" EVAL="$EVAL" "$PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
d=Path(os.environ['RUN_DIR'])
payload={'scene':os.environ['SCENE'],'arm':os.environ['ARM'],
 'bank':os.environ['BANK'],'observation':os.environ['OBS'],'nag':os.environ['NAG'],'members':os.environ['MEMBERS'],
 'sha256':{k:sha(os.environ[k]) for k in ('BANK','OBS','NAG','MEMBERS','EVAL')},
 'protocol':{'center':'image','text':'canon','renorm':0,'sel':'gap','topk':5,'level_until':3,'filter_small':50,
             'temperature':.08,'coefficient':.25,'M2':os.environ['EVIDENCE_MODE']}}
(d/'manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True))
PY
}

run_arm() {
  local scene="$1" arm="$2" scope="$3" gain="$4" bank="$5" members="$6" evidence_mode="${7:-reliable}"
  case_inputs "$scene"
  local dir="$OUT/$scene/$arm"
  mkdir -p "$dir"
  if [[ -f "$dir/per_query.csv" && -f "$dir/${scene}_${arm}_summary.json" ]]; then
    echo "REUSE_COMPLETED $scene $arm"
    return
  fi
  write_manifest "$scene" "$arm" "$dir" "$bank" "$OBS" "$NAG" "$members" "$evidence_mode"
  echo "RUN $scene $arm scope=$scope gain=$gain"
  "$PY" "$EVAL" --scene "$scene" --source "$SOURCE" --model "$MODEL" --nag "$NAG" --gt_root "$GT" \
    --out "$dir" --csv "$dir/per_query.csv" --tag "$arm" \
    --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap \
    --evidence-v2 "$evidence_mode" --evidence-bank "$bank" --evidence-observations "$OBS" --evidence-members "$members" \
    --evidence-temperature .08 --factor-scope "$scope" --factor-gain "$gain" >"$dir/run.log" 2>&1
  write_manifest "$scene" "$arm" "$dir" "$bank" "$OBS" "$NAG" "$members" "$evidence_mode"
}

build_raw() {
  local scene="$1"
  case_inputs "$scene"
  local dir="$OUT/generated_raw/$scene" raw="$OUT/generated_raw/$scene/B1_raw.npz"
  mkdir -p "$dir"
  if [[ ! -f "$raw" ]]; then
    "$PY" "$CODE/deco/lerf_ovs/build_raw_grouping_bank.py" --observations "$OBS" --nag "$NAG" --out "$raw" --max-modes 3 --geo-weight .25 >"$dir/build.log" 2>&1
  fi
  "$PY" - "$BANK" "$raw" "$dir/bank_control.json" <<'PY'
import hashlib,json,sys,numpy as np
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
deb,raw,out=sys.argv[1:]
with np.load(deb,allow_pickle=False) as a,np.load(raw,allow_pickle=False) as b:
 p={'debiased_sha256':sha(deb),'raw_sha256':sha(raw),'region_ids_equal':bool(np.array_equal(a['region_ids'],b['region_ids'])),
 'features_shape_equal':a['features'].shape==b['features'].shape,'raw_group_counts':b['group_count'].tolist(),
 'debiased_group_counts':a['group_count'].tolist(),'raw_metadata':json.loads(str(b['metadata'].item()))}
open(out,'w').write(json.dumps(p,indent=2))
PY
  echo "$raw"
}

for scene in figurines ramen teatime waldo_kitchen; do
  case_inputs "$scene"
  # The two previously absent cells; all+absolute reliable and selected+relative reliable stay reused from V2/V3.
  run_arm "$scene" selected_absolute_reliable selected absolute "$BANK" "$MEMBERS"
  run_arm "$scene" all_relative_reliable all relative "$BANK" "$MEMBERS"
done

for scene in figurines ramen teatime waldo_kitchen; do
  case_inputs "$scene"
  raw=$(build_raw "$scene")
  raw_members="${raw%.npz}.members.json"
  # M1-only check: same absolute/all M3 readout, same main path, no M2 in either grouping arm.
  run_arm "$scene" raw_all_absolute_semantic all absolute "$raw" "$raw_members" semantic
done

touch "$OUT/COMPLETE"
