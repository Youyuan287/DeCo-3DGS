#!/usr/bin/env bash
set -euo pipefail
DECO_ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
OUT=$ROOT/evidence_deco_li_20261029
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
export HF_HUB_OFFLINE=1 PYTHONPATH=$DECO_ROOT:$RELAGS_REPO:$RELAGS_REPO/ext
mkdir -p "$OUT"

run_scene() {
  local arm="$1" scene="$2"
  case "$scene" in
    figurines) source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines; model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines; nag=$model/sai_nag.pt; bank=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/B1.npz; obs=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/observations_level3.npz; members=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/B1.members.json ;;
    ramen) source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen; model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen; nag=$model/sai_nag.pt; bank=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.npz; obs=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz; members=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.members.json ;;
    teatime) source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/teatime; model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/teatime; nag=$model/sai_nag.pt; bank=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/B1.npz; obs=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/observations_level3.npz; members=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/B1.members.json ;;
    waldo_kitchen) source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen; model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen; nag=$model/sai_nag.pt; bank=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.npz; obs=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz; members=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.members.json ;;
  esac
  local d="$OUT/$arm/$scene"; mkdir -p "$d"
  if [[ "$arm" == "E0" ]]; then
    "$PY" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_v3.py" --scene "$scene" --source "$source" --model "$model" --nag "$nag" --gt_root "$GT" --out "$d" --csv "$d/per_query.csv" --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap --evidence-v2 off --tag E0 > "$d/run.log" 2>&1
  else
    local method=softmax
    [[ "$arm" == "E2" ]] && method=maxsim
    [[ "$arm" == "E3" ]] && method=prior_maxsim
    "$PY" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_v3.py" --scene "$scene" --source "$source" --model "$model" --nag "$nag" --gt_root "$GT" --out "$d" --csv "$d/per_query.csv" --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap --evidence-v2 semantic --evidence-bank "$bank" --evidence-observations "$obs" --evidence-members "$members" --evidence-temperature 0.08 --evidence-coefficient 0.25 --evidence-aggregation "$method" --tag "$arm" > "$d/run.log" 2>&1
  fi
}

ARMS="${ARMS:-E0 E1 E2 E3}"
for arm in $ARMS; do
  pids=()
  for scene in figurines ramen teatime waldo_kitchen; do run_scene "$arm" "$scene" & pids+=("$!"); done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo COMPLETE > "$OUT/COMPLETE"
