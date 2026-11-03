#!/usr/bin/env bash
set -euo pipefail
DECO_ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
OUT=$ROOT/evidence_m1_rcs_20261029
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
export HF_HUB_OFFLINE=1 PYTHONPATH=$DECO_ROOT:$RELAGS_REPO:$RELAGS_REPO/ext
mkdir -p "$OUT"

case_inputs() {
  case "$1" in
    figurines) SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines; MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines; NAG=$MODEL/sai_nag.pt; OBS=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/observations_level3.npz ;;
    ramen) SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen; MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen; NAG=$MODEL/sai_nag.pt; OBS=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz ;;
    teatime) SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/teatime; MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/teatime; NAG=$MODEL/sai_nag.pt; OBS=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/observations_level3.npz ;;
    waldo_kitchen) SOURCE=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen; MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen; NAG=$MODEL/sai_nag.pt; OBS=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz ;;
  esac
}

for scene in figurines ramen teatime waldo_kitchen; do
  case_inputs "$scene"
  mkdir -p "$OUT/$scene"
  { time "$PY" "$DECO_ROOT/deco/lerf_ovs/build_bank_rcs.py" --observations "$OBS" --nag "$NAG" --out "$OUT/$scene/B1_RCS.npz" --b1 1 --b2 0 --geo-weight 0.25 --max-modes 3 --mode-selection rcs; } > "$OUT/$scene/build.log" 2>&1
  sha256sum "$OBS" "$NAG" "$OUT/$scene/B1_RCS.npz" > "$OUT/$scene/INPUT_SHA256.txt"
done

for scene in figurines ramen teatime waldo_kitchen; do
  case_inputs "$scene"
  mkdir -p "$OUT/eval/$scene"
  "$PY" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_v3.py" --scene "$scene" --source "$SOURCE" --model "$MODEL" --nag "$NAG" --gt_root "$GT" --out "$OUT/eval/$scene" --csv "$OUT/eval/$scene/per_query.csv" --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap --evidence-v2 semantic --evidence-bank "$OUT/$scene/B1_RCS.npz" --evidence-observations "$OBS" --evidence-members "$OUT/$scene/B1_RCS.members.json" --evidence-temperature 0.08 --evidence-coefficient 0.25 --evidence-aggregation softmax --tag RCS > "$OUT/eval/$scene/run.log" 2>&1 &
done
wait
echo COMPLETE > "$OUT/COMPLETE"
