#!/usr/bin/env bash
set -euo pipefail

DECO_ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS_REPO=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
OUT=$ROOT/evidence_v2_lambda_tau_20261029
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label

export HF_HUB_OFFLINE=1
export PYTHONPATH=$DECO_ROOT:$RELAGS_REPO:$RELAGS_REPO/ext
mkdir -p "$OUT"

run_one() {
  local tag="$1" scene="$2" lam="$3" tau="$4" source="$5" model="$6" nag="$7" bank="$8" obs="$9" members="${10}"
  local d="$OUT/$tag/$scene"
  mkdir -p "$d"
  "$PY" "$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_v3.py" \
    --scene "$scene" --source "$source" --model "$model" --nag "$nag" --gt_root "$GT" \
    --out "$d" --csv "$d/per_query.csv" --topk 5 --level_until 3 --filter_small 50 \
    --center image --text canon --renorm 0 --sel gap --evidence-v2 semantic \
    --evidence-bank "$bank" --evidence-observations "$obs" --evidence-members "$members" \
    --evidence-temperature "$tau" --evidence-coefficient "$lam" --tag "$tag" \
    > "$d/run.log" 2>&1
}

for lam in 0 0.125 0.25 0.5 1.0; do
  for tau in 0.04 0.08 0.16; do
    if [[ "$lam" == "0" && "$tau" != "0.04" ]]; then
      continue
    fi
    tag="lambda_${lam}_tau_${tau}"
    if [[ -f "$OUT/$tag/figurines/figurines_${tag}_summary.json" && -f "$OUT/$tag/ramen/ramen_${tag}_summary.json" && -f "$OUT/$tag/teatime/teatime_${tag}_summary.json" && -f "$OUT/$tag/waldo_kitchen/waldo_kitchen_${tag}_summary.json" ]]; then
      continue
    fi
    pids=()
    for scene in figurines ramen teatime waldo_kitchen; do
      case "$scene" in
        figurines)
          source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines
          model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/figurines
          nag=$model/sai_nag.pt
          bank=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/B1.npz
          obs=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/observations_level3.npz
          members=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/B1.members.json ;;
        ramen)
          source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen
          model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
          nag=$model/sai_nag.pt
          bank=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.npz
          obs=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz
          members=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.members.json ;;
        teatime)
          source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/teatime
          model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/teatime
          nag=$model/sai_nag.pt
          bank=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/B1.npz
          obs=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/observations_level3.npz
          members=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/B1.members.json ;;
        waldo_kitchen)
          source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen
          model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen
          nag=$model/sai_nag.pt
          bank=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.npz
          obs=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz
          members=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.members.json ;;
      esac
      test -f "$bank" -a -f "$obs" -a -f "$members"
      run_one "$tag" "$scene" "$lam" "$tau" "$source" "$model" "$nag" "$bank" "$obs" "$members" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
done

echo COMPLETE > "$OUT/COMPLETE"
