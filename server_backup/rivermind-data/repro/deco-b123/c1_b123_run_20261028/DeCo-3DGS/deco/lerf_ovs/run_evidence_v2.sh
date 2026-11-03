#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
RELAGS=/root/rivermind-data/repro/deco-b123/frozen_ramen_relags
GT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label
OUT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/evidence_v2_20261028
export RELAGS_REPO="$RELAGS"
export PYTHONPATH="$ROOT:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin"

run_one() {
  local scene="$1" mode="$2" source model nag bank obs members
  if [[ "$scene" == "ramen" ]]; then
    source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen
    model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen
    nag="$model/sai_nag.pt"
    bank=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.npz
    obs=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz
    members=/root/rivermind-data/repro/deco-b123/b123_eval_official_20261028/B1.members.json
  else
    source=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen
    model=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen
    nag="$model/sai_nag.pt"
    bank=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.npz
    obs=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/observations_level3.npz
    members=/root/rivermind-data/repro/deco-b123/waldo_official_b012_20261028/eval/B1.members.json
  fi
  local dir="$OUT/$scene/$mode"
  mkdir -p "$dir"
  local bank_args=()
  if [[ "$mode" != "off" ]]; then
    bank_args=(--evidence-bank "$bank" --evidence-observations "$obs" --evidence-members "$members")
  fi
  "$PY" "$ROOT/deco/lerf_ovs/eval_lerf_ovs_v2.py" \
    --scene "$scene" --source "$source" --model "$model" --nag "$nag" --gt_root "$GT" \
    --out "$dir" --csv "$dir/per_query.csv" --tag "$mode" \
    --topk 5 --level_until 3 --filter_small 50 --center image --text canon --renorm 0 --sel gap \
    --evidence-v2 "$mode" --evidence-temperature 0.08 "${bank_args[@]}" \
    2>&1 | tee "$dir/run.log"
}

if [[ $# -gt 0 ]]; then
  modes=("$@")
else
  modes=(off semantic reliable)
fi
for scene in ramen waldo_kitchen; do
  for mode in "${modes[@]}"; do
    run_one "$scene" "$mode"
  done
done
