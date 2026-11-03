#!/usr/bin/env bash
set -euo pipefail

: "${DECO_ROOT:?set DECO_ROOT to the isolated DeCo checkout}"
: "${PY:?set PY to the ReLaGS Python runtime}"
: "${SCENE:?set SCENE to ramen or waldo_kitchen}"
: "${SRC:?set SRC to the scene source}"
: "${MODEL:?set MODEL to the Gaussian model}"
: "${NAG:?set NAG to the scene NAG}"
: "${GT_ROOT:?set GT_ROOT to the LERF-OVS labels root}"
: "${K0_BANK:?set K0_BANK}"
: "${B1_BANK:?set B1_BANK}"
: "${B2_BANK:?set B2_BANK}"
: "${OUT:?set OUT to a new output directory}"

EVAL="$DECO_ROOT/deco/lerf_ovs/eval_lerf_ovs_b123.py"
mkdir -p "$OUT"
export PYTHONPATH="$DECO_ROOT:$DECO_ROOT/deco:${RELAGS_REPO:+$RELAGS_REPO/ext:}$RELAGS_REPO/ext/spt/dependencies/grid_graph/python/bin:$RELAGS_REPO/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS_REPO/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"

run_one() {
  local id="$1" center="$2" k0="$3" b1="$4" b2="$5" b3="$6" bank="$7"
  local d="$OUT/$id"
  mkdir -p "$d"
  local args=(--scene "$SCENE" --source "$SRC" --model "$MODEL" --nag "$NAG" --gt_root "$GT_ROOT" --out "$d" --csv "$d/per_query.csv" --center "$center" --control-k1 "$k0" --b1 "$b1" --b2 "$b2" --b3 "$b3" --text canon --renorm 0 --sel gap --topk 5 --level_until 3 --filter_small 50 --evidence-temperature 0.08 --tag "$id")
  if [[ -n "$bank" ]]; then args+=(--evidence-bank "$bank"); fi
  "$PY" "$EVAL" "${args[@]}" 2>&1 | tee "$d/run.log"
}

run_one A0 none 0 0 0 0 ""
run_one A1 image 0 0 0 0 ""
run_one K0_off none 1 0 0 0 "$K0_BANK"
run_one K0_on image 1 0 0 0 "$K0_BANK"
run_one A2 none 0 1 0 0 "$B1_BANK"
run_one A3 image 0 1 0 0 "$B1_BANK"
run_one A4 none 0 1 1 1 "$B2_BANK"
run_one A5 image 0 1 1 1 "$B2_BANK"
