#!/bin/bash
# LERF-OVS, 4 scenes, 208 (image, query) samples.
#
#   export RELAGS_REPO=/path/to/ReLaGS
#   export RIVERMIND_ROOT=/path/to/artefacts      # ref/hf_only/<scene>, ref/relags_hf/<scene>, ref/hf_lerf/label
#   bash scripts/run_lerf_ovs.sh none  canon 0    # baseline, reproduces 64.01
#   bash scripts/run_lerf_ovs.sh image canon 0    # + C1: 0.6393 (constant shift, 173/208 identical)
#   bash scripts/run_lerf_ovs.sh image canon 1    # WRONG BRANCH: 0.4686, renormalised
#
# args: <center: none|image> <text: canon|template|lexicon|lex_tpl|dot> <renorm: 0|1>
set -eu
CENTER="${1:-image}"
TEXT="${2:-canon}"
RENORM="${3:-0}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RELAGS_REPO:?set RELAGS_REPO}"
: "${RIVERMIND_ROOT:?set RIVERMIND_ROOT}"
PY="${PYTHON:-python3}"

export PYTHONPATH="$HERE:$RELAGS_REPO:$RELAGS_REPO/ext:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

OUT="$HERE/out/lerf_ovs_${CENTER}_${TEXT}_r${RENORM}"
mkdir -p "$OUT"
for sc in figurines ramen teatime waldo_kitchen; do
  "$PY" "$HERE/lerf_ovs/eval_lerf_ovs.py" \
    --scene "$sc" \
    --source "$RIVERMIND_ROOT/ref/hf_only/$sc" \
    --model "$RIVERMIND_ROOT/ref/relags_hf/$sc" \
    --nag "$RIVERMIND_ROOT/ref/relags_hf/$sc/sai_nag.pt" \
    --gt_root "$RIVERMIND_ROOT/ref/hf_lerf/lerf_ovs/label" \
    --out "$OUT" --csv "$OUT/all.csv" \
    --topk 5 --level_until 3 --filter_small 50 \
    --center "$CENTER" --text "$TEXT" --renorm "$RENORM" \
    --lexicon "$HERE/lerf_ovs/lexicon.json"
done