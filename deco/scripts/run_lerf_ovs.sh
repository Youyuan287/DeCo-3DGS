#!/bin/bash
# LERF-OVS, 4 scenes, 208 (image, query) samples.
#
#   export RELAGS_REPO=/path/to/ReLaGS
#   export RIVERMIND_ROOT=/path/to/artefacts      # ref/hf_only/<scene>, ref/hf_lerf/lerf_ovs/label
#   bash scripts/run_lerf_ovs.sh none canon       # baseline (reproduces 64.01)
#   bash scripts/run_lerf_ovs.sh image canon      # + C1 on the retrieval axis (inert: -0.08)
set -eu
CENTER="${1:-image}"
TEXT="${2:-canon}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RELAGS_REPO:?set RELAGS_REPO}"
: "${RIVERMIND_ROOT:?set RIVERMIND_ROOT}"
PY="${PYTHON:-python3}"

export PYTHONPATH="$RELAGS_REPO:$RELAGS_REPO/ext:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

OUT="$HERE/out/lerf_ovs_${CENTER}_${TEXT}"
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
    --center "$CENTER" --text "$TEXT" --lexicon "$HERE/lerf_ovs/lexicon.json"
done