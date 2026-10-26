#!/bin/bash
# LERF-Mask (Gaussian Grouping protocol), 3 scenes, 61 queries.
#
#   export RELAGS_REPO=/path/to/ReLaGS            # contains scene/, gaussian_renderer/, nag_data.py
#   export RIVERMIND_ROOT=/path/to/artefacts      # contains lerf_oseg/output/<scene>/sai_nag_3.0.pt
#   bash scripts/run_lerf_mask.sh none            # baseline
#   bash scripts/run_lerf_mask.sh image           # + C1
set -eu
CENTER="${1:-image}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RELAGS_REPO:?set RELAGS_REPO}"
: "${RIVERMIND_ROOT:?set RIVERMIND_ROOT}"
PY="${PYTHON:-python3}"

export PYTHONPATH="$RELAGS_REPO:$RELAGS_REPO/ext:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

for sc in figurines ramen teatime; do
  "$PY" "$HERE/lerf_mask/eval_lerf_mask.py" \
    --scene "$sc" \
    --eval_source "$RIVERMIND_ROOT/lerf_oseg/data/$sc" \
    --model "$RIVERMIND_ROOT/lerf_oseg/output/$sc" \
    --nag "$RIVERMIND_ROOT/lerf_oseg/output/$sc/sai_nag_3.0.pt" \
    --gt_root "$RIVERMIND_ROOT/lerf_mask/gt" \
    --out "$HERE/out/lerf_mask_$CENTER" \
    --csv "$HERE/out/lerf_mask_$CENTER/rows.csv" \
    --center "$CENTER"
done