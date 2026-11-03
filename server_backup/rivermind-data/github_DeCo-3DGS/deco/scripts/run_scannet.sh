#!/bin/bash
# ScanNet-10: score the ten certified scenes, then aggregate to dataset level.
#
#   export RIVERMIND_ROOT=/path/to/artefacts   # contains output/scannet/<scene>/sai_nag_3.0.pt
#   bash scripts/run_scannet.sh          # unified operator (--renorm 0) -> 58.55
#   bash scripts/run_scannet.sh 1        # renormalised variant          -> 57.85
set -eu
RENORM="${1:-0}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RIVERMIND_ROOT:?set RIVERMIND_ROOT to the directory holding output/scannet/}"
PY="${PYTHON:-python3}"

# scorer and aggregator must agree on one directory; both default to $SCANNET_OUT
export SCANNET_OUT="${SCANNET_OUT:-$HERE/out/scannet_k19}"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

"$PY" "$HERE/scannet/score_scannet.py" --renorm "$RENORM"
"$PY" "$HERE/scannet/aggregate.py" --results "$SCANNET_OUT" --verbose