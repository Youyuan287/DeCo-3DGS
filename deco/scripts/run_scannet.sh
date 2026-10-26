#!/bin/bash
# ScanNet-10: score the ten certified scenes, then aggregate to dataset level.
#
#   export RIVERMIND_ROOT=/path/to/artefacts   # contains output/scannet/<scene>/sai_nag_3.0.pt
#   bash scripts/run_scannet.sh
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${RIVERMIND_ROOT:?set RIVERMIND_ROOT to the directory holding output/scannet/}"
PY="${PYTHON:-python3}"

export PYTHONPATH="$HERE/scannet:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

"$PY" "$HERE/scannet/score_scannet.py" --emit
"$PY" "$HERE/scannet/aggregate.py"