#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
CASES=$ROOT/deco/analysis/SELECTED_M1_CASES.csv
GT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/label
BASE=$ROOT/deco/analysis/results/m1_m2_m3_theory_validation/CASE_FIGURES_CORRECTED
mkdir -p "$BASE/ramen" "$BASE/waldo_kitchen"
cd "$ROOT"
export PYTHONPATH="$ROOT"
"$PY" deco/analysis/generate_m1_case_figures.py \
  --scene ramen \
  --source /root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen \
  --model /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen \
  --nag /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/ramen/sai_nag_3.0.pt \
  --gt-root "$GT" \
  --support "$ROOT/deco/analysis/results/m1_m2_m3_theory_validation/ramen" \
  --observations /root/rivermind-data/repro/deco-b123/frozen_ramen_export_20261028/observations_level3.npz \
  --members /root/rivermind-data/repro/deco-b123/b123_eval_20261028/B2.members.json \
  --cases "$CASES" \
  --out "$BASE/ramen"
"$PY" deco/analysis/generate_m1_case_figures.py \
  --scene waldo_kitchen \
  --source /root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/waldo_kitchen \
  --model /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen \
  --nag /root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/waldo_kitchen/sai_nag_3.0.pt \
  --gt-root "$GT" \
  --support "$ROOT/deco/analysis/results/m1_m2_m3_theory_validation/waldo_kitchen" \
  --observations /root/rivermind-data/repro/deco-b123/waldo_b123_20261028/observations_level3.npz \
  --members /root/rivermind-data/repro/deco-b123/waldo_b123_20261028/eval/B2.members.json \
  --cases "$CASES" \
  --out "$BASE/waldo_kitchen"
