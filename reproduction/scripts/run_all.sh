#!/bin/bash
# ReLaGS LERF-OVS reproduction — runner
# 见 reproduction/README.md 与 reproduction/FINAL_REPORT.md
# Full LERF-OVS ReLaGS reproduction: pipeline + evaluation for all four scenes.
set -euo pipefail
R=/root/rivermind-data/ReLaGS_Reproduction
DATA=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
PY=$R/venv/bin/python
SCENES="figurines ramen teatime waldo_kitchen"
for sc in $SCENES; do
  bash $R/scripts/run_scene.sh $sc
done
cd $R/ReLaGS
: > $R/results/lerf_query_metrics.csv
for sc in $SCENES; do
  NAG=$(ls $R/work/lerf/$sc/sai_nag_*.pt | head -1)
  $PY -u $R/scripts/eval_lerf_relags.py --scene $sc \
    --source $DATA/$sc --model $R/work/lerf/$sc --nag "$NAG" \
    --gt_root $DATA/label --out $R/results \
    --csv $R/results/lerf_query_metrics.csv --save_masks
done
$PY $R/scripts/summarize.py
