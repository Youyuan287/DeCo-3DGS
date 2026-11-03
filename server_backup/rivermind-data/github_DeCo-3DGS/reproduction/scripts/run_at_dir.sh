#!/bin/bash
# ReLaGS LERF-OVS reproduction — runner
# 见 reproduction/README.md 与 reproduction/FINAL_REPORT.md
set -euo pipefail
SC=$1; M=$2; TAG=$3
R=/root/rivermind-data/ReLaGS_Reproduction
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$SC
LOG=$R/logs/${TAG}_$SC.log
PY=$R/venv/bin/python
cd $R/ReLaGS
echo "=== [$(date -Is)] $SC $TAG sp1 ===" >> $LOG
$PY -u sp_partition.py -m $M -i 0 -a >> $LOG 2>&1
echo "=== [$(date -Is)] $SC $TAG gw ===" >> $LOG
$PY -u graph_weight.py --iteration 0 -s $SRC -m $M --config $R/configs/lerf_local.yml --level 1 >> $LOG 2>&1
echo "=== [$(date -Is)] $SC $TAG sp2 ===" >> $LOG
$PY -u sp_partition.py -m $M -i 0 -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a >> $LOG 2>&1
echo "=== [$(date -Is)] $SC $TAG merge ===" >> $LOG
$PY -u merge_proj.py -s $SRC -m $M --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 >> $LOG 2>&1
echo "=== [$(date -Is)] $SC $TAG DONE ===" >> $LOG
