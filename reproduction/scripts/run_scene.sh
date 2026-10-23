#!/bin/bash
# ReLaGS LERF-OVS reproduction — runner
# 见 reproduction/README.md 与 reproduction/FINAL_REPORT.md
set -euo pipefail
SC=$1
R=/root/rivermind-data/ReLaGS_Reproduction
RE=$R/ReLaGS
DATA=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
SRC=$DATA/$SC
M=$R/work/lerf/$SC
LOG=$R/logs/pipeline_$SC.log
PY=$R/venv/bin/python
mkdir -p $M/point_cloud/iteration_30000
if [ ! -f $M/point_cloud/iteration_30000/point_cloud.ply ]; then
  cp /root/rivermind-data/THGS/output/lerf/$SC/point_cloud/iteration_30000/point_cloud.ply $M/point_cloud/iteration_30000/point_cloud.ply
fi
cat > $M/cfg_args <<CFG
Namespace(sh_degree=3, source_path='$SRC', model_path='$M', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
cd $RE
echo "=== [$(date -Is)] $SC MWP ===" >> $LOG
$PY -u max_weight_pruning.py -s $SRC -m $M --contribution_threshold 0.005 >> $LOG 2>&1
echo "=== [$(date -Is)] $SC sp_partition pass1 ===" >> $LOG
$PY -u sp_partition.py -m $M -i 0 -a >> $LOG 2>&1
echo "=== [$(date -Is)] $SC graph_weight ===" >> $LOG
$PY -u graph_weight.py --iteration 0 -s $SRC -m $M --config $R/configs/lerf_local.yml --level 1 >> $LOG 2>&1
echo "=== [$(date -Is)] $SC sp_partition pass2 ===" >> $LOG
$PY -u sp_partition.py -m $M -i 0 -k neighbor_new.pt --pcp_regularization 0.1 --pcp_spatial_weight 0.1 -a >> $LOG 2>&1
echo "=== [$(date -Is)] $SC merge_proj ===" >> $LOG
$PY -u merge_proj.py -s $SRC -m $M --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 >> $LOG 2>&1
echo "=== [$(date -Is)] $SC DONE ===" >> $LOG
