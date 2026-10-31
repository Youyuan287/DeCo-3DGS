#!/bin/bash
set -e
SCENE=$1
B=/root/rivermind-data/deco
V=/root/rivermind-data/ReLaGS_Reproduction/venv
export PYTHONPATH=$B/DeCo-3DGS/ext:$B/DeCo-3DGS:/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/ext/spt/dependencies/grid_graph/python/bin:/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin
export HF_HUB_OFFLINE=1
cd $B/DeCo-3DGS
S=$B/data/3DOVS/$SCENE
M=$B/output/3dovs/$SCENE
echo "### 1 max_weight_pruning"
$V/bin/python max_weight_pruning.py -s $S -m $M --contribution_threshold 0.0005
echo "### 2 sp_partition -a"
$V/bin/python sp_partition.py -s $S -m $M --iteration 0 -a
echo "### 3 graph_weight"
$V/bin/python graph_weight.py --iteration 0 -s $S -m $M --config ./configs/3dovs.yml --level 1
echo "### 4 sp_partition -k"
$V/bin/python sp_partition.py -s $S -m $M --iteration 0 -k neighbor_new.pt --pcp_regularization 0.3 --pcp_spatial_weight 0.2 -a
echo "### STAGE14_DONE"
ls -la $M
