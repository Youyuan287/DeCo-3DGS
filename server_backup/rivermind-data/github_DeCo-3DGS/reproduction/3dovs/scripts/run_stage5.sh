#!/bin/bash
set -e
SCENE=$1
B=/root/rivermind-data/deco
R=/root/rivermind-data/ReLaGS_Reproduction
V=$R/venv
export PYTHONPATH=$B/DeCo-3DGS/ext:$B/DeCo-3DGS:$B/DeCo-3DGS/deco:$R/ReLaGS/ext/spt/dependencies/grid_graph/python/bin:$R/ReLaGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/ext/spt/dependencies/grid_graph/python/bin:/root/rivermind-data/ReLaGS_Reproduction/ReLaGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin
export HF_HUB_OFFLINE=1
cd $B/DeCo-3DGS
S=$B/data/3DOVS/$SCENE
M=$B/output/3dovs/$SCENE
echo "### 5 merge_proj $(date +%H:%M:%S)"
$V/bin/python merge_proj.py -s $S -m $M --thres_connect 0.9,0.7,0.7 --thres_merge 100 --tau 3 --iteration 0
echo "### STAGE5_DONE $(date +%H:%M:%S)"
ls -la $M/*.pt
