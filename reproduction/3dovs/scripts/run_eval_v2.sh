#!/bin/bash
# v2 canonical eval: 写入 results/v2（全新目录，避免 CSV 追加污染），cand-mode 默认 gap2
set -e
SCENE=$1; ARM=$2; CENTER=$3; TEXT=$4; RENORM=${5:-0}; TOPK=${6:-5}
B=/root/rivermind-data/deco
R=/root/rivermind-data/ReLaGS_Reproduction
V=$R/venv
export PYTHONPATH=$B/DeCo-3DGS/ext:$B/DeCo-3DGS:$B/DeCo-3DGS/deco:$R/ReLaGS/ext/spt/dependencies/grid_graph/python/bin:$R/ReLaGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin
export HF_HUB_OFFLINE=1
cd $B/DeCo-3DGS
mkdir -p $B/results/v2
$V/bin/python deco/3dovs_eval.py \
  --scene $SCENE --source $B/data/3DOVS/$SCENE --model $B/output/3dovs/$SCENE \
  --nag $B/output/3dovs/$SCENE/sai_nag_3.0.pt --gt_root $B/data/3DOVS \
  --out $B/results/v2 --csv $B/results/v2/${SCENE}_${ARM}.csv \
  --center $CENTER --text $TEXT --renorm $RENORM --topk $TOPK \
  --cand-mode ${CANDMODE:-gap2} --tag $ARM
