#!/bin/bash
set -e
SCENE=$1
B=/root/rivermind-data/deco
V=/root/rivermind-data/ReLaGS_Reproduction/venv
export PYTHONPATH=$B/DeCo-3DGS/ext:$B/DeCo-3DGS
export HF_HUB_OFFLINE=1
if [ ! -d $B/data/3DOVS_r4/$SCENE/images ]; then
  $V/bin/python $B/prep_r4.py $SCENE
fi
cd $B/DeCo-3DGS
$V/bin/python scripts/image_encoding.py \
  --source_path $B/data/3DOVS_r4/$SCENE \
  --sam_ckpt_path /root/rivermind-data/ReLaGS_Reproduction/weights/sam_vit_h_4b8939.pth
rm -rf $B/data/3DOVS/$SCENE/language_features
mv $B/data/3DOVS_r4/$SCENE/language_features $B/data/3DOVS/$SCENE/language_features
