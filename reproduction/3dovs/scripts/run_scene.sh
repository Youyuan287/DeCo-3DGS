#!/bin/bash
set -e
SCENE=$1
B=/root/rivermind-data/deco
echo "########## SCENE $SCENE START $(date +%H:%M:%S)"
[ -f $B/data/3DOVS/$SCENE/sparse/0/cameras.bin.orig ] || \
  /root/rivermind-data/deco/venv/bin/python $B/to_pinhole.py $B/data/3DOVS/$SCENE/sparse/0/cameras.bin
if [ ! -f $B/output/3dovs/$SCENE/point_cloud/iteration_30000/point_cloud.ply ]; then
  echo "--- train $SCENE $(date +%H:%M:%S)"
  $B/run_train.sh $SCENE > $B/logs/train_$SCENE.log 2>&1
fi
rm -rf $B/output/3dovs/$SCENE/point_cloud/iteration_7000
echo "--- encode $SCENE $(date +%H:%M:%S)"
$B/run_enc.sh $SCENE > $B/logs/enc_$SCENE.log 2>&1
echo "--- stages1-4 $SCENE $(date +%H:%M:%S)"
$B/run_stage14.sh $SCENE > $B/logs/stage14_$SCENE.log 2>&1
echo "--- merge $SCENE $(date +%H:%M:%S)"
$B/run_stage5.sh $SCENE > $B/logs/stage5_$SCENE.log 2>&1
echo "--- arms $SCENE $(date +%H:%M:%S)"
$B/run_all_arms.sh $SCENE > $B/logs/arms_$SCENE.log 2>&1
echo "########## SCENE $SCENE DONE $(date +%H:%M:%S)"
