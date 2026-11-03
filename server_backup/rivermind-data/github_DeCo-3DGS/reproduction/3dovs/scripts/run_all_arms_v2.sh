#!/bin/bash
# 7 arms: base / c1(renorm0, 上游默认) / c1r(renorm1, DeCo 有效版) / c2 / c1c2 / c1rc2 / base_t10
SCENE=$1
B=/root/rivermind-data/deco
$B/run_eval_v2.sh $SCENE base    none  canon    0 5
$B/run_eval_v2.sh $SCENE c1      image canon    0 5
$B/run_eval_v2.sh $SCENE c1r     image canon    1 5
$B/run_eval_v2.sh $SCENE c2      none  template 0 5
$B/run_eval_v2.sh $SCENE c1c2    image template 0 5
$B/run_eval_v2.sh $SCENE c1rc2   image template 1 5
$B/run_eval_v2.sh $SCENE base_t10 none canon   0 10
echo "### ARMS_V2_DONE $SCENE"
