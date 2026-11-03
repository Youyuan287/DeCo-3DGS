#!/bin/bash
B=/root/rivermind-data/deco
want() { echo $(( $(ls $B/data/3DOVS_r4/$1/images 2>/dev/null | wc -l) * 2 )); }
have() { ls $B/data/3DOVS/$1/language_features 2>/dev/null | wc -l; }
enc() { sc=$1
  if [ "$(have $sc)" -eq "$(want $sc)" ] && [ "$(want $sc)" -gt 0 ]; then echo "### enc $sc already done $(date +%H:%M:%S)"; return; fi
  rm -rf $B/data/3DOVS/$sc/language_features $B/data/3DOVS_r4/$sc/language_features
  setsid nohup $B/run_enc.sh $sc > $B/logs/enc_$sc.log 2>&1 &
  until [ "$(have $sc)" -eq "$(want $sc)" ] && [ "$(want $sc)" -gt 0 ]; do sleep 30; done
  echo "### encoded $sc $(date +%H:%M:%S)"; }
waitenc() { until [ "$(have $1)" -eq "$(want $1)" ] && [ "$(want $1)" -gt 0 ]; do sleep 30; done; }
finish() { sc=$1; echo "### finish $sc $(date +%H:%M:%S)"
  if [ -f $B/output/3dovs/$sc/point_cloud/iteration_0/point_cloud.ply ]; then
    $B/run_stage24.sh $sc > $B/logs/stage24_$sc.log 2>&1 || { echo "STAGE24 FAIL $sc"; return 1; }
  else
    $B/run_stage14.sh $sc > $B/logs/stage14_$sc.log 2>&1 || { echo "STAGE14 FAIL $sc"; return 1; }
  fi
  $B/run_stage5.sh $sc > $B/logs/stage5_$sc.log 2>&1 || { echo "MERGE FAIL $sc"; return 1; }
  $B/run_all_arms.sh $sc > $B/logs/arms_$sc.log 2>&1 || { echo "ARMS FAIL $sc"; return 1; }
  echo "### DONE $sc $(date +%H:%M:%S)"; grep SCENE_RESULT $B/logs/arms_$sc.log | tail -4; }
waitenc sofa; finish sofa
finish room
enc bench & enc lawn & wait
finish bench; finish lawn
if [ ! -f $B/output/3dovs/bed/point_cloud/iteration_30000/point_cloud.ply ]; then
  echo "### train bed $(date +%H:%M:%S)"; $B/run_train.sh bed > $B/logs/train_bed.log 2>&1
  rm -rf $B/output/3dovs/bed/point_cloud/iteration_7000
fi
enc bed; finish bed
echo "### SCHED5_DONE $(date +%H:%M:%S)"
