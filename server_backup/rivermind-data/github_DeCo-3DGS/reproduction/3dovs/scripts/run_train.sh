#!/bin/bash
set -e
SCENE=$1
B=/root/rivermind-data/deco
V=$B/venv
export TMPDIR=$B/tmp
mkdir -p $TMPDIR
export PYTHONPATH=$B/2dgs/submodules/simple-knn
cd $B/2dgs
$V/bin/python train.py \
  -s $B/data/3DOVS/$SCENE \
  -m $B/output/3dovs/$SCENE \
  --iterations 30000 -r 4
