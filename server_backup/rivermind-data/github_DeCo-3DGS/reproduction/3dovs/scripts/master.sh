#!/bin/bash
B=/root/rivermind-data/deco
# wait for sofa's own chain (nag + 4 arms) to finish first
while [ ! -f $B/output/3dovs/sofa/sai_nag_3.0.pt ]; do sleep 30; done
while ! grep -q "ARMS_DONE sofa" $B/logs/arms_sofa.log 2>/dev/null; do sleep 30; done
for sc in room bench lawn bed; do
  $B/run_scene.sh $sc || echo "SCENE $sc FAILED"
done
echo "### MASTER_DONE"
