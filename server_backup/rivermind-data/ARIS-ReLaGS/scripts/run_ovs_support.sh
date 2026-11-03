#!/usr/bin/env bash
set -euo pipefail
R=/root/rivermind-data
A=$R/ARIS-ReLaGS
O=$A/runs/ovs_support_20261024
test ! -e "$O" || { echo 'Run exists; refusing overwrite'; exit 2; }
python3 - <<'PY'
import shutil,subprocess
for path,reserve in [('/',5),('/root/rivermind-data',10)]:
    assert shutil.disk_usage(path).free>=reserve*2**30, 'Disk reserve violated'
used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()[0])
assert used<500, 'GPU not idle; do not interrupt other workloads'
PY
mkdir -p "$O"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$R/thgs_mamba_root/envs/thgs
export PATH=$CUDA_HOME/bin:$PATH
export RELAGS_REPO=$R/ReLaGS_Reproduction/ReLaGS
export PYTHONPATH=$RELAGS_REPO/ext:$RELAGS_REPO:$R/ReLaGS_Reproduction/ext:$R/ReLaGS_Reproduction
export HF_HUB_OFFLINE=1
PY=$R/ReLaGS_Reproduction/venv/bin/python
cd "$A"
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$O/tests.log"
"$PY" -u scripts/run_ovs_support.py --out "$O/sanity" --sanity 2>&1 | tee "$O/sanity.log"
echo 'SANITY_PASSED' | tee "$O/SANITY_PASSED.txt"
"$PY" -u scripts/run_ovs_support.py --out "$O/full" 2>&1 | tee "$O/full.log"
echo 'ALL_FOUR_GROUPS_COMPLETE' | tee "$O/COMPLETE.txt"
