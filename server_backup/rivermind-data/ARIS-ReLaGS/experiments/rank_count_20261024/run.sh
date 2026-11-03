#!/usr/bin/env bash
set -euo pipefail
R=/root/rivermind-data
A=$R/ARIS-ReLaGS
O=$A/runs/ovs_rank_count_20261024
test ! -e "$O" || { echo 'Refusing existing output'; exit 2; }
python3 - <<'PY'
import shutil,subprocess
for path,reserve in [('/',5),('/root/rivermind-data',10)]:
    assert shutil.disk_usage(path).free>=reserve*2**30
used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().splitlines()[0])
assert used<500, 'GPU occupied; do not interrupt'
PY
mkdir -p "$O"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$R/thgs_mamba_root/envs/thgs
export PATH=$CUDA_HOME/bin:$PATH
export RELAGS_REPO=$R/ReLaGS_Reproduction/ReLaGS
export PYTHONPATH=$RELAGS_REPO/ext:$RELAGS_REPO:$R/ReLaGS_Reproduction/ext:$R/ReLaGS_Reproduction:$A/scripts
export HF_HUB_OFFLINE=1
PY=$R/ReLaGS_Reproduction/venv/bin/python
cd "$A"
"$PY" -m unittest discover -s tests -v 2>&1 | tee "$O/existing_tests.log"
"$PY" -m unittest discover -s experiments/rank_count_20261024 -v 2>&1 | tee "$O/new_tests.log"
"$PY" -u experiments/rank_count_20261024/run_rank_count.py --prior "$A/runs/ovs_support_20261024/full" --out "$O/full" 2>&1 | tee "$O/run.log"
echo 'TWO_MISSING_CELLS_COMPLETE' | tee "$O/COMPLETE.txt"
