#!/usr/bin/env bash
set -euo pipefail
cd /root/rivermind-data/ARIS-ReLaGS/experiments/view_consensus_20261024
out=/root/rivermind-data/ARIS-ReLaGS/runs/view_consensus_capture_20261025
test ! -e "$out/capture.log"
sha256sum capture_fixed.py capture_contracts.py > "$out/launch_source_sha256.txt"
printf '%s\n' 'CUDA_VISIBLE_DEVICES=0 python capture_fixed.py capture; timeout=15min inside capture' > "$out/command.txt"
printf '%s\n' 'CUDA_VISIBLE_DEVICES=0' 'HF_HUB_OFFLINE=1' > "$out/environment.txt"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
set +e
/root/rivermind-data/thgs_mamba_root/envs/thgs/bin/python -u capture_fixed.py capture > "$out/capture.log" 2>&1
status=$?
printf '%s\n' "$status" > "$out/capture_exit.txt"
exit "$status"
