#!/usr/bin/env bash
set -euo pipefail
R=/root/rivermind-data
A=$R/ARIS-ReLaGS
O=$A/runs/spatial_support_20261024
mkdir -p "$O"
test ! -e "$O/baseline" || { echo 'baseline output exists; refusing overwrite'; exit 2; }
df -Pk / "$R"
python3 - <<'PY'
import shutil
for path,gib in [('/',5),('/root/rivermind-data',10)]:
    if shutil.disk_usage(path).free < gib*2**30:
        raise SystemExit('Insufficient reserve: '+path)
PY
mkdir "$O/baseline"
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$R/thgs_mamba_root/envs/thgs
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=$R/ReLaGS_Reproduction/ReLaGS/ext:$R/ReLaGS_Reproduction/ReLaGS:$R/ReLaGS_Reproduction/ext:$R/ReLaGS_Reproduction
export HF_HUB_OFFLINE=1
export RELAGS_REPO=$R/ReLaGS_Reproduction/ReLaGS
sha256sum "$R/output/lerf_mask/figurines/sai_nag_3.0.pt" "$R/output/lerf_mask/figurines/point_cloud/iteration_0/point_cloud.ply" "$R/lerf_mask_repro/scripts/eval_lerf_mask.py" "$RELAGS_REPO/nag_data.py" "$RELAGS_REPO/utils/vlm_utils.py" "$R/datasets/lerf_mask/build/figurines_eval/split.json" > "$O/baseline/input_sha256.txt"
cmd=("$R/ReLaGS_Reproduction/venv/bin/python" -u "$R/lerf_mask_repro/scripts/eval_lerf_mask.py" --scene figurines --eval-source "$R/datasets/lerf_mask/build/figurines_eval" --model "$R/output/lerf_mask/figurines" --nag "$R/output/lerf_mask/figurines/sai_nag_3.0.pt" --gt-root "$R/datasets/lerf_mask" --out "$O/baseline/pred" --csv "$O/baseline/query_metrics.csv" --topk 5 --level-until 3 --filter-small 50)
printf '%q ' "${cmd[@]}" > "$O/baseline/command.txt"
printf '\n' >> "$O/baseline/command.txt"
"${cmd[@]}" 2>&1 | tee "$O/baseline/run.log"
printf 'baseline replay complete\n' | tee "$O/baseline/COMPLETE.txt"
