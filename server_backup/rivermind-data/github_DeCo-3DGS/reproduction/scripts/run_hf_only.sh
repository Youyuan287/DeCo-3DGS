#!/usr/bin/env bash
# ReLaGS LERF-OVS reproduction — runner
# 见 reproduction/README.md 与 reproduction/FINAL_REPORT.md
# Four-scene LERF-OVS evaluation on HuggingFace-only inputs (final protocol).
set -e
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R/ReLaGS"
export PYTHONPATH="$R/ReLaGS/ext:$R/ReLaGS:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
for sc in figurines ramen teatime waldo_kitchen; do
  "$R/venv/bin/python" -u "$R/scripts/eval_lerf_relags.py" \
    --scene "$sc" --source "$R/ref/hf_only/$sc" --model "$R/ref/relags_hf/$sc" \
    --nag "$R/ref/relags_hf/$sc/sai_nag.pt" --gt_root "$R/ref/hf_lerf/lerf_ovs/label" \
    --out "$R/results" --csv "$R/results/lerf_query_metrics.csv" \
    --topk 5 --level_until 3 --filter_small 50 --save_masks
done
