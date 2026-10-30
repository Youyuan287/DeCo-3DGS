#!/usr/bin/env bash
# Controlled A/B of ROFA modes inside ReLaGS (PREREG_relags_rofa.md).
set -uo pipefail
B=/root/rivermind-data/ReLaGS_Reproduction
PY=$B/venv/bin/python
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
GT=$B/ref/hf_lerf/lerf_ovs/label
OUT=$B/results_rofa
mkdir -p "$OUT"
cd "$B/ReLaGS"
export PYTHONPATH="$B/ReLaGS/ext:$B/ReLaGS${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0

for mode in ${MODES:-raw center off}; do
  csv="$OUT/${mode}_query_metrics.csv"; rm -f "$csv"
  mkdir -p "$OUT/$mode"
  for sc in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
    M=$B/work_rofa/$sc/$mode
    mkdir -p "$M"
    ln -sfn "$B/ref/relags_hf/$sc/point_cloud" "$M/point_cloud"
    ln -sfn "$B/ref/relags_hf/$sc/nag-l1.pt" "$M/nag-l1.pt"
    cat > "$M/cfg_args" <<CFG
Namespace(sh_degree=3, source_path='$SRCROOT/$sc', model_path='$M', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
    echo "=== [$(date -Is)] $sc $mode merge ==="
    "$PY" -u merge_proj_rofa.py -s "$SRCROOT/$sc" -m "$M" \
      --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 \
      --rofa_mode "$mode" > "$M/merge.log" 2>&1
    echo "MERGE $sc $mode rc=$? $(grep -o 'ROFA_STATS.*' "$M/merge.log" | tail -1)"
    rm -f "$M"/*saifine.ply "$M"/saifine_labels.npy
    echo "=== [$(date -Is)] $sc $mode eval ==="
    "$PY" -u "$B/scripts/eval_lerf_relags.py" --scene "$sc" \
      --source "$B/ref/hf_only/$sc" --model "$M" --nag "$M/sai_nag_3.0.pt" \
      --gt_root "$GT" --out "$OUT/$mode" --csv "$csv" \
      --topk 5 --level_until 3 --filter_small 50 > "$M/eval.log" 2>&1
    echo "EVAL $sc $mode rc=$? $(grep -o 'SCENE_RESULT.*' "$M/eval.log" | tail -1)"
  done
done
echo ALL_DONE