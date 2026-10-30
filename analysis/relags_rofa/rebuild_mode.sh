#!/usr/bin/env bash
set -uo pipefail
B=/root/rivermind-data/ReLaGS_Reproduction
SC=${1:-ramen}
MODE=${2:-center}
S=$B/ref/relags_hf/$SC
M=$B/work_rofa/$SC/mp_$MODE
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$SC
mkdir -p "$M"
ln -sfn "$S/point_cloud" "$M/point_cloud"
ln -sfn "$S/nag-l1.pt" "$M/nag-l1.pt"
cat > "$M/cfg_args" <<CFG
Namespace(sh_degree=3, source_path='$SRC', model_path='$M', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
cd "$B/ReLaGS"
export PYTHONPATH="$B/ReLaGS/ext:$B/ReLaGS" HF_HUB_OFFLINE=1
"$B/venv/bin/python3" -u merge_proj_rofa.py -s "$SRC" -m "$M" \
  --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 \
  --rofa_mode "$MODE" > "$M/merge.log" 2>&1
echo "MERGE $SC $MODE rc=$? $(grep -o 'ROFA_STATS.*' "$M/merge.log" | tail -1)"
rm -f "$M"/*saifine.ply "$M"/saifine_labels.npy
D=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
O=$B/results_rofa/mp_${SC}_${MODE}; mkdir -p "$O"
export PYTHONPATH="$D:$B/ReLaGS:$B/ReLaGS/ext"
"$B/venv/bin/python3" "$D/deco/lerf_ovs/eval_lerf_ovs.py" --scene "$SC" \
  --source "$SRC" --model "$M" --nag "$M/sai_nag_3.0.pt" \
  --gt_root "$B/ref/hf_lerf/lerf_ovs/label" --out "$O" --csv "$O/q.csv" \
  --center none --renorm 0 > "$O/eval.log" 2>&1
echo "EVAL $SC $MODE $(grep -o 'SCENE_RESULT.*' "$O/eval.log" | tail -1)"