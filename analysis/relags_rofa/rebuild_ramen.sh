#!/usr/bin/env bash
set -uo pipefail
B=/root/rivermind-data/ReLaGS_Reproduction
S=$B/ref/relags_hf/ramen
M=$B/work_rofa/ramen/raw
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/ramen
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
  --rofa_mode raw > "$M/merge2.log" 2>&1
echo "MERGE_RC=$?"
grep -o 'ROFA_STATS.*' "$M/merge2.log" | tail -1
ls -lh "$M/sai_nag_3.0.pt" 2>/dev/null