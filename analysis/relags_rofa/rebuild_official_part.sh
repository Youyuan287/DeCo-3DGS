#!/usr/bin/env bash
set -uo pipefail
B=/root/rivermind-data/ReLaGS_Reproduction
SC=${1:-ramen}
MODE=${2:-raw}
S=$B/ref/relags_hf/$SC
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$SC
M=$B/work_official_part/$SC/$MODE
mkdir -p "$M"
ln -sfn "$S/point_cloud" "$M/point_cloud"
# nag-l1.pt here is the official level-0 partition extracted from sai_nag.pt
[ -f "$M/nag-l1.pt" ] || echo "MISSING official partition $M/nag-l1.pt" >&2
cat > "$M/cfg_args" <<CFG
Namespace(sh_degree=3, source_path='$SRC', model_path='$M', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
cd "$B/ReLaGS"
export PYTHONPATH="$B/ReLaGS/ext:$B/ReLaGS" HF_HUB_OFFLINE=1
"$B/venv/bin/python3" -u merge_proj_rofa.py -s "$SRC" -m "$M" \
  --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 \
  --rofa_mode "$MODE" > "$M/merge.log" 2>&1
echo "MERGE $SC $MODE rc=$? $(grep -o 'ROFA_STATS.*' "$M/merge.log" | tail -1)"
# compare level-0 features with official
"$B/venv/bin/python3" - "$B" "$SC" "$M" <<'PY'
import sys, torch
B, SC, M = sys.argv[1], sys.argv[2], sys.argv[3]
o = torch.load(f"{B}/ref/relags_hf/{SC}/sai_nag.pt", map_location="cpu")
r = torch.load(f"{M}/sai_nag_3.0.pt", map_location="cpu")
fo = torch.nn.functional.normalize(o["nag_feat"][0], dim=-1)
fr = torch.nn.functional.normalize(r["nag_feat"][0], dim=-1)
n = min(len(fo), len(fr)); c = (fo[:n] * fr[:n]).sum(1)
print("L0 clusters official %d rebuild %d" % (len(fo), len(fr)))
print("L0 feat cos mean %.4f median %.4f p5 %.4f" % (c.mean(), c.median(), torch.quantile(c, 0.05)))
PY
rm -f "$M"/*saifine.ply "$M"/saifine_labels.npy
D=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
O=$B/results_rofa/op_${SC}_${MODE}; mkdir -p "$O"
export PYTHONPATH="$D:$B/ReLaGS:$B/ReLaGS/ext"
"$B/venv/bin/python3" "$D/deco/lerf_ovs/eval_lerf_ovs.py" --scene "$SC" \
  --source "$SRC" --model "$M" --nag "$M/sai_nag_3.0.pt" \
  --gt_root "$B/ref/hf_lerf/lerf_ovs/label" --out "$O" --csv "$O/q.csv" \
  --center none --renorm 0 --text canon > "$O/eval.log" 2>&1
echo "EVAL $SC $MODE $(grep -o 'SCENE_RESULT.*' "$O/eval.log" | tail -1)"