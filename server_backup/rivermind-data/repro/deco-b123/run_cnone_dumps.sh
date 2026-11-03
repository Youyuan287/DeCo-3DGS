#!/usr/bin/env bash
set -uo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODELROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GTROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
SAS=/root/rivermind-data/repro/deco-b123/m1_sas_20261029
BASE=/root/rivermind-data/repro/deco-b123/cnone_20261101
mkdir -p "$BASE"/{cand,ev,segid}
cd "$REPO"
export RELAGS_REPO="$RELAGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
for scene in figurines ramen teatime waldo_kitchen; do
  b="$SAS/${scene}_new"; [ -d "$b" ] || b="$SAS/waldo_new"
  echo "START cand $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_candidate_dump_cnone.py --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --nag "$MODELROOT/$scene/sai_nag.pt" --gt_root "$GTROOT" --out "$BASE/cand" > "$BASE/cand/$scene.log" 2>&1
  echo "END cand $scene rc=$? $(date -Is)"
  echo "START ev $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_evidence_dump_cnone.py --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --nag "$MODELROOT/$scene/sai_nag.pt" --gt_root "$GTROOT" --sas "$b" --out "$BASE/ev" > "$BASE/ev/$scene.log" 2>&1
  echo "END ev $scene rc=$? $(date -Is)"
  echo "START segid $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_segment_identity_dump_cnone.py --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --nag "$MODELROOT/$scene/sai_nag.pt" --gt_root "$GTROOT" --sas "$b" --out "$BASE/segid" > "$BASE/segid/$scene.log" 2>&1
  echo "END segid $scene rc=$? $(date -Is)"
done
echo ALL_DONE
