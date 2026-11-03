#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/rivermind-data/repro/deco-b123
RUN=$ROOT/query_spatial_repair_20261101
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=$ROOT/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
mkdir -p "$RUN/results" "$RUN/tmp" "$RUN/cache/cuda" "$RUN/cache/torch_extensions"
export TMPDIR="$RUN/tmp" TMP="$RUN/tmp" TEMP="$RUN/tmp"
export XDG_CACHE_HOME="$RUN/cache" CUDA_CACHE_PATH="$RUN/cache/cuda" TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 RELAGS_REPO="$RELAGS"
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export PYTHONPATH="$REPO:$REPO/deco:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
cd "$REPO"
for scene in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
  echo "START $scene $(date -Is)"
  "$PY" -u "$RUN/run_spatial_repair.py" --scene "$scene" \
    --source "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$scene" \
    --model "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$scene" \
    --nag "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$scene/sai_nag.pt" \
    --gt_root /root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label \
    --out "$RUN/results" --preds "$ROOT/lerf_subset_20261030/preds.csv" \
    --evidence "$ROOT/lerf_evidence_20261030" \
    --id_csv "$ROOT/lerf_segidentity_20261030/${scene}_segidentity.csv" \
    ${EXTRA:-} > "$RUN/results/$scene.log" 2>&1
  tail -n 2 "$RUN/results/$scene.log"
done
echo "ALL_DONE $(date -Is)"
