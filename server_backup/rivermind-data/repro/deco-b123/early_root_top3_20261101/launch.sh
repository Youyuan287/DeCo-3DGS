#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/rivermind-data/repro/deco-b123
RUN=$ROOT/early_root_top3_20261101
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=$ROOT/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
MODE=${MODE:-full}
RESULTS="$RUN/results"
if [ "$MODE" = sanity ]; then
  SCENES=figurines
  EXTRA="--limit-queries 1"
  RESULTS="$RUN/sanity"
fi
mkdir -p "$RESULTS" "$RUN/tmp" "$RUN/cache/cuda" "$RUN/cache/torch_extensions"
exec > "$RUN/${MODE}_driver.log" 2>&1
export TMPDIR="$RUN/tmp" TMP="$RUN/tmp" TEMP="$RUN/tmp"
export XDG_CACHE_HOME="$RUN/cache" CUDA_CACHE_PATH="$RUN/cache/cuda" TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 RELAGS_REPO="$RELAGS"
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export PYTHONPATH="$REPO:$REPO/deco:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$REPO"
for scene in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
  sas="$ROOT/m1_sas_20261029/${scene}_new"
  if [ "$scene" = waldo_kitchen ]; then sas="$ROOT/m1_sas_20261029/waldo_new"; fi
  echo "START $scene $(date -Is)"
  "$PY" -u "$RUN/run_early_root.py" --scene "$scene" \
    --source "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$scene" \
    --model "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$scene" \
    --nag "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$scene/sai_nag.pt" \
    --gt_root /root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label \
    --sas "$sas" --out "$RESULTS" ${EXTRA:-} > "$RESULTS/$scene.log" 2>&1
  tail -n 2 "$RESULTS/$scene.log"
done
echo "ALL_DONE $(date -Is)"
