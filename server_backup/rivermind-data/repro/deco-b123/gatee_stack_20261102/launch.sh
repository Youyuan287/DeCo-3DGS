#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/rivermind-data/repro/deco-b123
RUN=$ROOT/gatee_stack_20261102
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=$ROOT/c1_b123_run_20261028/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
MODE=${MODE:-full}
case "$MODE" in
  sanity) SCENES=(figurines); EXTRA=(--limit-queries 1); RESULTS="$RUN/sanity" ;;
  full) SCENES=(figurines ramen teatime waldo_kitchen); EXTRA=(); RESULTS="$RUN/results" ;;
  *) exit 2 ;;
esac
mkdir -p "$RESULTS" "$RUN/tmp" "$RUN/cache/cuda" "$RUN/cache/torch_extensions" "$RUN/cache/huggingface/hub" "$RUN/cache/torch"
exec > "$RUN/${MODE}_driver.log" 2>&1
export TMPDIR="$RUN/tmp" TMP="$RUN/tmp" TEMP="$RUN/tmp"
export XDG_CACHE_HOME="$RUN/cache" CUDA_CACHE_PATH="$RUN/cache/cuda"
export TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions" TORCH_HOME="$RUN/cache/torch"
export HF_HOME="$RUN/cache/huggingface" HF_HUB_CACHE="$RUN/cache/huggingface/hub"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 RELAGS_REPO="$RELAGS"
export PYTHONPATH="$REPO:$REPO/deco:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
cd "$REPO"
for scene in "${SCENES[@]}"; do
  echo "START $scene $(date -Is)"
  "$PY" -u "$RUN/render_predictions.py" --scene "$scene" \
    --source "/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$scene" \
    --model "/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf/$scene" \
    --gt_root /root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label \
    --predictions "$RUN/predictions.jsonl" --out "$RESULTS" "${EXTRA[@]}" \
    > "$RESULTS/$scene.log" 2>&1
  tail -n 2 "$RESULTS/$scene.log"
done
echo "ALL_DONE $(date -Is)"
