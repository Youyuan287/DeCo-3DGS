#!/usr/bin/env bash
B=/root/rivermind-data/ReLaGS_Reproduction
PY=$B/venv/bin/python
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines
SC=figurines
cd $B/ReLaGS
export PYTHONPATH="$B/ReLaGS/ext:$B/ReLaGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
SEEDS="0 7 13 42 1234 2024"
FORKS=3
run_one () {
  s=$1; M=$B/work_seed/${SC}_s$s; mkdir -p $M
  ln -sfn $B/ref/relags_hf/$SC/point_cloud $M/point_cloud
  ln -sfn $B/ref/relags_hf/$SC/nag-l1.pt $M/nag-l1.pt
  cat > "$M/cfg_args" <<CFG
Namespace(sh_degree=3, source_path='$SRC', model_path='$M', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
  $PY -u $B/tools/run_seeded.py $s merge_proj_rofa.py -s "$SRC" -m "$M" \
      --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 \
      --rofa_mode raw > "$M/merge.log" 2>&1
  echo "RESULT seed=$s rc=$? | $(grep -o 'ROFA_STATS.*' "$M/merge.log" | tail -1) | $(grep -o 'nag shape.*' "$M/merge.log" | tail -1) | $(grep -o 'Final labels.*' "$M/merge.log" | tail -1)"
}
for s in $SEEDS; do
  run_one $s &
  while [ "$(jobs -rp | wc -l)" -ge "$FORKS" ]; do sleep 5; done
done
wait
echo SWEEP_DONE