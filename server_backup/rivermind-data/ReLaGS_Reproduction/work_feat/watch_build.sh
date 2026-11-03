#!/bin/bash
B=/root/rivermind-data/ReLaGS_Reproduction
PY=$B/venv/bin/python
D=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
ORIG=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/figurines
GT=$B/ref/hf_lerf/lerf_ovs/label
FEAT=$B/work_feat/figurines_relags
LOG=$B/work_feat/WATCH.log
echo "[$(date +%H:%M)] watcher start" >> $LOG
while true; do
  N=$(ls $FEAT/*_f.npy 2>/dev/null | wc -l)
  M=$(ls $FEAT/*_s.npy 2>/dev/null | wc -l)
  if [ "$N" -ge 299 ] && [ "$M" -ge 299 ]; then echo "[$(date +%H:%M)] encode complete f=$N s=$M" >> $LOG; break; fi
  sleep 120
done
# 1) 新源码目录（复用原 images/sparse，只换 language_features）
SR=$B/src_fig_relags
mkdir -p $SR
ln -sfn $ORIG/images $SR/images
ln -sfn $ORIG/sparse $SR/sparse
ln -sfn $ORIG/poses_bounds.npy $SR/poses_bounds.npy 2>/dev/null
ln -sfn $FEAT $SR/language_features
# 2) 模型目录
MD=$B/work_seed/fig_relags_s7
mkdir -p $MD
ln -sfn $B/ref/relags_hf/figurines/point_cloud $MD/point_cloud
ln -sfn $B/ref/relags_hf/figurines/nag-l1.pt $MD/nag-l1.pt
cat > "$MD/cfg_args" <<CFG
Namespace(sh_degree=3, source_path='$SR', model_path='$MD', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False, render_items=['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature'])
CFG
cd $B/ReLaGS
export PYTHONPATH="$B/ReLaGS/ext:$B/ReLaGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
$PY -u $B/tools/run_seeded.py 7 merge_proj.py -s "$SR" -m "$MD" \
    --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 > "$MD/merge.log" 2>&1
echo "[$(date +%H:%M)] merge rc=$? :: $(grep -o 'nag shape.*' $MD/merge.log | tail -1)" >> $LOG
P=$(ls $MD/sai_nag*.pt 2>/dev/null | head -1); [ -n "$P" ] && ln -sfn "$P" $MD/sai_nag_3.0.pt
echo "[$(date +%H:%M)] pt=$P" >> $LOG
# 3) 评测
export PYTHONPATH="$D:$B/ReLaGS:$B/ReLaGS/ext"
O=$B/work_seed/eval_fig_relags_s7; mkdir -p $O
$PY $D/deco/lerf_ovs/eval_lerf_ovs.py --scene figurines --source "$SR" --model "$MD" \
   --nag "$MD/sai_nag_3.0.pt" --gt_root "$GT" --out "$O" --csv "$O/q.csv" \
   --center none --renorm 0 > "$O/eval.log" 2>&1
echo "[$(date +%H:%M)] EVAL :: $(grep -o 'SCENE_RESULT.*' $O/eval.log | tail -1)" >> $LOG
echo "[$(date +%H:%M)] feature compare vs official:" >> $LOG
$PY - <<'PY' >> $LOG 2>&1
import torch, numpy as np
B='/root/rivermind-data/ReLaGS_Reproduction'
A=torch.load(B+'/ref/relags_hf/figurines/sai_nag.pt',map_location='cpu')
O=torch.load(B+'/work_seed/fig_relags_s7/sai_nag_3.0.pt',map_location='cpu')
la=[x.numpy() for x in A['nag']]; lo=[x.numpy() for x in O['nag']]
for k in range(3):
    fa,fo=A['nag_feat'][k],O['nag_feat'][k]
    ks,kd=int(la[k+1].max())+1,int(lo[k+1].max())+1
    cnt=np.bincount(la[k+1].astype(np.int64)*kd+lo[k+1],minlength=ks*kd).reshape(ks,kd)
    j=cnt.argmax(1); w=cnt.sum(1).astype(np.float64); w/=w.sum()
    an=fa/fa.norm(dim=1,keepdim=True); bn=fo/fo.norm(dim=1,keepdim=True)
    cos=(an*bn[torch.from_numpy(j)]).sum(1)
    print(f'L{k+1}: label-matched cos weighted={float((cos*w).sum()):.4f}')
PY
echo "[$(date +%H:%M)] DONE" >> $LOG
