#!/usr/bin/env bash
set -uo pipefail
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
REPO=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028/DeCo-3DGS
REL=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
SRC=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
MODEL=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
BASE=/root/rivermind-data/repro/deco-b123/cnone_20261101
OUT=$BASE/gate
mkdir -p "$OUT"
# training-free preds: pred = NAG root score (no learned predictor)
$PY - "$BASE" <<PYEOF
import csv, glob, sys, os
base=sys.argv[1]
out=os.path.join(base,"preds_cnone.csv")
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["scene","image","prompt","rank","pred","iou","cat"])
    for p in sorted(glob.glob(base+"/cand/*_candidates.csv")):
        for r in csv.DictReader(open(p)):
            w.writerow([r["scene"],r["image"],r["prompt"],r["rank"],r["root_score"],r["iou"],""])
print("preds ->",out)
PYEOF
cd "$REPO"
export RELAGS_REPO="$REL" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$REL:$REL/ext:$REL/ext/spt/dependencies/grid_graph/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$REL/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"
for sc in figurines ramen teatime waldo_kitchen; do
  echo "START gate $sc $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_c_mask_gate.py --scene "$sc" --source "$SRC/$sc" --model "$MODEL/$sc" \
    --nag "$MODEL/$sc/sai_nag.pt" --gt_root "$GT" --out "$OUT" \
    --preds "$BASE/preds_cnone.csv" --evidence "$BASE/ev" \
    --id_csv "$BASE/segid/${sc}_segidentity.csv" --center none > "$OUT/$sc.log" 2>&1
  echo "END gate $sc rc=$? $(date -Is)"; tail -1 "$OUT/$sc.log"
done
echo ALL_DONE
