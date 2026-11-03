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
WIT=$BASE/witness_image
GATE=$BASE/gate_image
mkdir -p "$WIT" "$GATE"
cd "$REPO"
export RELAGS_REPO="$RELAGS" HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO:$REPO/deco/lerf_ovs:$RELAGS:$RELAGS/ext:$RELAGS/ext/spt/dependencies/grid_graph/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/bin:$RELAGS/ext/spt/dependencies/parallel_cut_pursuit/python/wrappers"

echo "== witness dump (center=image)"
for scene in figurines ramen teatime waldo_kitchen; do
  b="$SAS/${scene}_new"; [ -d "$b" ] || b="$SAS/waldo_new"
  echo "START wit $scene $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_segment_identity_dump_witness.py --scene "$scene" --source "$SRCROOT/$scene" --model "$MODELROOT/$scene" \
    --nag "$MODELROOT/$scene/sai_nag.pt" --gt_root "$GTROOT" --sas "$b" --out "$WIT" > "$WIT/$scene.log" 2>&1
  echo "END wit $scene rc=$? $(date -Is)"; tail -1 "$WIT/$scene.log"
done

echo "== preds from center=image candidate dump"
"$PY" - "$BASE" <<PYEOF
import csv, glob, sys, os
base=sys.argv[1]
out=os.path.join(base,"preds_image.csv")
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["scene","image","prompt","rank","pred","iou","cat"])
    for p in sorted(glob.glob("/root/rivermind-data/repro/deco-b123/lerf_selector_20261030/*_candidates.csv")):
        for r in csv.DictReader(open(p)):
            w.writerow([r["scene"],r["image"],r["prompt"],r["rank"],r["root_score"],r["iou"],""])
print("preds ->",out)
PYEOF

echo "== gate (center=image, training-free preds)"
for sc in figurines ramen teatime waldo_kitchen; do
  echo "START gate $sc $(date -Is)"
  "$PY" deco/lerf_ovs/lerf_c_mask_gate.py --scene "$sc" --source "$SRCROOT/$sc" --model "$MODELROOT/$sc" \
    --nag "$MODELROOT/$sc/sai_nag.pt" --gt_root "$GTROOT" --out "$GATE" \
    --preds "$BASE/preds_image.csv" --evidence /root/rivermind-data/repro/deco-b123/lerf_evidence_20261030 \
    --id_csv /root/rivermind-data/repro/deco-b123/lerf_segidentity_20261030/${sc}_segidentity.csv --center image > "$GATE/$sc.log" 2>&1
  echo "END gate $sc rc=$? $(date -Is)"; tail -1 "$GATE/$sc.log"
done
echo ALL_DONE
