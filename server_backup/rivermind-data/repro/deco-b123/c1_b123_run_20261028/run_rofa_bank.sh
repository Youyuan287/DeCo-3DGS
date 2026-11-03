#!/usr/bin/env bash
# ROFA variants in the repository's per-view evidence bank (B2), evaluated on the
# 0.6401 anchor features. No merge_proj rebuild (PREREG_relags_rofa.md).
set -uo pipefail
ROOT=/root/rivermind-data/repro/deco-b123/c1_b123_run_20261028
DECO=$ROOT/DeCo-3DGS
RELAGS=/root/rivermind-data/ReLaGS_Reproduction/ReLaGS
PY=/root/rivermind-data/ReLaGS_Reproduction/venv/bin/python3
GT=/root/rivermind-data/ReLaGS_Reproduction/ref/hf_lerf/lerf_ovs/label
MODELROOT=/root/rivermind-data/ReLaGS_Reproduction/ref/relags_hf
SRCROOT=/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs
OUT=$ROOT/rofa_bank_20261031
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 PYTHONPATH="$DECO:$RELAGS:$RELAGS/ext"

inputs() {
  case "$1" in
    figurines) OBS=$ROOT/evidence_v2_four_20261028/generated_inputs/figurines/observations_level3.npz ;;
    ramen) OBS=/root/rivermind-data/repro/deco-b123/official_nag_export_20261028/observations_level3.npz ;;
    teatime) OBS=$ROOT/evidence_v2_four_20261028/generated_inputs/teatime/observations_level3.npz ;;
    waldo_kitchen) OBS=$ROOT/waldo_official_b012_20261028/observations_level3.npz ;;
  esac
}

for arm in anchor off raw center; do
  csv="$OUT/${arm}_query_metrics.csv"; rm -f "$csv"
  mkdir -p "$OUT/$arm"
  for sc in ${SCENES:-figurines ramen teatime waldo_kitchen}; do
    inputs "$sc"
    MODEL=$MODELROOT/$sc; NAG=$MODEL/sai_nag.pt
    B=$OUT/banks/$sc; mkdir -p "$B"
    if [ "$arm" = anchor ]; then
      echo "=== [$(date -Is)] eval $sc anchor ==="
      "$PY" "$DECO/deco/lerf_ovs/eval_lerf_ovs_v3.py" --scene "$sc" \
        --source "$SRCROOT/$sc" --model "$MODEL" --nag "$NAG" --gt_root "$GT" \
        --out "$OUT/$arm" --csv "$csv" --topk 5 --level_until 3 --filter_small 50 \
        --center none --text canon --renorm 0 --sel gap --evidence-v2 off \
        --tag "ROFA_anchor" > "$OUT/$arm/$sc.eval.log" 2>&1
      echo "EVAL $sc anchor rc=$? $(grep -o 'SCENE_RESULT.*' "$OUT/$arm/$sc.eval.log" | tail -1)"
      continue
    fi
    if [ "$arm" = off ]; then B2=0; RM=raw; else B2=1; RM=$arm; fi
    if [ ! -f "$B/B1_${arm}.npz" ]; then
      echo "=== [$(date -Is)] bank $sc $arm ==="
      "$PY" "$DECO/deco/lerf_ovs/build_bank_rcs.py" --observations "$OBS" --nag "$NAG" \
        --out "$B/B1_${arm}.npz" --b1 1 --b2 "$B2" --geo-weight 0.25 --max-modes 3 \
        --mode-selection rcs --rofa-mode "$RM" > "$OUT/$arm/$sc.build.log" 2>&1
      echo "BANK $sc $arm rc=$?"
    fi
    echo "=== [$(date -Is)] eval $sc $arm ==="
    "$PY" "$DECO/deco/lerf_ovs/eval_lerf_ovs_v3.py" --scene "$sc" \
      --source "$SRCROOT/$sc" --model "$MODEL" --nag "$NAG" --gt_root "$GT" \
      --out "$OUT/$arm" --csv "$csv" --topk 5 --level_until 3 --filter_small 50 \
      --center none --text canon --renorm 0 --sel gap \
      --evidence-v2 semantic --evidence-bank "$B/B1_${arm}.npz" \
      --evidence-observations "$OBS" --evidence-members "$B/B1_${arm}.members.json" \
      --evidence-temperature 0.08 --evidence-coefficient 0.25 --evidence-aggregation softmax \
      --tag "ROFA_$arm" > "$OUT/$arm/$sc.eval.log" 2>&1
    echo "EVAL $sc $arm rc=$? $(grep -o 'SCENE_RESULT.*' "$OUT/$arm/$sc.eval.log" | tail -1)"
  done
done
echo ALL_DONE