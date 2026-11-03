#!/usr/bin/env bash
set -Eeuo pipefail

# One-shot prototype reproduction for LERF-OVS/ramen.
# Status of final IoU: PROTOTYPE/UNVERIFIED.
# Requires the ReLaGS reproduction tree and its existing virtualenv.

REPRO="${REPRO:-/root/rivermind-data/ReLaGS_Reproduction}"
CODE="${CODE:-$REPRO/ReLaGS}"
PY="${PY:-$REPRO/venv/bin/python}"
SCENE="ramen"
SRC0="/root/rivermind-data/VALA/dataset/3dgs/lerf_ovs/$SCENE"
FEAT="/root/rivermind-data/lerf_oseg/build/${SCENE}_run/language_features"
MODEL0="$REPRO/ref/relags_hf/$SCENE"
GT="$REPRO/ref/hf_lerf/lerf_ovs/label/$SCENE"
RUN="${RUN:-$REPRO/work/multievidence_manual/$SCENE}"
OBS="${OBS:-$REPRO/observations_manual/$SCENE}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$PY" ]] || die "Python environment not found: $PY"
[[ -d "$SRC0/images" ]] || die "LERF-OVS source missing: $SRC0"
[[ -d "$FEAT" ]] || die "language_features missing: $FEAT"
[[ -d "$MODEL0" ]] || die "author model missing: $MODEL0"
[[ -d "$GT" ]] || die "LERF-OVS GT missing: $GT"
[[ -f "$CODE/production_observation_adapter.py" ]] || die "adapter not installed in $CODE"
[[ -f "$CODE/multi_evidence.py" ]] || die "multi_evidence.py not installed in $CODE"
[[ -f "$CODE/analyze_ramen_observations.py" ]] || die "analysis helper not installed in $CODE"
[[ -f "$CODE/prototype_b3_ramen.py" ]] || die "B3 helper not installed in $CODE"
[[ -f "$CODE/eval_ramen_evidence_prototype.py" ]] || die "prototype evaluator not installed in $CODE"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found; merge/render requires GPU"
nvidia-smi >/dev/null || die "GPU is not available"

echo "[1/7] Checking frame/feature correspondence"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
find "$SRC0/images" -maxdepth 1 -type f -printf '%f\n' \
  | sed 's/\.[^.]*$//' | sort > "$tmp/images.txt"
find "$FEAT" -maxdepth 1 -name '*_s.npy' -type f -printf '%f\n' \
  | sed 's/_s\.npy$//' | sort > "$tmp/features.txt"
missing="$(comm -23 "$tmp/images.txt" "$tmp/features.txt" | wc -l)"
extra="$(comm -13 "$tmp/images.txt" "$tmp/features.txt" | wc -l)"
[[ "$missing" == 0 && "$extra" == 0 ]] || die "frame mismatch: missing=$missing extra=$extra"
echo "  frames=$(wc -l < "$tmp/images.txt") missing=$missing extra=$extra"

[[ ! -e "$RUN" ]] || die "run directory already exists; choose RUN=/new/path"
[[ ! -e "$OBS" ]] || die "observation directory already exists; choose OBS=/new/path"

echo "[2/7] Creating isolated input/model copies"
mkdir -p "$RUN/input" "$OBS" "$REPRO/logs"
cp -a "$SRC0" "$RUN/input/$SCENE"
if [[ -L "$RUN/input/$SCENE/language_features" ]]; then
  unlink "$RUN/input/$SCENE/language_features"
elif [[ -d "$RUN/input/$SCENE/language_features" ]]; then
  [[ -z "$(find "$RUN/input/$SCENE/language_features" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "copied language_features directory is not empty"
  rmdir "$RUN/input/$SCENE/language_features"
fi
ln -s "$FEAT" "$RUN/input/$SCENE/language_features"
cp -a "$MODEL0" "$RUN/$SCENE"

echo "[3/7] Exporting real production observations (GPU)"
cd "$CODE"
RELAGS_OBSERVATION_OUT="$OBS/observations_level{level}.npz" \
  "$PY" -u merge_proj.py \
    -s "$RUN/input/$SCENE" -m "$RUN/$SCENE" \
    --thres_connect 0.9,0.7,0.7 --thres_merge 20 --tau 3 --iteration 0 \
    2>&1 | tee "$REPRO/logs/multievidence_${SCENE}.log"

for level in 1 2 3; do
  [[ -s "$OBS/observations_level${level}.npz" ]] || die "missing level ${level} NPZ"
done

echo "[4/7] K=1 check and B2 raw bank (CPU)"
"$PY" analyze_ramen_observations.py \
  "$OBS/observations_level3.npz" \
  "$OBS/b2_raw_level3.npz" \
  "$OBS/structural_check_level3.json"
cat "$OBS/structural_check_level3.json"

echo "[5/7] B3 same-scene prototype (CPU; not certified)"
"$PY" prototype_b3_ramen.py \
  "$OBS/observations_level3.npz" \
  "$OBS/b2_raw_level3.npz" \
  "$OBS/b3_debiased_same_scene_prototype_level3.npz" \
  "$OBS/b3_same_scene_prototype_level3.json"
cat "$OBS/b3_same_scene_prototype_level3.json"

echo "[6/7] Prototype IoU evaluation (GPU rendering; not certified)"
"$PY" eval_ramen_evidence_prototype.py \
  "$OBS/observations_level3.npz" \
  "$OBS/b2_raw_level3.npz" \
  "$OBS/b3_debiased_same_scene_prototype_level3.npz" \
  "$RUN/input/$SCENE" "$RUN/$SCENE" \
  "$RUN/$SCENE/sai_nag_3.0.pt" "$GT" "$OBS/prototype_eval"

echo "[7/7] Outputs"
find "$OBS" -maxdepth 2 -type f -printf '%p %s bytes\n' | sort
cat "$OBS/prototype_eval/summary.json"
echo "DONE: all numbers are PROTOTYPE/UNVERIFIED."
