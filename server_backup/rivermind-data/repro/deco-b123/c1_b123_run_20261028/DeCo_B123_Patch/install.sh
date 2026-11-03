#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?usage: bash install.sh /path/to/DeCo-3DGS}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for f in deco/lerf_ovs/eval_lerf_ovs.py deco/deco_ops.py nag_data.py; do test -f "$ROOT/$f" || { echo "Missing full DeCo file $f" >&2; exit 1; }; done
for f in evidence_readout.py evidence_modes.py; do
 target="$ROOT/deco/lerf_ovs/$f"
 test ! -e "$target" || { echo "Refusing overwrite: $target" >&2; exit 1; }
 cp "$HERE/$f" "$target"
done
cp "$HERE/build_bank.py" "$ROOT/deco/lerf_ovs/build_bank_b123.py"
python "$HERE/patch_deco_evaluator.py" "$ROOT"
python -m py_compile "$ROOT/deco/lerf_ovs/"{evidence_readout,evidence_modes,build_bank_b123,eval_lerf_ovs_b123}.py
