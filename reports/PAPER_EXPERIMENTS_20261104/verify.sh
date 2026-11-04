#!/usr/bin/env bash
# Thin wrapper: recompute every R0 number and exit non-zero on any mismatch.
set -euo pipefail
cd "$(dirname "$0")/../.."
python reports/PAPER_EXPERIMENTS_20261104/verify.py