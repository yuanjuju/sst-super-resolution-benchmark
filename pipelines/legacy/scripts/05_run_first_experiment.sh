#!/usr/bin/env bash
set -euo pipefail
WMSR_ROOT=/root/wmsr-repro
export XDG_CACHE_HOME="$WMSR_ROOT/.cache"
export MPLCONFIGDIR="$WMSR_ROOT/.cache/matplotlib"
export TRITON_CACHE_DIR="$WMSR_ROOT/.cache/triton"
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
cd "$WMSR_ROOT"
"$WMSR_ROOT/.venv/bin/python" -u scripts/05_train_oisst.py --epochs 100
"$WMSR_ROOT/.venv/bin/python" -u scripts/06_make_figures.py
printf 'FIRST_EXPERIMENT_COMPLETE\n'
