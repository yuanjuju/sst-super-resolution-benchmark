#!/usr/bin/env bash
set -euo pipefail

PFT_STAGE_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WMSR_BASE_ROOT="${WMSR_BASE_ROOT:-/root/wmsr-repro}"
PFT_PYTHON="${PFT_PYTHON:-$WMSR_BASE_ROOT/.toolchains/python/cpython-3.10.12-linux-x86_64-gnu/bin/python3.10}"
case "$PFT_STAGE_ROOT" in
  /root/wmsr-repro/strong_models/*) ;;
  *) echo 'Use a separate cloud directory under /root/wmsr-repro/strong_models/.' >&2; exit 2 ;;
esac
test -x "$PFT_PYTHON"
export WMSR_BASE_ROOT
export CUDA_HOME="$WMSR_BASE_ROOT/.toolchains/cuda-12.8.1"
test -x "$CUDA_HOME/bin/nvcc"
export PATH="$PFT_STAGE_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$WMSR_BASE_ROOT/.toolchains/python/cpython-3.10.12-linux-x86_64-gnu/include/python3.10"
export PIP_CACHE_DIR="$PFT_STAGE_ROOT/.cache/pip"
export XDG_CACHE_HOME="$PFT_STAGE_ROOT/.cache"
export TORCH_EXTENSIONS_DIR="$PFT_STAGE_ROOT/.cache/torch_extensions"
export CUDA_CACHE_PATH="$PFT_STAGE_ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$PFT_STAGE_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PFT_STAGE_ROOT/.cache/torch_inductor"
export PYTORCH_KERNEL_CACHE_PATH="$PFT_STAGE_ROOT/.cache/torch_kernels"
export TMPDIR="$PFT_STAGE_ROOT/.cache/tmp"
export PYTHONPYCACHEPREFIX="$PFT_STAGE_ROOT/.cache/pycache"
export MAX_JOBS=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$PFT_STAGE_ROOT/.cache/tmp" "$PFT_STAGE_ROOT/records"

