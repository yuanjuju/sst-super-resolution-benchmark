#!/usr/bin/env bash
set -euo pipefail
WMSR_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WMSR_ROOT"
export UV_CACHE_DIR="$WMSR_ROOT/.cache/uv"
export XDG_CACHE_HOME="$WMSR_ROOT/.cache"
export UV_HTTP_TIMEOUT=120
printf 'Installing pinned core packages inside %s\n' "$WMSR_ROOT/.venv"
uv pip install --python "$WMSR_ROOT/.venv/bin/python" --index-url https://pypi.org/simple -r configs/requirements-core.txt
printf 'Installing official Mamba CUDA wheel\n'
uv pip install --python "$WMSR_ROOT/.venv/bin/python" --no-deps 'https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
uv pip freeze --python "$WMSR_ROOT/.venv/bin/python" > records/environment.lock.txt
printf 'ENVIRONMENT_SETUP_COMPLETE\n'
