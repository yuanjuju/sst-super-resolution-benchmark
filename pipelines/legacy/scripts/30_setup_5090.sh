#!/usr/bin/env bash
set -euo pipefail
cd /root/wmsr-repro
export UV_CACHE_DIR="$PWD/.cache/uv5090"
export XDG_CACHE_HOME="$PWD/.cache"
export UV_HTTP_TIMEOUT=120
export UV_PYTHON_INSTALL_DIR="$PWD/.toolchains/python"
if [ ! -x .venv5090/bin/python ]; then
  uv venv --python /usr/bin/python3 .venv5090
fi
uv python install 3.10.12
uv pip install --python .venv5090/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
uv pip install --python .venv5090/bin/python --index-url https://pypi.org/simple -r configs/requirements-5090.txt
.venv5090/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); x=torch.randn(32,32,device="cuda",requires_grad=True); x.square().mean().backward(); print("BLACKWELL_TORCH_PASS")'
printf 'Next: python3 scripts/32_build_mamba5090.py; compiler and headers stay inside the project.\n'
