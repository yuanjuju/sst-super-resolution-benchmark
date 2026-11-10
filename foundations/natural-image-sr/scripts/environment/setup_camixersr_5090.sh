#!/usr/bin/env bash
# Run in a detached shell and retain its stdout as an environment setup log.
set -euo pipefail

sr_project_root="${1:-/root/fsas/sr-repro}"
camixer_env="$sr_project_root/CAMixerSR/.venv-5090"

echo 'STAGE: create project-local environment'
# The container lacks ensurepip. Use the existing project's pip to bootstrap
# the new interpreter; all installed packages go into the new environment.
python3 -m venv --without-pip "$camixer_env"
"$sr_project_root/.venv/bin/python" -m pip --python "$camixer_env/bin/python" install --no-cache-dir --index-url https://pypi.org/simple pip==25.2 setuptools==69.5.1 wheel==0.45.1

echo 'STAGE: install Blackwell-compatible PyTorch'
"$camixer_env/bin/python" -m pip install --no-cache-dir --progress-bar off --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple torch==2.7.1+cu128 torchvision==0.22.1+cu128 numpy==1.26.4

echo 'STAGE: install training dependencies'
"$camixer_env/bin/python" -m pip install --no-cache-dir --progress-bar off --index-url https://pypi.org/simple -r "$sr_project_root/requirements-camixersr-5090.txt"
"$camixer_env/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation --index-url https://pypi.org/simple basicsr==1.4.2

echo 'STAGE: adapt legacy BasicSR import for torchvision 0.22'
"$camixer_env/bin/python" - <<'PY'
from pathlib import Path
import sysconfig

p = Path(sysconfig.get_path('purelib')) / 'basicsr/data/degradations.py'
old = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
new = 'from torchvision.transforms.functional import rgb_to_grayscale'
source = p.read_text()
if old in source:
    p.with_suffix('.py.original').write_text(source)
    p.write_text(source.replace(old, new))
elif new not in source:
    raise RuntimeError('Unexpected BasicSR degradation import; inspect before editing')
print('BasicSR compatibility import ready:', p)
PY

echo 'STAGE: verify CUDA computation'
"$camixer_env/bin/python" - <<'PY'
import torch
import torchvision
import basicsr

print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('basicsr:', basicsr.__version__, basicsr.__file__)
print('gpu:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
print('arches:', torch.cuda.get_arch_list())
x = torch.randn(64, 64, device='cuda', requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
print('CUDA forward/backward: OK')
PY

"$camixer_env/bin/python" -m pip freeze > "$sr_project_root/camixersr_5090_environment.txt"
echo 'SETUP_COMPLETE'
