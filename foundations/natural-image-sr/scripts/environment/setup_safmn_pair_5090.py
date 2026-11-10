#!/usr/bin/env python3
"""Clone verified packages into a separate SAFMN environment, offline."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/fsas/sr-repro')
suite = root / 'safmn_fft_pair_30k'
suite.mkdir(parents=True, exist_ok=True)
source = root / 'CAMixerSR/.venv-5090'
target = root / 'SAFMN/.venv-5090'
if target.exists():
    raise RuntimeError(f'Refusing to overwrite environment: {target}')
source_site = source / 'lib/python3.10/site-packages'
if not source_site.is_dir():
    raise RuntimeError('Verified Python 3.10 environment is missing')
print('Creating independent project environment', target, flush=True)
subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(target)], check=True)
target_site = target / 'lib/python3.10/site-packages'
print('Copying verified packages without downloads or changes to source environment', flush=True)
subprocess.run(['cp', '-a', '--reflink=auto', str(source_site) + '/.', str(target_site)], check=True)
path = root / 'SAFMN/basicsr/data/degradations.py'
old = 'from torchvision.transforms.functional_tensor import rgb_to_grayscale'
new = 'from torchvision.transforms.functional import rgb_to_grayscale'
before = path.read_text()
backup = suite / 'originals/degradations.py'
backup.parent.mkdir(exist_ok=True)
if old in before:
    if backup.exists():
        raise RuntimeError('Compatibility backup already exists; inspect manually')
    backup.write_text(before)
    path.write_text(before.replace(old, new))
elif new not in before:
    raise RuntimeError('Unexpected grayscale import; inspect manually')
python = target / 'bin/python'
with (suite / 'environment.txt').open('w') as stream:
    subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=stream, check=True)
probe = subprocess.check_output([str(python), '-c',
    'import torch,torchvision,PIL,sys; print(sys.prefix); print(torch.__version__,torchvision.__version__,PIL.__version__); print(torch.cuda.is_available(),torch.cuda.get_device_name(0))'], text=True)
print(probe, flush=True)
record = {'status': 'ready', 'checked_at': datetime.datetime.now().astimezone().isoformat(),
          'source_environment': str(source), 'independent_environment': str(target),
          'packages_copied_offline': True, 'probe': probe,
          'compatibility_file': str(path), 'compatibility_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(suite / 'setup.json').write_text(json.dumps(record, indent=2) + '\n')
print('SETUP_COMPLETE', flush=True)
