#!/usr/bin/env python3
"""Launch the unchanged queue with the existing project-local build environment."""
import datetime
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    assert ROOT == Path('/root/wmsr-repro'), 'Cloud-only launcher'
    headers = list((ROOT / '.toolchains/python').glob('*/include/python3.10/Python.h'))
    assert len(headers) == 1, 'Expected the existing isolated CPython3.10 headers'
    existing = [line for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines()
                if str(HERE) + '/' in line and ('train_controlled.py' in line or 'run_queue.py' in line)]
    assert not existing, existing
    environment = os.environ.copy()
    overrides = {
        'CPATH': str(headers[0].parent),
        'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'XDG_CACHE_HOME': str(ROOT / '.cache'),
        'MPLCONFIGDIR': str(ROOT / '.cache/matplotlib'),
        'TRITON_CACHE_DIR': str(ROOT / '.cache/triton'),
        'CUDA_CACHE_PATH': str(ROOT / '.cache/cuda5090'),
        'CUDA_VISIBLE_DEVICES': '0',
    }
    environment.update(overrides)
    python = str(ROOT / '.venv5090/bin/python')
    subprocess.run([python, '-c',
                    'import torch, mamba_ssm; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; '
                    'assert torch.cuda.is_available(); print("Project-local Mamba/CUDA import passed")'],
                   env=environment, cwd=ROOT, check=True)
    status_path = HERE / 'records/queue_status.json'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if status_path.exists():
        (HERE / 'records' / f'queue_before_resume_{stamp}.json').write_bytes(status_path.read_bytes())
    with (HERE / 'logs/queue.log').open('a') as stream:
        child = subprocess.Popen([python, '-u', str(HERE / 'run_queue.py')], env=environment,
                                 cwd=HERE, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = {'resumed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'queue_pid': child.pid, 'runtime_environment': overrides,
              'training_configuration_changed': False,
              'startup_import_check': 'pass',
              'boundary': 'Reuse existing project headers and caches; no global install or network/trainer change.'}
    (HERE / 'records/queue_launch.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
