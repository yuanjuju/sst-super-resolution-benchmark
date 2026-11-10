"""Project-owned caches, required Python headers, and duplicate process check."""
import datetime
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path('/root/wmsr-repro')
PYTHON = BASE / '.venv5090/bin/python'


def main():
    assert not (HERE / 'pilot/STOP_REQUESTED').exists(), 'Explicit pause exists; require authorized resume'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: args = (p / 'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if str(HERE / 'run_queue.py').encode() in args or str(HERE / 'train_continue.py').encode() in args:
            raise RuntimeError(f'Existing stage process {p.name}; inspect instead of duplicate launch')
    env = os.environ.copy()
    for key, name in {'PYTHONPYCACHEPREFIX':'pycache', 'XDG_CACHE_HOME':'xdg', 'TRITON_CACHE_DIR':'triton',
                      'TORCHINDUCTOR_CACHE_DIR':'inductor', 'CUDA_CACHE_PATH':'cuda',
                      'PYTORCH_KERNEL_CACHE_PATH':'torch_kernels', 'TMPDIR':'tmp', 'MPLCONFIGDIR':'matplotlib'}.items():
        path = HERE / '.cache' / name; path.mkdir(parents=True, exist_ok=True); env[key] = str(path)
    env.update(CPATH=str(BASE / '.toolchains/python/cpython-3.10.12-linux-x86_64-gnu/include/python3.10'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with (HERE / 'logs/queue.log').open('a') as log:
        child = subprocess.Popen([str(PYTHON), str(HERE / 'run_queue.py')], cwd=HERE, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': child.pid,
              'python': str(PYTHON), 'cache_root': str(HERE / '.cache'), 'CPATH': env['CPATH'],
              'total_updates_per_model': 5000, 'source_step': 1000, 'models': ['ESC','WMSR']}
    path = HERE / 'records/queue_launch.json'
    if path.exists():
        old = json.loads(path.read_text())
        (HERE / 'records' / f'queue_launch_before_{child.pid}.json').write_text(json.dumps(old, indent=2)+'\n')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
