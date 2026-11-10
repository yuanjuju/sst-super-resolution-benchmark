"""Two fixed continuations only; exclusive lock and graceful pause propagation."""
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STOP = False
CHILD = None


def stop(_sig, _frame):
    global STOP
    STOP = True
    if CHILD and CHILD.poll() is None: CHILD.send_signal(signal.SIGTERM)


def save(value):
    value.update(pid=os.getpid(), updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    path = HERE / 'records/queue_status.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def main():
    global CHILD
    lock = (HERE / 'records/queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    completed = []
    state = {'status': 'running', 'total': 2, 'completed': completed, 'failed': [], 'active': None}
    for name in protocol['models']:
        if STOP or (HERE / 'pilot/STOP_REQUESTED').exists():
            state.update(status='paused', active=None); save(state); return 75
        run = HERE / 'pilot/results' / f'{name}_OISST_x4_s10'
        result = run / 'pilot_results.json'
        if result.exists():
            r = json.loads(result.read_text())
            assert r['status'] == 'complete' and r['updates'] == 5000
            completed.append(name); continue
        with (HERE / f'logs/continue_{name}.log').open('a') as log:
            CHILD = subprocess.Popen([sys.executable, str(HERE / 'train_continue.py'), '--model', name], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            if STOP: CHILD.send_signal(signal.SIGTERM)
            state.update(active=name, trainer_pid=CHILD.pid); save(state)
            code = CHILD.wait()
        CHILD = None
        if code == 75 or STOP:
            state.update(status='paused', active=None, trainer_pid=None); save(state); return 75
        if code != 0:
            state.update(status='failed', failed=[name], returncode=code, trainer_pid=None); save(state); return code
        assert json.loads(result.read_text())['updates'] == 5000
        completed.append(name)
        state.update(active=None, trainer_pid=None); save(state)
    state.update(status='complete', active=None, trainer_pid=None); save(state)
    return 0


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    sys.exit(main())
