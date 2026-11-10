#!/usr/bin/env python3
"""Two fixed pilot jobs, exclusive queue lock, no automatic expansion."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from train_pilot import save_json

HERE = Path(__file__).resolve().parent


def main():
    (HERE / 'pilot').mkdir(exist_ok=True)
    lock = (HERE / 'pilot/queue.lock').open('a')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Existing pilot queue holds lock; no duplicate started')
    protocol = json.loads((HERE / 'pilot_protocol.json').read_text())
    status = {'status': 'running', 'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'completed': [], 'failed': [], 'active': None, 'total': len(protocol['models'])}
    path = HERE / 'records/pilot_queue_status.json'
    for model in protocol['models']:
        result = HERE / 'pilot/results' / f'{model}_OISST_x4_s10/pilot_results.json'
        if result.exists(): status['completed'].append(model); continue
        if (HERE / 'pilot/STOP_REQUESTED').exists(): status['status'] = 'paused'; break
        status['active'] = model
        save_json(path, status)
        with (HERE / f'logs/pilot_{model}.log').open('a', buffering=1) as log:
            process = subprocess.Popen([sys.executable, str(HERE / 'train_pilot.py'), '--model', model], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            status['trainer_pid'] = process.pid; save_json(path, status)
            code = process.wait()
        if code != 0:
            status.update(status='paused' if code == 75 else 'failed', exit_code=code)
            if code != 75: status['failed'].append(model)
            break
        assert result.exists()
        status['completed'].append(model)
        status['active'] = None
    else:
        status['status'] = 'complete'
        summary = {model: json.loads((HERE / 'pilot/results' / f'{model}_OISST_x4_s10/pilot_results.json').read_text()) for model in protocol['models']}
        save_json(HERE / 'records/pilot_summary.json', summary)
    status['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_json(path, status)
    print(json.dumps(status), flush=True)


if __name__ == '__main__': main()
