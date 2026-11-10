#!/usr/bin/env python3
"""Sequential, resumable controlled queue. Stop on a new failure; never retune."""
import datetime
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(value):
    path = HERE / 'records/queue_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    (HERE / 'records').mkdir(exist_ok=True)
    (HERE / 'logs').mkdir(exist_ok=True)
    lock = (HERE / 'records/queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ['preflight.json', 'reference_replay.json']:
        assert json.loads((HERE / 'records' / name).read_text())['status'] == 'pass', name
    jobs = json.loads((HERE / 'plan.json').read_text())['jobs']
    status = {'status': 'running', 'started_at': now(), 'updated_at': now(), 'pid': os.getpid(),
              'total_jobs': len(jobs), 'completed': [], 'failed': [], 'active': [], 'pending': [j['id'] for j in jobs]}
    for spec in jobs:
        ident = spec['id']
        result = HERE / 'results' / ident / 'final_results.json'
        if result.is_file():
            old = json.loads(result.read_text())
            assert old['spec'] == spec and old['epochs'] == 100 and old['optimizer_steps'] == 10000
            assert old['test_model']['count'] == 200
            status['completed'].append(ident)
            status['pending'].remove(ident)
            continue
        log = HERE / 'logs' / (ident + '.log')
        with log.open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(HERE / 'train_controlled.py'),
                                      '--spec', str(HERE / 'configs' / (ident + '.json'))],
                                     stdout=stream, stderr=subprocess.STDOUT, cwd=HERE.parent)
            status['active'] = [{'id': ident, 'pid': child.pid}]
            status['pending'].remove(ident)
            status['updated_at'] = now()
            save(status)
            returncode = child.wait()
        status['active'] = []
        status['updated_at'] = now()
        if returncode or not result.is_file():
            status['status'] = 'needs_attention'
            status['failed'].append({'id': ident, 'returncode': returncode, 'log': str(log)})
            save(status)
            return
        status['completed'].append(ident)
        save(status)
    status['status'] = 'training_complete'
    status['updated_at'] = now()
    save(status)


if __name__ == '__main__':
    main()
