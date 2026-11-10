#!/usr/bin/env python3
"""Stop the current SAFMN cloud run on a verified checkpoint and score trend.

This is a pragmatic early-stop rule for the exploratory run, plus a 50k cost
ceiling. It changes no train config and does not claim paper reproduction.
"""

import os
import re
import signal
import time
from pathlib import Path


BASE = Path('/root/fsas/sr-repro')
LOG = BASE / 'full500k_resume.stdout.log'
EXPERIMENT = BASE / 'SAFMN/experiments/SAFMN_DF2K_x4_cloud_sub_500K'
WATCH_LOG = BASE / 'stop_safmn_at_50000.log'
PID = 8451
TARGET = 50000
MIN_EARLY = 20000
BASELINE = 28.9615
MAX_GAP = 0.20
MAX_GAIN_PER_10K = 0.05
EXPECTED = b'options/train/SAFMN/SAFMN_train_DF2K_x4_sub_cloud.yml'


def record(message):
    with WATCH_LOG.open('a') as output:
        output.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + message + '\n')
        output.flush()


def process_matches():
    try:
        command = Path(f'/proc/{PID}/cmdline').read_bytes()
    except FileNotFoundError:
        return False
    return b'basicsr/train.py' in command and EXPECTED in command


def validation_scores():
    try:
        lines = LOG.read_text(errors='replace').splitlines()
    except FileNotFoundError:
        return {}
    scores = {}
    progress_iter = saved_iter = validation_iter = None
    for line in lines:
        progress = re.search(r'INFO: \[SAFMN.*\biter:\s*([\d,]+),', line)
        if progress:
            progress_iter = int(progress.group(1).replace(',', ''))
        if 'Saving models and training states.' in line:
            saved_iter = progress_iter
        if 'Validation DIV2K_val100' in line:
            validation_iter = saved_iter
        metric = re.search(r'# psnr:\s*([0-9.]+)', line)
        if metric and validation_iter is not None:
            scores[validation_iter] = float(metric.group(1))
            validation_iter = None
    return scores


def checkpoint_ready(iteration):
    model = EXPERIMENT / f'models/net_g_{iteration}.pth'
    state = EXPERIMENT / f'training_states/{iteration}.state'
    return all(path.is_file() and path.stat().st_size > 1_000_000
               for path in (model, state))


def stop_candidate(scores):
    if not scores:
        return None
    iteration = max(scores)
    if iteration < MIN_EARLY or not checkpoint_ready(iteration):
        return None
    if iteration >= TARGET:
        return iteration, scores[iteration], '50k cost ceiling'
    previous = scores.get(iteration - 10000)
    if (previous is not None
            and scores[iteration] >= BASELINE - MAX_GAP
            and scores[iteration] - previous < MAX_GAIN_PER_10K):
        return iteration, scores[iteration], 'near baseline and gain under 0.05 dB per 10k'
    return None


def main():
    record(f'watching PID {PID}; early stop from {MIN_EARLY}; upper limit {TARGET}')
    while True:
        if not process_matches():
            record('training process absent or changed; watcher exiting without action')
            return
        candidate = stop_candidate(validation_scores())
        if candidate is not None:
            iteration, score, reason = candidate
            if process_matches():
                os.kill(PID, signal.SIGTERM)
                record(f'SIGTERM sent after {iteration} checkpoint and DIV2K_val100 PSNR {score:.4f}: {reason}')
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
