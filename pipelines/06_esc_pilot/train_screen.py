#!/usr/bin/env python3
"""Resumable paired development pilot. Opens only the existing training NPZ."""
import argparse
import copy
import csv
import fcntl
import datetime
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from screen_helpers import HERE, BASE, accumulated_l1, sha

import numpy as np
import torch

STOP = False


def stop(_sig, _frame):
    global STOP
    STOP = True


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def make_model(name, scale):
    assert name == 'ESC'
    from factory_esc import create_model
    return create_model(scale)


def batch_tensors(lr, hr, indices, augmentation=None):
    lows, highs = [], []
    for index in indices:
        a, b = lr[index], hr[index]
        if augmentation:
            if augmentation.random() < .5: a, b = a[:, ::-1], b[:, ::-1]
            if augmentation.random() < .5: a, b = a[::-1, :], b[::-1, :]
            if augmentation.random() < .5: a, b = a.T, b.T
        lows.append(a.copy()); highs.append(b.copy())
    return (torch.from_numpy(np.stack(lows))[:, None].repeat(1, 3, 1, 1).cuda(),
            torch.from_numpy(np.stack(highs))[:, None].repeat(1, 3, 1, 1).cuda())


@torch.inference_mode()
def evaluate(model, lr, hr, indices, samples, scale):
    sys.path.insert(0, str(BASE / 'upstream'))
    from utils import rgb_to_ycbcr
    from pytorch_msssim import ssim
    model.eval()
    rows = []
    for start in range(0, len(indices), 1):
        ids = indices[start:start + 1]
        low, high = batch_tensors(lr, hr, ids)
        prediction = model(low)
        assert torch.isfinite(prediction).all()
        b = scale
        error = ((prediction.mean(1, keepdim=True) - high.mean(1, keepdim=True)) * 42.)[:, :, b:-b, b:-b]
        mse = error.square().flatten(1).mean(1)
        mae = error.abs().flatten(1).mean(1)
        y_p = rgb_to_ycbcr(prediction.clamp(0, 1) * 255.)[:, :1, b:-b, b:-b]
        y_h = rgb_to_ycbcr(high.clamp(0, 1) * 255.)[:, :1, b:-b, b:-b]
        psnr = -10 * torch.log10(((y_p.double() - y_h.double()) / 255.).square().flatten(1).mean(1).clamp_min(1e-16))
        structure = ssim(y_p, y_h, data_range=255., size_average=False)
        for i, a, c, d, e in zip(ids, mse.tolist(), mae.tolist(), psnr.tolist(), structure.tolist()):
            rows.append({'sample_id': samples[i]['sample_id'], 'mse_c2': a, 'mae_c': c, 'psnr_y_db': d, 'ssim_y': e})
    result = {'count': len(rows), 'rmse_c': math.sqrt(sum(r['mse_c2'] for r in rows) / len(rows)),
              'mae_c': sum(r['mae_c'] for r in rows) / len(rows),
              'psnr_y_db': sum(r['psnr_y_db'] for r in rows) / len(rows),
              'ssim_y': sum(r['ssim_y'] for r in rows) / len(rows)}
    assert len(rows) == 80 and all(math.isfinite(v) for v in result.values())
    return result, rows


def run(name):
    protocol = json.loads((HERE / 'pilot_protocol.json').read_text())
    assert name in protocol['models']
    preflight = json.loads((HERE / 'records/esc_preflight.json').read_text())
    assert preflight['status'] == 'pass'
    assert json.loads((HERE / 'records/esc_profile_and_reference.json').read_text())['status'] == 'pass'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    seed, scale = protocol['seed'], protocol['scale']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    augmentation = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    data_path = BASE / 'records/formal' / f"{protocol['dataset']}_manifest.json"
    data = json.loads(data_path.read_text())
    split_path = HERE / 'development_splits' / f"{protocol['dataset']}.json"
    assert sha(split_path) == protocol['development_manifest_sha256']
    split = json.loads(split_path.read_text())
    assert sha(data_path) == split['source_manifest_sha256']
    train_path = BASE / data['splits']['train']['file']
    assert sha(train_path) == split['source_train_npz_sha256']
    with np.load(train_path) as archive:
        hr = ((archive['hr_celsius'] + 2) / 42).copy()
        lr = ((archive[f'lr_x{scale}_celsius'] + 2) / 42).copy()
    fit, validation = split['fit_indices'], split['validation_indices']
    assert len(fit) == 720 and len(validation) == 80 and len(hr) == 800
    run_dir = HERE / 'pilot/results' / f'{name}_{protocol["dataset"]}_x{scale}_s{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (run_dir / 'run.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run_dir / 'pilot_results.json').exists(): return 0
    config = {'model': name, 'protocol': protocol, 'protocol_sha256': sha(HERE / 'pilot_protocol.json'),
              'trainer_sha256': sha(Path(__file__)), 'preflight_helper_sha256': sha(HERE / 'screen_helpers.py'),
              'base_factory_sha256': sha(BASE / 'extended_models/factory.py'),
              'upstream_architecture_sha256': sha(HERE / 'sources/ESC/esc/archs/esc_arch.py'),
              'candidate_commit': json.loads((HERE / 'selected_candidate.json').read_text())['commit'],
              'baseline_reference_config_sha256': protocol['baseline_reference_config_sha256'],
              'fit_count': len(fit), 'validation_count': len(validation), 'final_test_images_evaluated': 0}
    config_path = run_dir / 'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text()) == config, 'Existing pilot cannot change configuration'
    save_json(config_path, config)
    from factory_esc import cache_rpe_indices
    model = cache_rpe_indices(make_model(name, scale).cuda(), 'cuda').train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['optimizer']['lr'], betas=tuple(protocol['optimizer']['betas']), eps=1e-8, weight_decay=0.)
    step, prior, order, position = 0, 0., [], 0
    checkpoints = run_dir / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    last = checkpoints / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert state['config'] == config
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema']); optimizer.load_state_dict(state['optimizer'])
        step, prior, order, position = state['step'], state['elapsed_seconds'], state['order'], state['position']
        augmentation.setstate(state['rng_augmentation']); generator.set_state(state['rng_loader'])
        random.setstate(state['rng_python']); np.random.set_state(state['rng_numpy']); torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
    start = time.monotonic()

    def checkpoint(status):
        elapsed = prior + time.monotonic() - start
        temp = last.with_suffix('.pt.tmp')
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'config': config, 'elapsed_seconds': elapsed, 'order': order, 'position': position,
                    'rng_augmentation': augmentation.getstate(), 'rng_loader': generator.get_state(),
                    'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all()}, temp)
        temp.replace(last)
        save_json(run_dir / 'progress.json', {'status': status, 'model': name, 'step': step, 'total_steps': protocol['updates'],
                                             'elapsed_seconds': elapsed, 'pid': os.getpid(), 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                             'final_test_images_evaluated': 0})

    checkpoint('training')
    losses = []
    metrics_path = run_dir / 'training_metrics.csv'
    stream = metrics_path.open('a' if step else 'w', newline='')
    writer = csv.DictWriter(stream, fieldnames=['step', 'lr', 'mean_l1_last_window', 'elapsed_seconds'])
    if step == 0: writer.writeheader(); stream.flush()
    try:
        while step < protocol['updates']:
            if STOP or (HERE / 'pilot/STOP_REQUESTED').exists(): checkpoint('paused'); return 75
            if position >= len(order):
                order = torch.randperm(len(fit), generator=generator).tolist(); position = 0
            ids = [fit[k] for k in order[position:position + 8]]
            assert len(ids) == 8
            position += 8
            low, high = batch_tensors(lr, hr, ids, augmentation)
            used_lr = protocol['optimizer']['lr'] * min((step + 1) / protocol['warmup_updates'], 1.)
            for group in optimizer.param_groups: group['lr'] = used_lr
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss, prediction = accumulated_l1(torch, model, low, high, micro_batch=2)
            if step % 100 == 0: assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters()): a.lerp_(b, 1. - protocol['ema_decay'])
                for a, b in zip(ema.buffers(), model.buffers()): a.copy_(b)
            step += 1; losses.append(loss)
            del low, high, prediction
            if step % 100 == 0 or step == protocol['updates']:
                row = {'step': step, 'lr': used_lr, 'mean_l1_last_window': sum(losses) / len(losses), 'elapsed_seconds': prior + time.monotonic() - start}
                writer.writerow(row); stream.flush(); losses = []
                checkpoint('training')
                print('PROGRESS', name, json.dumps(row), flush=True)
            if step % protocol['validation_every'] == 0 or step == protocol['updates']:
                checkpoint('validating')
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                val_result = {'step': step, 'split': 'training_only_development80', 'final_test_images_evaluated': 0}
                for key, network in [('raw', model), ('ema', ema)]:
                    metrics, rows = evaluate(network, lr, hr, validation, data['splits']['train']['samples'], scale)
                    val_result[key] = metrics
                    save_json(run_dir / f'validation_{step:06d}_{key}.json', {'summary': metrics, 'samples': rows})
                torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng)
                with (run_dir / 'validation_history.jsonl').open('a') as f: f.write(json.dumps(val_result) + '\n')
                print('VALIDATION', name, json.dumps(val_result), flush=True)
                checkpoint('training')
        final = checkpoints / 'pilot_final.pt'
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'step': step, 'config': config}, final)
        summary = {'status': 'complete', 'model': name, 'updates': step, 'elapsed_seconds': prior + time.monotonic() - start,
                   'last_validation': val_result, 'checkpoint_sha256': sha(final),
                   'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'boundary': '1000-update ESC development screen only. No final200 test evaluation; no convergence or general-superiority claim.'}
        save_json(run_dir / 'pilot_results.json', summary)
        checkpoint('complete')
        print('PILOT_COMPLETE', json.dumps(summary), flush=True)
        return 0
    finally:
        stream.close()


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    parser = argparse.ArgumentParser(); parser.add_argument('--model', required=True)
    args = parser.parse_args()
    try: sys.exit(run(args.model))
    except Exception:
        out = HERE / 'pilot/results' / f'{args.model}_OISST_x4_s10'
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / 'failure.json', {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'error': traceback.format_exc()})
        raise
