#!/usr/bin/env python3
"""Validate both paired configurations on real data; discard diagnostic updates."""
import copy
import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml
import basicsr
from basicsr.archs import build_network
from basicsr.data import build_dataset, build_dataloader
from basicsr.models import build_model
from basicsr.utils.options import parse_options


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '/root/fsas/sr-repro')
    suite = root / 'safmn_fft_pair_30k'
    project = root / 'SAFMN'
    assert Path(basicsr.__file__).resolve().is_relative_to(project.resolve())
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    configs = [suite / f'configs/train_{name}_30000.yml' for name in ('fft', 'nofft')]
    raw = [yaml.safe_load(path.read_text()) for path in configs]
    left, right = [copy.deepcopy(value) for value in raw]
    for value in (left, right):
        value.pop('name')
        value['train'].pop('fft_opt', None)
    assert left == right, 'Unexpected paired configuration difference'
    assert raw[0]['train']['fft_opt']['loss_weight'] == 0.05
    assert raw[1]['train'].get('fft_opt') is None
    init_path = suite / 'common_init.pth'
    if init_path.exists():
        raise RuntimeError('Refusing to replace existing common initializer')
    torch.manual_seed(10)
    initial = build_network(copy.deepcopy(raw[0]['network_g'])).cpu().state_dict()
    torch.save({'params': initial, 'params_ema': {k: v.clone() for k, v in initial.items()}}, init_path)
    results = []
    for name, config in zip(('fft', 'nofft'), configs):
        sys.argv = ['preflight', '-opt', str(config)]
        opt, _ = parse_options(str(project), is_train=True)
        model = build_model(opt)
        assert all(torch.equal(model.net_g.state_dict()[k].cpu(), v) for k, v in initial.items())
        assert all(torch.equal(model.net_g_ema.state_dict()[k].cpu(), v) for k, v in initial.items())
        dataset_opt = copy.deepcopy(opt['datasets']['train'])
        dataset_opt['num_worker_per_gpu'] = 0  # diagnostics only; formal configs retain 8
        dataset = build_dataset(dataset_opt)
        loader = build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=opt['manual_seed'])
        iterator = iter(loader)
        batch = next(iterator)
        assert tuple(batch['lq'].shape) == (64, 3, 64, 64)
        assert tuple(batch['gt'].shape) == (64, 3, 256, 256)
        before = next(model.net_g.parameters()).detach().clone()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        seconds = []
        for iteration in range(1, 4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model.feed_data(batch)
            model.optimize_parameters(iteration)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            row = {k: float(v) for k, v in model.get_current_log().items()}
            assert all(math.isfinite(v) for v in row.values())
            losses.append(row)
        changed = float((next(model.net_g.parameters()).detach() - before).abs().max())
        assert changed > 0
        model.test()
        assert tuple(model.output.shape) == (64, 3, 256, 256)
        assert torch.isfinite(model.output).all()
        result = {'run': name, 'initial_weights_match': True, 'losses': losses,
                  'step_compute_seconds': seconds, 'max_parameter_change': changed,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                  'lq_shape': list(batch['lq'].shape), 'gt_shape': list(batch['gt'].shape),
                  'ema_output_shape': list(model.output.shape), 'diagnostic_workers': 0}
        results.append(result)
        print(json.dumps(result), flush=True)
        del iterator, loader, dataset, model, batch, before
        gc.collect()
        torch.cuda.empty_cache()
    record = {'status': 'passed', 'checked_at': datetime.datetime.now().astimezone().isoformat(),
              'torch': torch.__version__, 'basicsr_file': basicsr.__file__,
              'basicsr_checkout_version': getattr(basicsr, '__version__', None),
              'gpu': torch.cuda.get_device_name(0), 'common_init_path': str(init_path),
              'common_init_sha256': hashlib.sha256(init_path.read_bytes()).hexdigest(),
              'config_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in configs},
              'initial_parameters': sum(v.numel() for v in initial.values()), 'runs': results,
              'diagnostic_updates_discarded': True,
              'notes': ['Both formal runs load the same seeded random initializer, not trained weights.',
                        'The original 500k cosine schedule is retained; this is a 30k single-seed exploration.']}
    (suite / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    print('PREFLIGHT_PASSED', flush=True)


if __name__ == '__main__':
    main()
