#!/usr/bin/env python3
"""Discarded training-only throughput/memory probe; never opens the test NPZ."""
import copy
import datetime
import json
import os
import random
import statistics
import sys
import time
import traceback
from cloud_preflight_pft import BASE, HERE, load_smm, model_namespace, sha, accumulated_l1

import numpy as np
import torch
from torch import nn


class WMSR01(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.network = model

    def forward(self, image):
        return self.network(image * 255.) / 255.


def main():
    assert json.loads((HERE / 'records/pft_preflight.json').read_text())['status'] == 'pass'
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_smm(torch, atomic=True)
    namespace = model_namespace(torch)
    sys.path.insert(0, str(BASE))
    from extended_models.factory import create_model
    selection = json.loads((HERE / 'candidate_selection.json').read_text())
    data = json.loads((BASE / 'records/formal/OISST_manifest.json').read_text())
    results = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'status': 'profiling', 'gpu': torch.cuda.get_device_name(),
               'test_images_evaluated': 0, 'formal_training_started': False, 'batches': [],
               'precision': {'dtype': 'float32', 'matmul_tf32': False, 'cudnn_tf32': False},
               'boundary': '3 warmup + 5 measured AdamW/L1/EMA updates per model/scale; all states discarded. Does not establish convergence or predictive quality.',
               'base_factory_sha256': sha(BASE / 'extended_models/factory.py')}
    out = HERE / 'records/model_profile.json'
    try:
        with np.load(BASE / data['splits']['train']['file']) as archive:
            for model_name in ['PFT', 'WMSR']:
                for scale in [2, 3, 4]:
                    random.seed(10); np.random.seed(10); torch.manual_seed(10); torch.cuda.manual_seed_all(10)
                    if model_name == 'PFT':
                        kwargs = dict(selection['official_network_and_training_configs'][str(scale)]['network_g'])
                        kwargs.pop('type')
                        model = namespace['PFT'](**kwargs).cuda().train()
                    else:
                        raw, _ = create_model({'model': 'WMSR', 'scale': scale, 'channels': 64})
                        model = WMSR01(raw).cuda().train()
                    ema = copy.deepcopy(model).eval().requires_grad_(False)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.9, .99), weight_decay=0.)
                    low = torch.from_numpy(((archive[f'lr_x{scale}_celsius'][:8] + 2) / 42).copy())[:, None].repeat(1, 3, 1, 1).cuda()
                    high = torch.from_numpy(((archive['hr_celsius'][:8] + 2) / 42).copy())[:, None].repeat(1, 3, 1, 1).cuda()
                    equivalence = None
                    if scale == 4:
                        model.zero_grad(set_to_none=True)
                        direct = model(low)
                        direct_loss = nn.functional.l1_loss(direct, high)
                        direct_loss.backward()
                        full_grad = {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                        direct_output, direct_value = direct.detach(), direct_loss.item()
                        del direct, direct_loss
                        model.zero_grad(set_to_none=True)
                        accumulated_value, accumulated_output = accumulated_l1(torch, model, low, high, micro_batch=2)
                        assert abs(direct_value - accumulated_value) <= 1e-6 + abs(direct_value) * 1e-5
                        torch.testing.assert_close(direct_output, accumulated_output, atol=1e-5, rtol=1e-4)
                        maximum = 0.
                        for n,p in model.named_parameters():
                            if n not in full_grad: assert p.grad is None; continue
                            torch.testing.assert_close(p.grad, full_grad[n], atol=2e-5, rtol=1e-3, msg=n)
                            maximum = max(maximum, (p.grad - full_grad[n]).abs().max().item())
                        equivalence = {'scale': 4, 'direct_batch': 8, 'micro_batch': 2,
                                       'forward_loss_and_parameter_gradients': 'pass',
                                       'loss_abs_difference': abs(direct_value - accumulated_value),
                                       'output_max_abs_difference': (direct_output - accumulated_output).abs().max().item(),
                                       'max_parameter_gradient_abs_difference': maximum,
                                       'boundary': 'Numerical execution check on training samples, no predictive test. Floating-point summation order differs.'}
                        del full_grad, direct_output, accumulated_output
                        model.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    seconds = []
                    for step in range(8):
                        torch.cuda.synchronize(); started = time.monotonic()
                        optimizer.zero_grad(set_to_none=True)
                        loss, prediction = accumulated_l1(torch, model, low, high, micro_batch=2)
                        if step == 0: assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                        optimizer.step()
                        with torch.no_grad():
                            for a, b in zip(ema.parameters(), model.parameters()): a.lerp_(b, .001)
                        torch.cuda.synchronize()
                        if step >= 3: seconds.append(time.monotonic() - started)
                    row = {'model': model_name, 'scale': scale, 'batch_size': 8,
                           'micro_batch': 2, 'gradient_accumulation': 4,
                           'direct_batch_vs_accumulation_check': equivalence,
                           'parameters': sum(p.numel() for p in model.parameters()),
                           'seconds_per_update': seconds, 'median_seconds_per_update': statistics.median(seconds),
                           'peak_allocated_mib_with_optimizer_and_ema': torch.cuda.max_memory_allocated() / 2**20,
                           'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
                           'finite_last_l1': loss}
                    results['batches'].append(row)
                    out.write_text(json.dumps(results, indent=2) + '\n')
                    print(json.dumps(row), flush=True)
                    del model, ema, optimizer, low, high, prediction, loss
                    torch.cuda.empty_cache()
        results['status'] = 'pass'
    except Exception:
        results.update(status='fail', error=traceback.format_exc())
        raise
    finally:
        results['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        out.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__': main()
