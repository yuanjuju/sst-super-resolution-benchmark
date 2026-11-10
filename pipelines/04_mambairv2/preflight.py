#!/usr/bin/env python3
"""Cloud-only full-data integrity and three-scale CUDA checks; does not start training."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path('/root/wmsr-repro')
HERE = ROOT / 'recent_models'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from focal_frequency_loss import FocalFrequencyLoss
    sys.path.insert(0, str(ROOT))
    from recent_models.factory import create_model
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    plan = json.loads((HERE / 'plan.json').read_text())
    proof = {'started_at_unix': time.time(), 'gpu': torch.cuda.get_device_name(),
             'torch': torch.__version__, 'cuda': torch.version.cuda, 'data': {}, 'shapes': []}
    for spec in plan['jobs']:
        ref = ROOT / 'results/extended' / spec['reference_id'] / 'config.json'
        assert digest(ref) == spec['reference_config_sha256'], ref
        manifest_path = ROOT / 'records/formal' / (spec['dataset'] + '_manifest.json')
        assert digest(manifest_path) == spec['data_manifest_sha256'], manifest_path
        manifest = json.loads(manifest_path.read_text())
        if spec['dataset'] not in proof['data']:
            assert manifest['count'] == 1000
            for name, part in manifest['splits'].items():
                assert part['count'] == {'train': 800, 'test': 200}[name]
                assert digest(ROOT / part['file']) == part['sha256'], part['file']
            proof['data'][spec['dataset']] = '1000 HR;800/200; both NPZ SHA256 matched'
    for scale in [4, 3, 2]:
        spec = next(x for x in plan['jobs'] if x['dataset'] == 'OISST' and x['scale'] == scale)
        manifest = json.loads((ROOT / 'records/formal/OISST_manifest.json').read_text())
        with np.load(ROOT / manifest['splits']['train']['file']) as arrays:
            low = torch.from_numpy((arrays[f'lr_x{scale}_celsius'][:8] + 2) / 42 * 255)[:, None].repeat(1, 3, 1, 1).cuda()
            high = torch.from_numpy((arrays['hr_celsius'][:8] + 2) / 42 * 255)[:, None].repeat(1, 3, 1, 1).cuda()
        random.seed(10); np.random.seed(10); torch.manual_seed(10); torch.cuda.manual_seed_all(10)
        model, details = create_model(spec)
        model = model.cuda().train()
        torch.cuda.reset_peak_memory_stats()
        output = model(low)
        assert output.shape == high.shape == (8, 3, 96, 96)
        loss = .1 * F.l1_loss(output, high) + .9 * FocalFrequencyLoss(loss_weight=1, alpha=1)(output, high)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        model.eval()
        with torch.inference_mode():
            torch.manual_seed(10); torch.cuda.manual_seed_all(10); one = model(low)
            torch.manual_seed(10); torch.cuda.manual_seed_all(10); two = model(low)
        difference = (one - two).abs().max().item()
        assert difference < 1e-4, difference
        row = {'scale': scale, 'batch': 8, 'output_shape': list(output.shape),
               'parameters': sum(p.numel() for p in model.parameters()), 'finite_loss': loss.item(),
               'finite_gradients': True, 'seeded_eval_max_abs_difference': difference,
               'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
        proof['shapes'].append(row)
        print(json.dumps(row), flush=True)
        del model, low, high, output, one, two, grads, loss
        torch.cuda.empty_cache()
    proof['status'] = 'pass'
    proof['boundary'] = 'Integrity and CUDA checks only, no SST test scores or training completion claimed.'
    (HERE / 'records').mkdir(exist_ok=True)
    (HERE / 'records/preflight.json').write_text(json.dumps(proof, indent=2) + '\n')


if __name__ == '__main__':
    main()
