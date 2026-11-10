#!/usr/bin/env python3
"""Re-evaluate the recovered OISSTx4 WMSR checkpoint without changing old files."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/root/wmsr-repro')


def main():
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'upstream'))
    from extended_models.factory import create_model
    from utils import rgb_to_ycbcr
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    folder = ROOT / 'results/extended/compare_WMSR_OISST_x4_s10'
    saved = json.loads((folder / 'final_results.json').read_text())
    checkpoint = folder / 'checkpoints/final.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == saved['checkpoint_sha256']
    model, _ = create_model(saved['spec'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model = model.cuda().eval()
    manifest = json.loads((ROOT / 'records/formal/OISST_manifest.json').read_text())
    part = manifest['splits']['test']
    assert hashlib.sha256((ROOT / part['file']).read_bytes()).hexdigest() == part['sha256']
    with np.load(ROOT / part['file']) as data:
        low = torch.from_numpy((data['lr_x4_celsius'] + 2) / 42 * 255)[:, None].repeat(1, 3, 1, 1)
        high = torch.from_numpy((data['hr_celsius'] + 2) / 42 * 255)[:, None].repeat(1, 3, 1, 1)
    values = []
    with torch.inference_mode():
        for begin in range(0, 200, 8):
            prediction = model(low[begin:begin + 8].cuda())
            y_sr = rgb_to_ycbcr(prediction.clamp(0, 255))[:, :1, 4:-4, 4:-4]
            y_hr = rgb_to_ycbcr(high[begin:begin + 8].cuda().clamp(0, 255))[:, :1, 4:-4, 4:-4]
            score = -10 * torch.log10(((y_sr.double() - y_hr.double()) / 255).square().flatten(1).mean(1).clamp_min(1e-16))
            values.extend(score.tolist())
    actual = sum(values) / len(values)
    expected = saved['test_model']['psnr_y_db']
    difference = actual - expected
    assert len(values) == 200 and abs(difference) <= .02, difference
    record = {'status': 'pass', 'count': 200, 'saved_psnr_y_db': expected,
              'current_psnr_y_db': actual, 'difference_db': difference,
              'checkpoint_sha256': saved['checkpoint_sha256'],
              'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    path = ROOT / 'recent_models/records/reference_replay.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
