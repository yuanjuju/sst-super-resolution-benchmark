"""Check a real paired batch through CAModel before the bounded cloud run."""

import copy
import json
import os
from pathlib import Path
import sys
import time

root = Path('/root/fsas/sr-repro')
extension_root = root / 'CAMixerSR/codes/basicsr'
sys.path.insert(0, str(extension_root))
os.environ.setdefault('MPLCONFIGDIR', str(root / 'CAMixerSR/.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME', str(root / 'CAMixerSR/.cache/torch'))

import torch
import yaml
import basicsr
import archs  # Register this project's network.
import models  # Register this project's CAModel.
from basicsr.data import build_dataset
from basicsr.models import build_model
from torch.utils.data import default_collate

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
torch.set_num_threads(2)
print('BASICSR_IMPORT', basicsr.__file__, flush=True)
print('TORCH', torch.__version__, 'GPU', torch.cuda.get_device_name(0), flush=True)
assert str(Path(sys.prefix)) in basicsr.__file__, 'Use the isolated CAMixerSR environment'

opt = yaml.safe_load((root / 'train_camixersr_5000.yml').read_text())
opt.update(is_train=True, dist=False, rank=0, world_size=1)
data_opt = copy.deepcopy(opt['datasets']['train'])
data_opt.update(phase='train', scale=opt['scale'])
dataset = build_dataset(data_opt)
batch_size = data_opt['batch_size_per_gpu']
batch = default_collate([dataset[i] for i in range(batch_size)])
print('BATCH', list(batch['lq'].shape), list(batch['gt'].shape), flush=True)

model = build_model(opt)
assert model.__class__.__name__ == 'CAModel'
first_parameter = next(model.net_g.parameters())
initial = first_parameter.detach().clone()
model.feed_data(batch)
torch.cuda.reset_peak_memory_stats()
seconds = []
losses = []
for current_iter in range(1, 4):
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.update_learning_rate(current_iter, warmup_iter=-1)
    model.optimize_parameters(current_iter)
    torch.cuda.synchronize()
    seconds.append(time.perf_counter() - start)
    values = {name: float(value) for name, value in model.get_current_log().items()}
    assert {'l_pix', 'l_ratio'} <= values.keys()
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
    losses.append(values)
    print('PREFLIGHT_STEP', current_iter, values, 'seconds', seconds[-1], flush=True)

weight_change = (first_parameter.detach() - initial).abs().max().item()
assert weight_change > 0
assert torch.isfinite(first_parameter.grad).all()
model.test()
assert model.output.shape == batch['gt'].shape
assert torch.isfinite(model.output).all()
torch.cuda.synchronize()
result = {
    'gpu': torch.cuda.get_device_name(0),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
    'basicsr': basicsr.__version__,
    'model_type': model.__class__.__name__,
    'dataset_size': len(dataset),
    'lq_shape': list(batch['lq'].shape),
    'gt_shape': list(batch['gt'].shape),
    'step_seconds': seconds,
    'losses': losses,
    'max_parameter_change': weight_change,
    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    'ema_output_shape': list(model.output.shape),
    'note': 'Three diagnostic updates; discarded before the fresh 5000-step run.',
}
(root / 'camixersr_5090_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
print('PREFLIGHT_COMPLETE', json.dumps(result), flush=True)
