#!/usr/bin/env python3
"""Independent dense-reference audit of upstream and corrected SMM kernels."""
import datetime
import json
import os
from cloud_preflight_pft import HERE, load_smm, model_namespace, kernel_checks

import torch

major, minor = torch.cuda.get_device_capability()
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(1)
report = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
          'test_images_evaluated': 0, 'training_started': False}
for key, atomic in [('upstream', False), ('corrected', True)]:
    load_smm(torch, atomic=atomic)
    namespace = model_namespace(torch)
    report[key] = [kernel_checks(torch, namespace, require_pass=False) for _ in range(3)]
    report[key].append(kernel_checks(torch, namespace, require_pass=False, size=1024, topk=256))
    report[key + '_all_pass'] = all(q['pass'] for rep in report[key] for item in rep for q in item['checks'])
    (HERE / 'records/kernel_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in {'upstream','corrected'}}), flush=True)
assert report['corrected_all_pass'], report
