#!/usr/bin/env python3
"""Bounded ESC compatibility checks using training images only; no optimizer updates."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
from factory_esc import create_model

HERE = Path(__file__).resolve().parent
BASE = Path('/root/wmsr-repro')


def main():
    assert str(HERE).startswith('/root/wmsr-repro/strong_models/')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status':'checking',
              'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__, 'optimizer_updates':0,
              'final_test_images_evaluated':0, 'attention_backend':'official Naive, unchanged full relative-position bias',
              'precision':{'dtype':'float32','matmul_tf32':False,'cudnn_tf32':False}, 'batches':[]}
    try:
        selection = json.loads((HERE / 'selected_candidate.json').read_text())
        path = BASE / 'records/formal/OISST_manifest.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == selection['OISST_manifest_sha256']
        data = json.loads(path.read_text())
        file = BASE / data['splits']['train']['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == data['splits']['train']['sha256']
        with np.load(file) as a:
            for scale in [2,3,4]:
                random.seed(10); np.random.seed(10); torch.manual_seed(10); torch.cuda.manual_seed_all(10)
                start = time.monotonic()
                model = create_model(scale).cuda().train()
                low = torch.from_numpy(((a[f'lr_x{scale}_celsius'][:8] + 2) / 42).copy())[:,None].repeat(1,3,1,1).cuda()
                high = torch.from_numpy(((a['hr_celsius'][:8] + 2) / 42).copy())[:,None].repeat(1,3,1,1).cuda()
                torch.cuda.reset_peak_memory_stats()
                losses = []
                for i in range(0,8,2):
                    output = model(low[i:i+2])
                    assert output.shape == high[i:i+2].shape == (2,3,96,96)
                    loss = torch.nn.functional.l1_loss(output, high[i:i+2]) / 4
                    assert torch.isfinite(loss)
                    loss.backward();losses.append(loss.item())
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                model.zero_grad(set_to_none=True)
                with torch.inference_mode():
                    model.train(); train_output = model(low[:1]).clone()
                    model.eval(); eval_output = model(low[:1]).clone()
                torch.testing.assert_close(train_output,eval_output,atol=2e-5,rtol=1e-4)
                torch.cuda.synchronize()
                row = {'scale':scale,'parameters':sum(p.numel() for p in model.parameters()),'effective_batch':8,
                       'micro_batch':2,'accumulations':4,'LR':list(low.shape),'HR':[8,3,96,96],
                       'loss_is_finite':True,'parameter_gradients_are_finite':True,'sum_normalized_l1':sum(losses),
                       'eval_batch':1,'train_eval_output_max_abs_difference':(train_output-eval_output).abs().max().item(),
                       'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                       'construction_and_first_batch_seconds':time.monotonic()-start,
                       'timing_boundary':'Includes CPU bias-index construction and first CUDA execution, not steady optimizer throughput.'}
                result['batches'].append(row)
                print(json.dumps(row),flush=True)
                del model,low,high,output,loss,train_output,eval_output
                torch.cuda.empty_cache()
        result['status'] = 'pass'
    except Exception:
        result.update(status='fail',error=traceback.format_exc())
        raise
    finally:
        result['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (HERE/'records/esc_preflight.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':main()
