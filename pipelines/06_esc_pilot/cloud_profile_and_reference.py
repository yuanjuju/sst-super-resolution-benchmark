#!/usr/bin/env python3
"""Discarded x4 resource probe and frozen WMSR validation recheck at batch1."""
import copy
import datetime
import json
import random
import statistics
import sys
import time
import traceback
import numpy as np
import torch
from screen_helpers import HERE,BASE,accumulated_l1,sha
from train_screen import make_model,batch_tensors,evaluate


class WMSR01(torch.nn.Module):
    def __init__(self,network):super().__init__();self.network=network
    def forward(self,x):return self.network(x*255.)/255.


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'checking','test200_evaluated':0,'pilot_training_started':False}
    try:
        protocol=json.loads((HERE/'pilot_protocol.json').read_text())
        reference=__import__('pathlib').Path(protocol['baseline_reference'])
        assert sha(reference/'config.json')==protocol['baseline_reference_config_sha256']
        ref_config=json.loads((reference/'config.json').read_text())
        controls=['dataset','scale','seed','updates','development_manifest_sha256','batch_size','micro_batch','accumulation_steps','optimizer','loss','warmup_updates','scheduler','ema_decay','validation_every','augmentations','precision','initialization']
        assert all(protocol[k]==ref_config['protocol'][k] for k in controls)
        result['equal_control_fields']=controls
        manifest=json.loads((BASE/'records/formal/OISST_manifest.json').read_text())
        with np.load(BASE/manifest['splits']['train']['file']) as a:
            low=((a['lr_x4_celsius']+2)/42).copy();high=((a['hr_celsius']+2)/42).copy()
        split=json.loads((HERE/'development_splits/OISST.json').read_text())
        random.seed(10);np.random.seed(10);torch.manual_seed(10);torch.cuda.manual_seed_all(10)
        model=make_model('ESC',4).cuda().train()
        x,y=batch_tensors(low,high,list(range(8)))
        output=model(x);loss=torch.nn.functional.l1_loss(output,y);loss.backward()
        expected={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        expected_output=output.detach();expected_loss=loss.item();del output,loss
        model.zero_grad(set_to_none=True)
        from factory_esc import cache_rpe_indices
        cache_rpe_indices(model,'cuda')
        value,output=accumulated_l1(torch,model,x,y,2)
        torch.testing.assert_close(output,expected_output,atol=2e-5,rtol=1e-4)
        gradient_error=0.
        for n,p in model.named_parameters():
            if n in expected:
                torch.testing.assert_close(p.grad,expected[n],atol=2e-5,rtol=1e-3,msg=n)
                gradient_error=max(gradient_error,(p.grad-expected[n]).abs().max().item())
        assert abs(value-expected_loss)<1e-6
        result['accumulation_equivalence']={'status':'pass','scale':4,'direct_batch':8,'micro_batch':2,'derived_RPE_index_device':'CPU direct baseline vs CUDA accumulated; identical integer values','loss_abs_difference':abs(value-expected_loss),'max_output_abs_difference':(output-expected_output).abs().max().item(),'max_parameter_gradient_abs_difference':gradient_error}
        del expected,expected_output,output
        model.zero_grad(set_to_none=True)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.99),weight_decay=0.)
        torch.cuda.reset_peak_memory_stats();seconds=[]
        for i in range(8):
            torch.cuda.synchronize();start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            value,output=accumulated_l1(torch,model,x,y,2);optimizer.step()
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.001)
            torch.cuda.synchronize()
            if i>=3:seconds.append(time.monotonic()-start)
        result['profile']={'scale':4,'measured_seconds_per_update':seconds,'median_seconds_per_update':statistics.median(seconds),'peak_allocated_mib_with_optimizer_and_ema':torch.cuda.max_memory_allocated()/2**20,'discarded_optimizer_updates':8,'boundary':'3warmup +5timed updates on8 training images; states discarded, no candidate predictive score'}
        del model,ema,optimizer,x,y,output
        torch.cuda.empty_cache()
        sys.path.insert(0,str(BASE))
        from extended_models.factory import create_model
        raw,_=create_model({'model':'WMSR','scale':4,'channels':64})
        model=WMSR01(raw).cuda().eval()
        checkpoint=torch.load(reference/'checkpoints/pilot_final.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['model']);del checkpoint
        metrics,rows=evaluate(model,low,high,split['validation_indices'],manifest['splits']['train']['samples'],4)
        old=json.loads((reference/'pilot_results.json').read_text())['last_validation']['raw']
        differences={k:abs(metrics[k]-old[k]) for k in metrics}
        assert differences['count']==0 and differences['psnr_y_db']<1e-3 and differences['rmse_c']<1e-5 and differences['ssim_y']<1e-5
        result['reference_recheck']={'model':'WMSR','checkpoint_sha256':sha(reference/'checkpoints/pilot_final.pt'),'original_eval_batch':2,'new_eval_batch':1,'metrics':metrics,'absolute_difference_from_original':differences,'samples':rows,'old_results_modified':False}
        result['status']='pass'
    except Exception:
        result.update(status='fail',error=traceback.format_exc());raise
    finally:
        result['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (HERE/'records/esc_profile_and_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reference_recheck'}),flush=True)


if __name__=='__main__':main()
