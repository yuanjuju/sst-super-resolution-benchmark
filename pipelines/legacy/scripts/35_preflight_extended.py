#!/usr/bin/env python3
"""Real training-batch forward/backward; publish ready markers only after finite gradients."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);args=p.parse_args()
    import torch,numpy as np
    sys.path.insert(0,str(ROOT));from extended_models.factory import create_model
    torch.set_num_threads(1)
    jobs=json.loads((ROOT/'records/extended_plan.json').read_text())['jobs'];seen=set()
    out=ROOT/'records/extended_ready';out.mkdir(exist_ok=True)
    for spec in jobs:
        key=spec['preflight_key']
        if spec['model'] not in args.models or key in seen:continue
        seen.add(key)
        if (out/(key+'.json')).exists():continue
        try:
            torch.manual_seed(10);torch.cuda.manual_seed_all(10)
            manifest=json.loads((ROOT/'records/formal'/f"{spec['dataset']}_manifest.json").read_text())
            with np.load(ROOT/manifest['splits']['train']['file']) as a:
                low=torch.from_numpy((a[f"lr_x{spec['scale']}_celsius"][:8]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
                high=torch.from_numpy((a['hr_celsius'][:8]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            model,details=create_model(spec);model=model.cuda();prediction=model(low)
            assert prediction.shape==high.shape
            loss=(prediction-high).abs().mean();loss.backward();torch.cuda.synchronize()
            assert torch.isfinite(prediction).all() and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
            active=sum(x.numel() for x in model.parameters() if x.grad is not None)
            result={'pass':True,'architecture_key':key,'example_spec':spec,'architecture':details,'real_batch':8,
                    'shape':list(prediction.shape),'parameters':sum(x.numel() for x in model.parameters()),
                    'parameters_with_gradient':active,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'seconds':time.monotonic()-start,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
            target=out/(key+'.json');temp=target.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(target)
            print('PREFLIGHT_PASS',spec['id'],result['parameters'],active,result['seconds'],flush=True)
            del model,low,high,prediction,loss;torch.cuda.empty_cache()
        except Exception:
            error=traceback.format_exc();print('PREFLIGHT_FAILED',spec['id'],error,flush=True)
            (ROOT/'records'/f'preflight_error_{key}.txt').write_text(error)
    print('PREFLIGHT_BATCH_FINISHED',flush=True)
if __name__=='__main__':main()
