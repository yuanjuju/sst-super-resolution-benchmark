#!/usr/bin/env python3
"""Uniform actual-model profiling with explicitly approximate selective-scan FLOPs."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    import numpy as np,torch
    sys.path.insert(0,str(ROOT));from extended_models.factory import create_model
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'records/extended_plan.json').read_text());seen=set();records=[];smoked=set()
    for spec in plan['jobs']:
        key=spec['preflight_key']
        if args.smoke:
            if spec['model'] not in ['EDSR','MambaIR','CRAN','WMSR_GROUPED'] or spec['model'] in smoked:continue
            smoked.add(spec['model'])
        if key in seen:continue
        seen.add(key);scale=spec['scale'];manifest=json.loads((ROOT/'records/formal'/f"{spec['dataset']}_manifest.json").read_text())
        with np.load(ROOT/manifest['splits']['train']['file']) as a:low=torch.from_numpy((a[f'lr_x{scale}_celsius'][:1]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
        model,details=create_model(spec);model=model.cuda().eval();scan_flops=[0];hooks=[]
        def scan_hook(module,inputs,output):
            batch,height,width,_=inputs[0].shape
            scan_flops[0]+=9*batch*height*width*4*module.d_inner*module.d_state+batch*height*width*4*module.d_inner
        for module in model.modules():
            if module.__class__.__name__=='SS2D':hooks.append(module.register_forward_hook(scan_hook))
        with torch.inference_mode():
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,with_flops=True) as profile:model(low);torch.cuda.synchronize()
        native=sum(event.flops for event in profile.key_averages());zero=[event.key for event in profile.key_averages() if not event.flops]
        for hook in hooks:hook.remove()
        with torch.inference_mode():
            for _ in range(5):model(low)
            torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):model(low)
            end.record();torch.cuda.synchronize();milliseconds=start.elapsed_time(end)/20
        records.append({'key':key,'spec':spec,'architecture':details,'parameters':sum(p.numel() for p in model.parameters()),
                        'input_shape':list(low.shape),'output_hr_shape':[1,3,96,96],'profiled_conv_matmul_flops':native,
                        'estimated_selective_scan_flops':scan_flops[0],'estimated_total_flops':native+scan_flops[0],
                        'zero_reported_operations':zero,'inference_ms_batch1':milliseconds,
                        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
                        'boundary':'Estimate: PyTorch-supported conv/matmul plus9*B*L*(4D)*N selective-scan arithmetic; other noncounted operations/exponentials omitted. Not an exact reproduction of paper FLOP tooling.'})
        del model,low;torch.cuda.empty_cache();print('PROFILED',spec['id'],native+scan_flops[0],flush=True)
    out=ROOT/'results/extended_summary';out.mkdir(exist_ok=True)
    target=ROOT/'records/profile_smoke.json' if args.smoke else out/'profiles.json'
    if args.smoke:
        for row in records:row['boundary']+=' SMOKE TEST WHILE TRAINING; latency is not a benchmark result.'
    target.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
