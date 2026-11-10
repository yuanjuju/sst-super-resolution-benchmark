#!/usr/bin/env python3
"""Kernel/reference gradients plus full200image replay of the saved4090 checkpoint."""
import json,math,sys
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def main():
    import torch,numpy as np
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn,selective_scan_ref
    torch.set_num_threads(1);torch.manual_seed(123)
    u=torch.randn(2,8,64,device='cuda',requires_grad=True);delta=torch.rand_like(u,requires_grad=True)
    A=-torch.exp(torch.randn(8,16,device='cuda'));B=torch.randn(2,16,64,device='cuda');C=torch.randn_like(B);D=torch.randn(8,device='cuda')
    out=selective_scan_fn(u,delta,A,B,C,D);ref=selective_scan_ref(u,delta,A,B,C,D)
    error=(out-ref).abs().max().item();assert torch.allclose(out,ref,atol=2e-4,rtol=2e-4)
    out.square().mean().backward();assert torch.isfinite(u.grad).all() and torch.isfinite(delta.grad).all()
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'upstream'))
    from extended_models.factory import create_model
    from utils import rgb_to_ycbcr
    run=ROOT/'results/public_control/OISST_x4_seed10';old=json.loads((run/'final_results.json').read_text())
    checkpoint=torch.load(run/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    model,_=create_model({'model':'WMSR','scale':4});assert set(checkpoint['model_state_dict'])==set(model.state_dict())
    model.load_state_dict(checkpoint['model_state_dict']);model=model.cuda().eval()
    manifest=json.loads((ROOT/'records/formal/OISST_manifest.json').read_text())
    with np.load(ROOT/manifest['splits']['test']['file']) as data:hr=data['hr_celsius'].copy();lr=data['lr_x4_celsius'].copy()
    scores=[];mse=[]
    with torch.inference_mode():
        for i in range(0,200,8):
            low=torch.from_numpy((lr[i:i+8]+2)/42*255)[:,None].repeat(1,3,1,1).cuda()
            high=torch.from_numpy((hr[i:i+8]+2)/42*255)[:,None].repeat(1,3,1,1).cuda();prediction=model(low)
            error_c=((prediction.mean(1)-high.mean(1))/255*42)[:,4:-4,4:-4]
            mse.extend(error_c.square().flatten(1).mean(1).tolist())
            a=rgb_to_ycbcr(prediction.clamp(0,255))[:,0:1,4:-4,4:-4];b=rgb_to_ycbcr(high.clamp(0,255))[:,0:1,4:-4,4:-4]
            scores.extend((-10*torch.log10(((a.double()-b.double())/255).square().flatten(1).mean(1).clamp_min(1e-16))).tolist())
    psnr=sum(scores)/200;rmse=math.sqrt(sum(mse)/200);difference=psnr-old['test_wmsr']['psnr_y_db']
    assert len(scores)==200 and abs(difference)<.02,'New CUDA environment does not agree with saved-checkpoint replay'
    result={'pass':True,'selective_scan_max_abs_difference':error,'finite_gradients':True,'replay_count':200,
            'saved4090_psnr_y_db':old['test_wmsr']['psnr_y_db'],'replayed5090_psnr_y_db':psnr,
            'replayed_rmse_c':rmse,'delta_psnr_db':difference,'acceptance_tolerance_db':.02,
            'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'meaning':'Checks kernel and checkpoint replay, not a new training or paper reproduction.'}
    (ROOT/'records/blackwell_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('BLACKWELL_ACCEPTANCE_PASS',json.dumps(result),flush=True)
if __name__=='__main__':main()
