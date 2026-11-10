from pathlib import Path
import sys,json,importlib.util,os,time,datetime
c=Path('/root/wmsr-repro/strong_models/pft_20261109')
sys.path.insert(0,str(c))
import torch,numpy as np
from cloud_preflight_pft import model_namespace
spec=importlib.util.spec_from_file_location('smm_cuda_atomic',c/'.cache/smm_cuda_atomic/smm_cuda_atomic.so')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);sys.modules['smm_cuda']=mod
namespace=model_namespace(torch)
s=json.loads((c/'candidate_selection.json').read_text());kwargs=dict(s['official_network_and_training_configs']['4']['network_g']);kwargs.pop('type')
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=namespace['PFT'](**kwargs).cuda()
ckpt=torch.load(c/'pilot/results/PFT_OISST_x4_s10/checkpoints/pilot_final.pt',map_location='cpu',weights_only=True)
model.load_state_dict(ckpt['model']);del ckpt
m=json.loads(Path('/root/wmsr-repro/records/formal/OISST_manifest.json').read_text())
with np.load(Path('/root/wmsr-repro')/m['splits']['train']['file']) as a:low=torch.from_numpy(((a['lr_x4_celsius'][:8]+2)/42).copy())[:,None].repeat(1,3,1,1).cuda()
rows=[]
with torch.inference_mode():
 for i in range(8):
  model.train();a=model(low[i:i+1]).clone()
  model.eval();b=model(low[i:i+1]).clone()
  delta=(a-b).abs()
  rows.append({'training_sample_index':i,'max_abs_difference_normalized':delta.max().item(),'mean_abs_difference_normalized':delta.mean().item(),'max_abs_difference_celsius':delta.max().item()*42})
print(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'PFT full raw1000','samples':'first8 original fitting800, inference only, no optimizer updates','rows':rows,'boundary':'Mode consistency diagnostic on8 training images, not a re-evaluation or rewriting of the frozen80 validation results; does not identify optimization causes.'}))
