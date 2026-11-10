from pathlib import Path
import sys,json,torch,numpy as np,datetime
p=Path('/root/wmsr-repro/strong_models/candidate_screen_20261109');sys.path.insert(0,str(p))
from train_screen import evaluate
base=Path('/root/wmsr-repro');m=json.loads((base/'records/formal/OISST_manifest.json').read_text())
with np.load(base/m['splits']['train']['file']) as a:lr=((a['lr_x4_celsius']+2)/42).copy();hr=((a['hr_celsius']+2)/42).copy()
s=json.loads((p/'development_splits/OISST.json').read_text())
class Bicubic(torch.nn.Module):
 def forward(self,x):return torch.nn.functional.interpolate(x,scale_factor=4,mode='bicubic',align_corners=False,antialias=False)
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
v,rows=evaluate(Bicubic().cuda(),lr,hr,s['validation_indices'],m['splits']['train']['samples'],4)
d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Bicubic, no learned parameters','split':'same training-only80','final_test_images_evaluated':0,'metrics':v,'samples':rows}
(p/'records/bicubic_development80.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(v))
