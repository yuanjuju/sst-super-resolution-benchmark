"""Pinned-source architectures, explicit variants, and public-block ablations."""
from pathlib import Path
from types import SimpleNamespace
import sys,copy,math
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path('/root/wmsr-repro')

class Normalized255(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,x):return self.model(x/255)*255

def create_model(spec):
    name=spec['model'];scale=spec['scale'];details={}
    if name=='EDSR':
        from .vendor.edsr import EDSR
        kwargs=dict(n_resblocks=16,n_feats=64,scale=[scale],res_scale=1,n_colors=3,rgb_range=255)
        model=EDSR(SimpleNamespace(**kwargs));details={'variant':'official EDSR baseline r16f64','kwargs':kwargs}
    elif name=='ELAN':
        import yaml
        from .vendor.elan import ELAN
        kwargs=yaml.safe_load((ROOT/'external/ELAN/configs/elan_light_x4.yml').read_text());kwargs['scale']=scale
        model=ELAN(SimpleNamespace(**kwargs));details={'variant':'official ELAN-light','kwargs':kwargs}
    elif name=='DAT':
        from .vendor.dat import DAT
        # These are the architecture class defaults; do not reverse-engineer a model by parameter count.
        kwargs=dict(upscale=scale,img_size=64,in_chans=3,img_range=1.,embed_dim=180,
                    split_size=[2,4],depth=[2,2,2,2],num_heads=[2,2,2,2],expansion_factor=4.,
                    resi_connection='1conv',upsampler='pixelshuffle')
        model=Normalized255(DAT(**kwargs));details={'variant':'official DAT architecture defaults','kwargs':kwargs}
    elif name=='SAFMN':
        from .vendor.safmn import SAFMN
        kwargs=dict(dim=128,n_blocks=16,ffn_scale=2.,upscaling_factor=scale)
        model=Normalized255(SAFMN(**kwargs));details={'variant':'SAFMN-L 128 channels/16 blocks','kwargs':kwargs}
    elif name=='MambaIR':
        import yaml
        from .vendor.mambair import MambaIR
        kwargs=yaml.safe_load((ROOT/'external/MambaIR/options/train/mambair/train_MambaIR_lightSR_x4.yml').read_text())['network_g']
        kwargs.pop('type');kwargs['upscale']=scale
        model=Normalized255(MambaIR(**kwargs));details={'variant':'official MambaIR lightSR configuration','kwargs':kwargs}
    elif name=='CRAN':
        from .cran_paper import CRANPaper
        model=CRANPaper(scale=scale)
        details={'variant':'paper-guided CRAN reconstruction, not an official released model',
                 'kwargs':{'groups':10,'plain_residual_blocks_per_group':19,'context_reasoning_blocks_per_group':1,'channels':64},
                 'boundary':'Architecture from CRAN section3.5; contextual convolution adapted from the pinned public third-party/CGC sources. Not the unidentified CRAN variant in WMSR.'}
    elif name in ['WMSR','WMSR_ABLATION','WMSR_GROUPED']:
        sys.path.insert(0,str(ROOT/'upstream'))
        import yaml
        from models.wmsr_network import create_model as upstream_create
        kwargs=yaml.safe_load((ROOT/'upstream/configs/wmsr_x4.yml').read_text());kwargs['scale']=scale
        kwargs['c_wmsr']=spec.get('channels',64)
        model=upstream_create(SimpleNamespace(**kwargs))
        details={'variant':'pinned public WMSR block/network','channels':kwargs['c_wmsr'],'actual_wam_blocks':4}
        if name=='WMSR_ABLATION':
            from .wmsr_variants import apply_ablation
            model=apply_ablation(model,spec['case']);details['case']=spec['case']
            details['boundary']='Ablations of released blocks; no-DWT branch/fusion interpretation is explicit, not author-confirmed TableVI code.'
        if name=='WMSR_GROUPED':
            from .wmsr_variants import GroupedWMSR
            model=GroupedWMSR(model,spec['groups'],spec['blocks'])
            details.update(groups=spec['groups'],blocks=spec['blocks'],actual_wam_blocks=spec['groups']*spec['blocks'],
                           boundary='Explicit residual-group assembly of public WAMs. Not the failed bottleneck reconstruction and not a verified original paper architecture.')
    else:raise ValueError(name)
    return model,details
