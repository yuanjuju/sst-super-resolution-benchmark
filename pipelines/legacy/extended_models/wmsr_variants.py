"""Declared approximations for TableVI and residual-group assembly of released WAMs."""
import copy
import torch
from torch import nn
from models.wmsr_block import DEConv,WAM

class Zero(nn.Module):
    def forward(self,x):return torch.zeros_like(x)

class VariantWAM(nn.Module):
    def __init__(self,block,case):
        super().__init__();channels=block.ln_2.normalized_shape[0]
        self.use_low=case not in [3,5,7];self.use_high=case not in [2,5,6]
        self.use_wavelet=case not in [4,6,7]
        self.dwt=block.dwt;self.iwt=block.iwt;self.norm=block.ln_2
        self.low=block.lfssm if self.use_low else None
        if case==9:self.low.self_attention=Zero()
        self.high=block.wsa if self.use_high else None
        if self.use_high and not self.use_wavelet:self.high=DEConv(channels)
        if case==8:self.high=copy.deepcopy(block.wsa.conv1_5)
        # Retain the same unrelated, unused source branches in all conditions.
        self.source_unused=nn.ModuleDict({'conv_blk':block.conv_blk,'conv':block.conv})
        self.skip_scale2=block.skip_scale2
    def forward(self,x):
        if self.use_wavelet:
            low,high=self.dwt(x)
            if self.low is not None:low=self.low(low)
            if self.high is not None:high=self.high(high)+high
            output=self.iwt(torch.cat([low,high],dim=1))
        else:
            branches=[]
            if self.low is not None:branches.append(self.low(x))
            if self.high is not None:branches.append(self.high(x)+x)
            output=sum(branches)/len(branches)
        return self.norm(output.permute(0,2,3,1)).permute(0,3,1,2).contiguous()

def apply_ablation(model,case):
    assert case in range(1,10)
    model.body=nn.Sequential(*[VariantWAM(block,case) for block in model.body])
    return model

class ResidualGroup(nn.Module):
    def __init__(self,channels,blocks):
        super().__init__();self.blocks=nn.Sequential(*[WAM(hidden_dim=channels) for _ in range(blocks)])
        self.refine=nn.Conv2d(channels,channels,3,padding=1)
    def forward(self,x):return x+self.refine(self.blocks(x))

class GroupedWMSR(nn.Module):
    def __init__(self,source,groups,blocks):
        super().__init__();self.source=source;channels=source.c_wmsr
        self.source.body=nn.Sequential(*[ResidualGroup(channels,blocks) for _ in range(groups)],nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return self.source(x)
