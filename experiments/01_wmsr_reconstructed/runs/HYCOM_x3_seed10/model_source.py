"""Paper-topology reconstruction; ambiguities are declared in ARCHITECTURE_NOTES.

Not the released four-WAM network. SS2D and Haar operations are reused unchanged
from the pinned author repository; missing groups, residuals and LF/HF gates are
reconstructed from Fig.1-3. Report parameter counts independently.
"""
import torch
from torch import nn
from torch.nn import functional as F
from models.wmsr_block import SS2D,MeanShift,dwt_init,iwt_init

ARCHITECTURE_NOTES={
    "groups":6,"blocks_per_group":4,"channels":64,
    "group_choice":"Fig.1 and TableII agree on6x4; one text sentence swaps6and4",
    "internal_projection":"64 outer channels, 32-channel wavelet bottleneck; 1x1 entry and group refinement. Inferred missing dimensions, not author-confirmed",
    "parameter_constraint":"Bottleneck interpretation approximately0.964M atx4, near reported0.977M; closeness is a consistency check, not proof of original implementation",
    "ss2d":"Author SS2D, d_state16, expand2; expansion not stated in paper",
    "hfem":"Fig3 order: linear+DWConv low and high branches, sigmoid high gate, then PDC; Eq14 places PDC differently",
    "pdc":"Shared across three high-frequency subbands; depthwise branches follow four differential branches of author code",
    "subband_fusion":"Concatenate LL/HL/LH/HH for valid IDWT dimensions; grouped1x1 refinement preserves four subbands",
    "normalization":"Post-IDWT LayerNorm retained from public WAM to reconcile omitted detail",
    "parameter_count":"Measured independently; paper977K cannot be guaranteed from incomplete dimensions",
}
class DifferenceConv(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.channels=channels
        self.vanilla=nn.Parameter(torch.empty(channels,1,3,3))
        self.central=nn.Parameter(torch.empty(channels,1,3,3))
        self.angular=nn.Parameter(torch.empty(channels,1,3,3))
        self.horizontal=nn.Parameter(torch.empty(channels,1,3))
        self.vertical=nn.Parameter(torch.empty(channels,1,3))
        self.bias=nn.Parameter(torch.zeros(5,channels))
        for value in [self.vanilla,self.central,self.angular,self.horizontal,self.vertical]:
            nn.init.kaiming_normal_(value,mode="fan_in",nonlinearity="relu")
    def fused_kernel(self):
        central=self.central.reshape(self.channels,1,9)
        correction=torch.zeros_like(central)
        correction[:,:,4]=central.sum(-1)
        central=(central-correction).reshape_as(self.central)
        angular=self.angular.reshape(self.channels,1,9)
        angular=(angular-angular[:,:,[3,0,1,6,4,2,7,8,5]]).reshape_as(self.angular)
        horizontal=self.horizontal.new_zeros(self.channels,1,9)
        horizontal[:,:,[0,3,6]]=self.horizontal
        horizontal[:,:,[2,5,8]]=-self.horizontal
        vertical=self.vertical.new_zeros(self.channels,1,9)
        vertical[:,:,[0,1,2]]=self.vertical
        vertical[:,:,[6,7,8]]=-self.vertical
        return self.vanilla+central+angular+horizontal.reshape_as(self.vanilla)+vertical.reshape_as(self.vanilla)
    def forward(self,x):
        return F.conv2d(x,self.fused_kernel(),self.bias.sum(0),padding=1,groups=self.channels)

class LowFrequency(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.norm1=nn.LayerNorm(channels)
        self.ssm=SS2D(d_model=channels,d_state=16,expand=2)
        self.norm2=nn.LayerNorm(channels)
        self.depthwise=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.project=nn.Conv2d(channels//2,channels,1)
    def forward(self,x):
        normalized=self.norm1(x.permute(0,2,3,1)).contiguous()
        z=x+self.ssm(normalized).permute(0,3,1,2).contiguous()
        branch=self.norm2(z.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        a,b=self.depthwise(branch).chunk(2,dim=1)
        return z+self.project(torch.sigmoid(a)*b)

class HighFrequency(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.low=nn.Sequential(nn.Conv2d(channels,channels,1),
                              nn.Conv2d(channels,channels,3,padding=1,groups=channels))
        self.high=nn.Sequential(nn.Conv2d(channels,channels,1),
                               nn.Conv2d(channels,channels,3,padding=1,groups=channels))
        self.difference=DifferenceConv(channels)
    def forward(self,low,high):
        batch,channels,height,width=low.shape
        high=high.reshape(batch*3,channels,height,width)
        context=self.low(low)[:,None].expand(batch,3,channels,height,width).reshape(batch*3,channels,height,width)
        gated=context*torch.sigmoid(self.high(high))
        return self.difference(gated).reshape(batch,3*channels,height,width)

class WAM(nn.Module):
    def __init__(self,channels,inner_channels=None):
        super().__init__()
        inner_channels=inner_channels or channels//2
        self.pre=nn.Conv2d(channels,inner_channels,1)
        self.low=LowFrequency(inner_channels)
        self.high=HighFrequency(inner_channels)
        self.refine=nn.Conv2d(4*inner_channels,4*channels,1,groups=4)
        self.norm=nn.LayerNorm(channels)
    def forward(self,x):
        ll,high=dwt_init(self.pre(x))
        ll=self.low(ll)
        high=self.high(ll,high)
        restored=iwt_init(self.refine(torch.cat([ll,high],dim=1)))
        restored=self.norm(restored.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        return x+restored

class RWAMG(nn.Module):
    def __init__(self,channels,blocks):
        super().__init__()
        self.blocks=nn.Sequential(*[WAM(channels) for _ in range(blocks)])
        self.refine=nn.Conv2d(channels,channels,1)
    def forward(self,x):
        return x+self.refine(self.blocks(x))

class WMSRPaper(nn.Module):
    def __init__(self,scale=4,channels=64,groups=6,blocks=4):
        super().__init__()
        self.scale=scale
        self.sub_mean=MeanShift(255)
        self.add_mean=MeanShift(255,sign=1)
        self.head=nn.Conv2d(3,channels,3,padding=1)
        self.groups=nn.Sequential(*[RWAMG(channels,blocks) for _ in range(groups)])
        self.refine=nn.Conv2d(channels,channels,3,padding=1)
        self.tail=nn.Sequential(nn.Conv2d(channels,3*scale*scale,3,padding=1),nn.PixelShuffle(scale))
        self.apply(self._initialize)
    @staticmethod
    def _initialize(module):
        if isinstance(module,nn.Conv2d) and any(p.requires_grad for p in module.parameters(recurse=False)):
            nn.init.kaiming_normal_(module.weight,mode="fan_in",nonlinearity="relu")
            if module.bias is not None:nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None and not getattr(module.bias,"_no_reinit",False):nn.init.zeros_(module.bias)
    def forward(self,x):
        height,width=x.shape[-2:]
        if height%2 or width%2:x=F.pad(x,(0,width%2,0,height%2),mode="reflect")
        shallow=self.head(self.sub_mean(x))
        deep=shallow+self.refine(self.groups(shallow))
        out=self.add_mean(self.tail(deep))
        return out[:,:,:height*self.scale,:width*self.scale]

