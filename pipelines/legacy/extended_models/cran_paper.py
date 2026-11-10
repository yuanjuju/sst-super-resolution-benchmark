"""CRAN section3.5 topology and Eq1-2 reasoning; clearly a reconstruction.

The third-party default places a CRAB inside every RB and disables CDRR.
Here each group uses19plain RBs then1CRAB, and sigmoid reasoning is active.
The contextual convolution comes from the pinned third-party/CGC integration.
"""
import torch
from torch import nn
from .vendor import cran_common as c
from .vendor import edsr_common as e

class Reasoning(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.phi=nn.Parameter(torch.empty(channels,channels));self.theta=nn.Parameter(torch.empty(channels,channels))
        self.graph=nn.Parameter(torch.empty(channels,channels));self.restore=nn.Parameter(torch.empty(channels,channels))
        for weight in self.parameters():nn.init.kaiming_normal_(weight)
    def forward(self,x):
        affinity=(self.phi@x).transpose(1,2)@(self.theta@x)
        gate=torch.sigmoid((affinity@x.transpose(1,2)@self.graph@self.restore).transpose(1,2))
        return x+gate*x

class ContextConv(c.Conv2d_CG):
    def __init__(self,channels):
        super().__init__(channels,channels)
        self.ce=nn.Sequential(self.ce,Reasoning(channels))
        # Base class creates inherited duplicate Conv2d parameters which its custom forward never uses.
        self.weight=self.weight_conv;self.bias=self.bias_conv
    def forward(self,x):return super().forward(x)+self.bias_conv.view(1,-1,1,1)

class ContextBlock(nn.Module):
    def __init__(self,channels):
        super().__init__();self.body=nn.Sequential(ContextConv(channels),nn.ReLU(True),ContextConv(channels))
    def forward(self,x):return x+self.body(x)

class Group(nn.Module):
    def __init__(self,channels):
        super().__init__();self.body=nn.Sequential(*[e.ResBlock(e.default_conv,channels,3,act=nn.ReLU(True)) for _ in range(19)],
            ContextBlock(channels),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return x+self.body(x)

class CRANPaper(nn.Module):
    def __init__(self,scale=4):
        super().__init__();channels=64
        self.sub_mean=e.MeanShift(255);self.add_mean=e.MeanShift(255,sign=1)
        self.head=nn.Conv2d(3,channels,3,padding=1)
        self.body=nn.Sequential(*[Group(channels) for _ in range(10)],nn.Conv2d(channels,channels,3,padding=1))
        self.tail=nn.Sequential(e.Upsampler(e.default_conv,scale,channels),nn.Conv2d(channels,3,3,padding=1))
    def forward(self,x):
        shallow=self.head(self.sub_mean(x));return self.add_mean(self.tail(shallow+self.body(shallow)))
