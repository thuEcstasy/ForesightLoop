# SPDX-License-Identifier: Apache-2.0
"""Explicit trajectory-coordinate source sets; all source cycles strictly precede receiver."""
import math
import torch
from torch import nn


def source_spec(receiver,mode):
    # cycle p = trajectory t + receiver k - K - 1.
    # completed[j-1][p-delay] corresponds to trajectory t+k-j-delay.
    assert receiver>=2
    result=[]
    for j in range(1,receiver):
        if mode=='legacy_post':delays=(1,)
        elif mode=='colored_post':delays=tuple(range(1,receiver-j)) if j<receiver-1 else (1,)
        elif mode=='colored_full':delays=range(1,receiver-j+2)
        else:raise ValueError('Source set must be explicitly selected; unbounded history is not this prototype')
        for delay in delays:
            offset=receiver-j-delay
            label='PRE' if offset==-1 else '1D' if offset==0 else 'POST'
            assert delay>=1 and offset>=-1
            result.append(dict(loop=j,delay=delay,trajectory_offset=offset,kind=label))
    return tuple(result)


def shifted(x,docs,valid,delay):
    B,T,D=x.shape
    if delay>=T:return torch.zeros_like(x),torch.zeros((B,T),device=x.device,dtype=torch.bool)
    present=docs[:,delay:]==docs[:,:-delay]
    if valid is not None:present=present & valid[:,delay:] & valid[:,:-delay]
    mask=torch.cat((torch.zeros((B,delay),device=x.device,dtype=torch.bool),present),1)
    state=torch.cat((torch.zeros_like(x[:,:delay]),x[:,:-delay]),1)
    return state,mask


class SourceSetAggregation(nn.Module):
    def __init__(self,dim,loops,eps,source_set,source_rmsnorm=True):
        super().__init__()
        self.dim,self.loops,self.eps,self.source_set=dim,loops,eps,source_set
        self.source_rmsnorm=bool(source_rmsnorm)
        self.mode='attnres'
        self.alpha=nn.ParameterList([nn.Parameter(torch.ones(())) for _ in range(loops-1)])
        self.query=nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(loops-1)])
        self.bias=nn.ParameterList([nn.Parameter(torch.zeros(len(source_spec(r,source_set)))) for r in range(2,loops+1)])

    def training_input(self,embedding,completed,docs,valid=None):
        from .aggregation import aggregate
        r=len(completed)+1
        states=[];masks=[]
        for spec in source_spec(r,self.source_set):
            x,mask=shifted(completed[spec['loop']-1],docs,valid,spec['delay'])
            states.append(x);masks.append(mask)
        sources=torch.stack(states,-2);mask=torch.stack(masks,-1)
        msg,weights=self.message(sources,r,source_mask=mask)
        collector=getattr(self,'audit_collector',None)
        if collector is not None:collector.append(dict(receiver=r,source=sources,valid=mask,message=msg,weights=weights))
        return (embedding.float()+msg).to(embedding.dtype)

    def message(self,sources,receiver,*,source_mask=None):
        from .aggregation import aggregate
        assert sources.shape[-2:]==(len(source_spec(receiver,self.source_set)),self.dim)
        return aggregate(sources,self.query[receiver-2],self.bias[receiver-2],self.alpha[receiver-2],self.eps,source_mask=source_mask,source_rmsnorm=self.source_rmsnorm)

    def prepared_inputs(self,*args,**kwargs):
        raise NotImplementedError('This new source set needs a multi-cycle decode cache; training-only until independently audited')
