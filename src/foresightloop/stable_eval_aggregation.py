# SPDX-License-Identifier: Apache-2.0
"""FP32 AttnRes reductions with a fixed tree independent of outer batch shape."""
import math
import torch

def fixed_sum(x,dim=-1):
    x=x.movedim(dim,-1)
    while x.shape[-1]>1:
        pairs=x.shape[-1]//2
        reduced=x[...,:2*pairs:2]+x[...,1:2*pairs:2]
        x=torch.cat((reduced,x[...,-1:]),-1) if x.shape[-1]%2 else reduced
    return x.squeeze(-1)

def aggregate(sources,query,bias,alpha,eps,*,history_valid=None,source_mask=None,accumulation_dtype=torch.float32,source_rmsnorm=True):
    assert accumulation_dtype in (torch.float32,torch.float64)
    mask=torch.ones(sources.shape[:-1],dtype=torch.bool,device=sources.device)
    if source_mask is not None:mask=mask&source_mask.to(device=sources.device,dtype=torch.bool)
    if history_valid is not None:mask=mask&history_valid.to(device=sources.device,dtype=torch.bool)[...,None]
    x=torch.where(mask[...,None],sources.to(accumulation_dtype),0.)
    mean=fixed_sum(x.square())/x.shape[-1]
    values=x*torch.rsqrt(mean[...,None]+eps) if source_rmsnorm else x
    scores=torch.zeros_like(x[...,0]) if query is None else fixed_sum(values*query.to(accumulation_dtype))/math.sqrt(x.shape[-1])
    if bias is not None:scores=scores+bias.to(accumulation_dtype)
    scores=scores.masked_fill(~mask,float('-inf'))
    scores=torch.where(mask.any(-1,keepdim=True),scores,torch.zeros_like(scores))
    shifted=scores-scores.max(-1,keepdim=True).values
    numer=shifted.exp()
    weights=numer/fixed_sum(numer)[...,None]*mask
    return fixed_sum(weights[...,None]*values,-2)*alpha.to(accumulation_dtype),weights

def install():
    from . import aggregation
    from . import trajectory
    # Keep the independent original aggregate callable available for parity
    # checks; only model message dispatch uses the stable implementation.
    old=aggregation.CrossLoopAggregation.message
    def message(self,sources,receiver,*,history_valid=None,source_mask=None):
        assert self.mode=='attnres' and 2<=receiver<=self.loops
        assert sources.shape[-2:]==(receiver-1,self.dim)
        return aggregate(sources,self.query[receiver-2],self.bias[receiver-2],self.alpha[receiver-2],self.eps,
                         history_valid=history_valid,source_mask=source_mask,source_rmsnorm=self.source_rmsnorm)
    aggregation.CrossLoopAggregation.message=message
    def normalize(h,eps,source_rmsnorm=True):
        x=h.float()
        return x*torch.rsqrt((fixed_sum(x.square())/x.shape[-1])[...,None]+eps) if source_rmsnorm else x
    def normalized_message(values,module,receiver,mask):
        x=torch.where(mask[...,None],values.float(),0.)
        scores=fixed_sum(x*module.query[receiver-2].float())/math.sqrt(module.dim)+module.bias[receiver-2].float()
        scores=scores.masked_fill(~mask,float('-inf'))
        scores=torch.where(mask.any(-1,keepdim=True),scores,torch.zeros_like(scores))
        numer=(scores-scores.max(-1,keepdim=True).values).exp()
        weights=numer/fixed_sum(numer)[...,None]*mask
        return fixed_sum(weights[...,None]*x,-2)*module.alpha[receiver-2].float(),weights
    trajectory.normalize=normalize
    trajectory.normalized_message=normalized_message
    return old
