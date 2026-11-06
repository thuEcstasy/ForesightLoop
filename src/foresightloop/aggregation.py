# SPDX-License-Identifier: Apache-2.0
"""Normalized-value, receiver-query aggregation for selected loop states.

Production reductions always use FP32. The pure function's explicit FP64 option
is only for finite-difference auditing (casting doubles to FP32 defeats gradcheck).
No R-by-R-by-d state tensor or additional backbone is created.
"""
import math
import torch
from torch import nn

MODES = ('legacy', 'normalized_chain', 'attnres', 'attnres_fullstate')


def aggregate(sources, query, bias, alpha, eps, *, history_valid=None,
              source_mask=None, accumulation_dtype=torch.float32, source_rmsnorm=True):
    """sources [..., K, d]; returns FP32 message [..., d] and weights [..., K]."""
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError('Reduction dtype must be FP32, or FP64 for gradcheck')
    if sources.ndim < 2 or sources.shape[-2] == 0 or eps <= 0:
        raise ValueError('Need at least one source, a feature axis, and positive eps')
    shape=sources.shape[:-1]
    mask=torch.ones(shape,dtype=torch.bool,device=sources.device)
    if source_mask is not None: mask=mask & source_mask.to(device=sources.device,dtype=torch.bool)
    if history_valid is not None: mask=mask & history_valid.to(device=sources.device,dtype=torch.bool)[...,None]
    # Invalid history is absent, including when its unused storage contains NaNs.
    x=torch.where(mask[...,None],sources.to(accumulation_dtype),0.)
    values=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps) if source_rmsnorm else x
    if query is None:
        scores=torch.zeros_like(x[...,0])
    else:
        scores=(values*query.to(accumulation_dtype)).sum(-1)/math.sqrt(sources.shape[-1])
    if bias is not None:scores=scores+bias.to(accumulation_dtype)
    scores=scores.masked_fill(~mask,float('-inf'))
    # Never softmax an all -inf row. Empty history has exactly zero weights/message.
    scores=torch.where(mask.any(-1,keepdim=True),scores,torch.zeros_like(scores))
    weights=torch.softmax(scores,-1)*mask
    message=(weights[...,None]*values).sum(-2)*alpha.to(accumulation_dtype)
    return message,weights


class CrossLoopAggregation(nn.Module):
    def __init__(self, dim, loops, mode, eps, source_trajectory_offset=None, cross_loop_source='previous_token', source_rmsnorm=True):
        super().__init__()
        if mode not in MODES[1:] or loops < 1 or dim < 1 or not math.isfinite(eps) or eps<=0:
            raise ValueError('Invalid cross-loop aggregation configuration')
        self.dim,self.loops,self.mode,self.eps=dim,loops,mode,float(eps)
        self.source_rmsnorm=bool(source_rmsnorm)
        if source_trajectory_offset not in (None, 0, 1):
            raise ValueError('Unknown trajectory offset')
        self.source_trajectory_offset=source_trajectory_offset
        if cross_loop_source not in ('previous_token','same_token'):
            raise ValueError('Unknown hidden source')
        if cross_loop_source=='same_token' and (source_trajectory_offset is not None or mode!='attnres'):
            raise ValueError('Serial aggregation cannot use a shifted trajectory')
        self.cross_loop_source=cross_loop_source
        self.alpha=nn.ParameterList([nn.Parameter(torch.ones(())) for _ in range(loops-1)])
        if mode=='attnres':
            # Keep the redundant receiver-2 query/bias explicitly: their gradients are zero.
            self.query=nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(loops-1)])
            self.bias=nn.ParameterList([nn.Parameter(torch.zeros(r-1)) for r in range(2,loops+1)])

    def message(self, sources, receiver, *, history_valid=None, source_mask=None):
        if not 2<=receiver<=self.loops or sources.shape[-2:]!=(receiver-1,self.dim):
            raise ValueError('Receiver r requires precisely the r-1 shallower stack outputs')
        if self.mode=='normalized_chain':
            chain=torch.arange(receiver-1,device=sources.device)==receiver-2
            source_mask=chain if source_mask is None else source_mask & chain
            q=b=None
        else:q,b=self.query[receiver-2],self.bias[receiver-2]
        return aggregate(sources,q,b,self.alpha[receiver-2],self.eps,
                         history_valid=history_valid,source_mask=source_mask,source_rmsnorm=self.source_rmsnorm)

    def training_input(self, embedding, completed, docs, valid=None):
        """completed[j] is H_(j+1)[B,T,d]; shift only the token axis."""
        if self.cross_loop_source=='same_token':
            from .serial_aggregation import training_input
            return training_input(self,embedding,completed,docs,valid)
        if self.source_trajectory_offset is not None:
            from .trajectory import training_input
            return training_input(self,embedding,completed,docs,valid)
        from .model import shift
        receiver=len(completed)+1
        sources=torch.stack([shift(h,docs) for h in completed],-2)
        has_history=torch.cat((torch.zeros_like(docs[:,:1],dtype=torch.bool),docs[:,1:]==docs[:,:-1]),1)
        if valid is not None:
            has_history=has_history & valid & torch.cat((torch.zeros_like(valid[:,:1]),valid[:,:-1]),1)
        message,_=self.message(sources,receiver,history_valid=has_history)
        return (embedding.float()+message).to(embedding.dtype)

    def prepared_inputs(self, embedding, previous, *, has_history=True, order=None):
        """Use one immutable previous-token snapshot; order cannot change dependencies."""
        if self.cross_loop_source=='same_token':
            raise ValueError('Serial inputs require current-token completed loops, not an old snapshot')
        if len(previous)!=self.loops:raise ValueError('Wrong previous-hidden loop count')
        order=tuple(range(1,self.loops+1)) if order is None else tuple(order)
        if sorted(order)!=list(range(1,self.loops+1)):raise ValueError('Order must be a loop permutation')
        result=[None]*self.loops
        valid=torch.full(embedding.shape[:-1],has_history,dtype=torch.bool,device=embedding.device)
        for receiver in order:
            if receiver==1:result[0]=embedding
            else:
                sources=torch.stack(previous[:receiver-1],-2)
                msg,_=self.message(sources,receiver,history_valid=valid)
                result[receiver-1]=(embedding.float()+msg).to(embedding.dtype)
        return tuple(result)
