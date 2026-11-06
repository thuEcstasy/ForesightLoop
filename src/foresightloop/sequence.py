# SPDX-License-Identifier: Apache-2.0
"""Packed/ragged math path, with the same PLT backbone and last-loop readout."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class GraphOutput:
    hidden: torch.Tensor
    loops: tuple
    inputs: tuple
    kv: dict


def layout(tokens, docs, valid=None):
    if tokens.ndim!=2 or docs.shape!=tokens.shape:
        raise ValueError('tokens and docs must be [B,T]')
    if valid is None:valid=torch.ones_like(tokens,dtype=torch.bool)
    if valid.shape!=tokens.shape or valid.dtype!=torch.bool:
        raise ValueError('valid must be a boolean [B,T] mask')
    # Baseline has unpadded rows. The new API supports a contiguous valid span
    # (left/right padding and empty rows), and explicitly rejects internal holes.
    if tokens.shape[1]>1:
        starts=valid[:,1:] & ~valid[:,:-1]
        if bool((starts.sum(1)+valid[:,0].long()>1).any()):
            raise ValueError('Interior padding holes are unsupported; compact valid tokens first')
    starts=torch.ones_like(docs,dtype=torch.bool)
    starts[:,1:]=(docs[:,1:]!=docs[:,:-1]) | ~valid[:,:-1]
    # Repeated external document IDs still identify separate contiguous segments.
    segments=starts.long().cumsum(1)
    return tokens.masked_fill(~valid,0),segments.masked_fill(~valid,-1),valid


def dense(model, tokens, docs, valid=None):
    """Token-parallel per-loop forward; no per-token Python loop in training."""
    from .model import positions,shift
    tokens,docs,valid=layout(tokens,docs,valid)
    e=model.embedding(tokens);pos=positions(docs);T=tokens.shape[1]
    delta=torch.arange(T,device=tokens.device)[:,None]-torch.arange(T,device=tokens.device)[None,:]
    gm=(docs[:,:,None]==docs[:,None,:]) & (delta>=0) & valid[:,:,None] & valid[:,None,:]
    lm=gm & (delta<model.c.window)
    h=e;outputs=[];inputs=[];trace={};gkv=None
    for r in range(model.H):
        if r:
            h=e+(outputs[-1] if model.cross_loop_source=='same_token' else shift(outputs[-1],docs)) if model.cross_loop_aggregation=='legacy' else model.cross_loop.training_input(e,outputs,docs,valid)
        inputs.append(h.masked_fill(~valid[...,None],0.))
        kvs=[]
        for i,layer in enumerate(model.stack):
            q,kv,gate=layer.project(h,pos)
            kv=tuple(t.masked_fill(~valid[:,None,:,None],0.) for t in kv);kvs.append(kv)
            h=layer.finish(h,q,gate,kv if r==0 else gkv[i],gm,None if r==0 else kv,lm)
        h=h.masked_fill(~valid[...,None],0.);outputs.append(h)
        if r==0:gkv=kvs
        trace['G' if r==0 else f'L{r+1}']=kvs
    return GraphOutput(h,tuple(outputs),tuple(inputs),trace)


def next_token_loss(logits,tokens,docs,valid=None):
    """h[t,R] predicts x[t+1]; exclude padding and cross-document targets."""
    safe,segments,valid=layout(tokens,docs,valid)
    mask=valid[:,:-1] & valid[:,1:] & (segments[:,:-1]==segments[:,1:])
    if tokens.shape[1]<2:return logits.sum()*0,mask.sum()
    target=safe[:,1:].masked_fill(~mask,-100)
    loss=F.cross_entropy(logits[:,:-1].float().flatten(0,1),target.flatten(),ignore_index=-100,reduction='sum')
    return loss,mask.sum()
