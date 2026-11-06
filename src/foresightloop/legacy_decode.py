# SPDX-License-Identifier: Apache-2.0
"""Layer-synchronous eager prefill/decode; immutable per-request cache updates.

This is a correctness implementation, not a fused serving engine. Cross-loop
state is O(B*R*d); global/local native KV retain the original PLT rules.
"""
from dataclasses import dataclass
import torch
from .checkpoint import signature


@dataclass(frozen=True)
class RequestState:
    prev_hidden: tuple
    kv: tuple
    document: int
    position: int


@dataclass(frozen=True)
class BatchCache:
    rows: tuple
    config: dict

    def reorder(self,indices):
        ids=tuple(int(i) for i in indices)
        if any(i<0 or i>=len(self.rows) for i in ids):raise IndexError('Invalid cache index')
        # Duplicates are safe: no method mutates stored tensors or tuple entries.
        return BatchCache(tuple(self.rows[i] for i in ids),self.config.copy())

    def reset_slots(self,indices):
        ids=set(int(i) for i in indices)
        if any(i<0 or i>=len(self.rows) for i in ids):raise IndexError('Invalid cache index')
        return BatchCache(tuple(None if i in ids else row for i,row in enumerate(self.rows)),self.config.copy())


@dataclass
class StreamOutput:
    logits: torch.Tensor
    cache: BatchCache
    hidden: object=None
    inputs: object=None
    kv: object=None


def _one(model,token,document,old,order=None):
    e=model.embedding(token.reshape(1,1));R=model.H;c=model.c
    if old is not None and (old.prev_hidden[0].device!=e.device or old.prev_hidden[0].dtype!=e.dtype):
        raise ValueError('Cache and model must have the same device and dtype')
    has_history=old is not None and old.document==document
    prev=old.prev_hidden if has_history else tuple(torch.zeros_like(e) for _ in range(R))
    pos=old.position+1 if has_history else 0
    history=old.kv if has_history else tuple(tuple(None for _ in model.stack) for _ in range(R))
    if model.cross_loop_aggregation=='legacy':
        prepared=(e,)+tuple(e+prev[r-1] for r in range(1,R))
    else:prepared=model.cross_loop.prepared_inputs(e,prev,has_history=has_history,order=order)
    states=list(prepared);native=[[None]*c.layers for _ in range(R)];updated=[[None]*c.layers for _ in range(R)]
    order=tuple(range(R)) if order is None else tuple(r-1 for r in order)
    if sorted(order)!=list(range(R)):raise ValueError('Invalid branch order')
    position=torch.tensor([[pos]],device=e.device)
    for i,layer in enumerate(model.stack):
        projected=[layer.project(states[r],position) for r in range(R)]
        for r,(_,kv,_) in enumerate(projected):
            native[r][i]=kv
            past=history[r][i]
            combined=kv if past is None else tuple(torch.cat((a,b),2) for a,b in zip(past,kv))
            updated[r][i]=tuple(t[:,:,-c.window:] for t in combined) if r else combined
        # Publish this layer's first-loop KV before any branch reads it.
        gkv=updated[0][i]
        gm=torch.ones((1,1,gkv[0].shape[2]),device=e.device,dtype=torch.bool)
        for r in order:
            q,_,gate=projected[r];local=updated[r][i] if r else None
            lm=torch.ones((1,1,local[0].shape[2]),device=e.device,dtype=torch.bool) if r else None
            states[r]=layer.finish(states[r],q,gate,gkv,gm,local,lm)
    # Atomic publication: every branch consumed the same old snapshot.
    new=RequestState(tuple(states),tuple(tuple(kvs) for kvs in updated),document,pos)
    return model.head(states[-1]),new,prepared,native


@torch.no_grad()
def decode(model,tokens,docs=None,*,valid=None,cache=None,return_trace=False):
    if tokens.ndim!=1:raise ValueError('decode tokens must be [B]')
    B=tokens.shape[0];c=model.c;R=model.H
    if cache is None:cache=BatchCache((None,)*B,signature(model))
    if len(cache.rows)!=B or cache.config!=signature(model):raise ValueError('Incompatible request cache')
    if docs is None:docs=torch.tensor([s.document if s is not None else 0 for s in cache.rows],device=tokens.device)
    if valid is None:valid=torch.ones_like(tokens,dtype=torch.bool)
    if docs.shape!=tokens.shape or valid.shape!=tokens.shape or valid.dtype!=torch.bool:
        raise ValueError('decode docs/valid shape mismatch')
    logits=[];rows=[];hidden=[];inputs=[];native=[]
    for b in range(B):
        if bool(valid[b]):
            y,state,z,kv=_one(model,tokens[b],int(docs[b]),cache.rows[b])
            h=state.prev_hidden
        else:
            state=cache.rows[b];y=model.embedding.weight.new_zeros(1,1,c.vocab)
            h=z=tuple(model.embedding.weight.new_zeros(1,1,c.dim) for _ in range(R))
            empty=model.embedding.weight.new_zeros(1,c.kv_heads,1,c.head_dim)
            kv=[[(empty,empty) for _ in range(c.layers)] for _ in range(R)]
        rows.append(state);logits.append(y)
        if return_trace:hidden.append(torch.stack(h,2));inputs.append(torch.stack(z,2));native.append(kv)
    output=torch.cat(logits,0) if B else model.embedding.weight.new_zeros(0,1,c.vocab)
    trace=None
    if return_trace:
        trace={('G' if r==0 else f'L{r+1}'):[tuple(torch.cat([v[r][i][j] for v in native],0) if B else model.embedding.weight.new_zeros(0,c.kv_heads,1,c.head_dim) for j in (0,1)) for i in range(c.layers)] for r in range(R)}
    return StreamOutput(output,BatchCache(tuple(rows),cache.config.copy()),
        torch.cat(hidden,0) if hidden else (model.embedding.weight.new_zeros(B,1,R,c.dim) if return_trace else None),
        torch.cat(inputs,0) if inputs else (model.embedding.weight.new_zeros(B,1,R,c.dim) if return_trace else None),trace)


@torch.no_grad()
def prefill(model,tokens,docs=None,*,valid=None,cache=None,return_trace=False):
    from .sequence import layout
    if tokens.ndim!=2:raise ValueError('prefill tokens must be [B,T]')
    B,T=tokens.shape;c=model.c;R=model.H
    if cache is None:cache=BatchCache((None,)*B,signature(model))
    if len(cache.rows)!=B or cache.config!=signature(model):raise ValueError('Incompatible request cache')
    if docs is None:
        docs=torch.tensor([s.document if s is not None else 0 for s in cache.rows],device=tokens.device)[:,None].expand_as(tokens)
    _,_,valid=layout(tokens,docs,valid)
    logits=[];hidden=[];inputs=[];native=[]
    for t in range(T):
        out=decode(model,tokens[:,t],docs[:,t],valid=valid[:,t],cache=cache,return_trace=return_trace)
        cache=out.cache;logits.append(out.logits)
        if return_trace:hidden.append(out.hidden);inputs.append(out.inputs);native.append(out.kv)
        # Keep only the newest cache. Retaining StreamOutput objects would keep
        # every old global-KV concatenation and all prior prev_hidden snapshots.
        del out
    logits=torch.cat(logits,1) if T else model.embedding.weight.new_zeros(B,0,c.vocab)
    h=z=kv=None
    if return_trace:
        h=torch.cat(hidden,1) if T else model.embedding.weight.new_zeros(B,0,R,c.dim)
        z=torch.cat(inputs,1) if T else model.embedding.weight.new_zeros(B,0,R,c.dim)
        kv={('G' if r==0 else f'L{r+1}'):[tuple(torch.cat([v['G' if r==0 else f'L{r+1}'][i][j] for v in native],2) if T else model.embedding.weight.new_zeros(B,c.kv_heads,0,c.head_dim) for j in (0,1)) for i in range(c.layers)] for r in range(R)}
    return StreamOutput(logits,cache,h,z,kv)
