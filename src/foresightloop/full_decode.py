# SPDX-License-Identifier: Apache-2.0
"""Immutable compact raw-hidden histories; layer-synchronous branch execution."""
from dataclasses import dataclass
import torch
from .checkpoint import signature
from .legacy_decode import BatchCache, StreamOutput
from .full_sources import source_spec

def capacities(loops,offset):
    assert offset is None
    return tuple(loops-j+1 for j in range(1,loops))


@dataclass(frozen=True)
class Entry:
    cycle: int
    document: int
    hidden: torch.Tensor


@dataclass(frozen=True)
class RequestState:
    histories: tuple
    kv: tuple
    document: int
    position: int


def prepared_inputs(model, embedding, old, document, position, order=None):
    R = model.H
    order = tuple(range(1, R+1)) if order is None else tuple(order)
    if sorted(order) != list(range(1, R+1)):
        raise ValueError('Invalid branch order')
    result = [None] * R
    for r in order:
        if r == 1:
            result[0] = embedding
            continue
        sources, available = [], []
        for spec in source_spec(r,'colored_full'):
            j=spec['loop'];distance=spec['delay']
            history = () if old is None else old.histories[j-1]
            entry = history[-distance] if len(history) >= distance else None
            good = entry is not None and entry.cycle == position-distance and entry.document == document
            sources.append(entry.hidden if good else torch.zeros_like(embedding))
            available.append(good)
        source = torch.stack(sources, -2)
        mask = torch.tensor(available, dtype=torch.bool, device=embedding.device)
        message, _ = model.cross_loop.message(source, r, source_mask=mask)
        result[r-1] = (embedding.float()+message).to(embedding.dtype)
    return tuple(result)


def _one(model, token, document, old, order=None):
    e = model.embedding(token.reshape(1, 1)); R = model.H; c = model.c
    if old is not None and old.document != document:
        old = None
    if old is not None and old.kv:
        tensor = old.kv[0][0][0]
        if tensor.device != e.device:
            raise ValueError('Cache device mismatch')
    position = 0 if old is None else old.position + 1
    prepared = prepared_inputs(model, e, old, document, position, order)
    states = list(prepared)
    past = old.kv if old is not None else tuple(tuple(None for _ in model.stack) for _ in range(R))
    native = [[None]*c.layers for _ in range(R)]
    updated = [[None]*c.layers for _ in range(R)]
    branch_order = tuple(range(R)) if order is None else tuple(r-1 for r in order)
    if sorted(branch_order) != list(range(R)):
        raise ValueError('Invalid branch order')
    pos = torch.tensor([[position]], device=e.device)
    for i, layer in enumerate(model.stack):
        projected = [layer.project(states[r], pos) for r in range(R)]
        for r, (_, kv, _) in enumerate(projected):
            native[r][i] = kv
            previous = past[r][i]
            joined = kv if previous is None else tuple(torch.cat((a, b), 2) for a, b in zip(previous, kv))
            updated[r][i] = tuple(t[:, :, -c.window:] for t in joined) if r else joined
        # The first-loop KV is published once at this layer, as in the old graph.
        gkv = updated[0][i]
        gm = torch.ones((1, 1, gkv[0].shape[2]), dtype=torch.bool, device=e.device)
        for r in branch_order:
            q, _, gate = projected[r]
            local = updated[r][i] if r else None
            lm = torch.ones((1, 1, local[0].shape[2]), dtype=torch.bool, device=e.device) if r else None
            states[r] = layer.finish(states[r], q, gate, gkv, gm, local, lm)
    # Publish only after every branch consumed the immutable old histories.
    histories = []
    for j, depth in enumerate(capacities(R, model.source_trajectory_offset)):
        previous = () if old is None else old.histories[j]
        histories.append((previous+(Entry(position, document, states[j]),))[-depth:])
    new = RequestState(tuple(histories), tuple(tuple(x) for x in updated), document, position)
    return model.head(states[-1]), new, tuple(states), prepared, native


@torch.no_grad()
def decode(model, tokens, docs=None, *, valid=None, cache=None, return_trace=False):
    if tokens.ndim != 1:
        raise ValueError('decode expects [B]')
    B = len(tokens); R = model.H; c = model.c
    if cache is None:
        cache = BatchCache((None,)*B, signature(model))
    if len(cache.rows) != B or cache.config != signature(model):
        raise ValueError('Incompatible request cache')
    if docs is None:
        docs = torch.tensor([s.document if s else 0 for s in cache.rows], device=tokens.device)
    if valid is None:
        valid = torch.ones_like(tokens, dtype=torch.bool)
    if docs.shape != tokens.shape or valid.shape != tokens.shape or valid.dtype != torch.bool:
        raise ValueError('docs/valid shape or dtype mismatch')
    ys=[]; rows=[]; hs=[]; zs=[]; kvs=[]
    for b in range(B):
        if bool(valid[b]):
            y, state, h, z, kv = _one(model, tokens[b], int(docs[b]), cache.rows[b])
        else:
            state=cache.rows[b]; y=model.embedding.weight.new_zeros(1,1,c.vocab)
            h=z=tuple(model.embedding.weight.new_zeros(1,1,c.dim) for _ in range(R))
            empty=model.embedding.weight.new_zeros(1,c.kv_heads,1,c.head_dim)
            kv=[[(empty,empty) for _ in model.stack] for _ in range(R)]
        ys.append(y); rows.append(state)
        if return_trace:
            hs.append(torch.stack(h,2));zs.append(torch.stack(z,2));kvs.append(kv)
    trace=None
    if return_trace:
        trace={('G' if r==0 else f'L{r+1}'):[tuple(torch.cat([v[r][i][k] for v in kvs],0)
                 for k in (0,1)) for i in range(c.layers)] for r in range(R)}
    return StreamOutput(torch.cat(ys,0), BatchCache(tuple(rows),cache.config.copy()),
                        torch.cat(hs,0) if return_trace else None,
                        torch.cat(zs,0) if return_trace else None, trace)


@torch.no_grad()
def parallel_prefill(model, tokens, docs, valid, return_trace):
    from .sequence import layout
    from .engine import hidden
    safe, segments, valid = layout(tokens, docs, valid)
    full_rows = bool(valid.all()) and bool((segments == segments[:, :1]).all())
    result = hidden(model, safe, None if full_rows else segments, single_document=full_rows,
                    valid=None if full_rows else valid, return_details=True)
    rows=[]; c=model.c; R=model.H
    for b in range(len(tokens)):
        indices=valid[b].nonzero().flatten()
        if not len(indices):
            rows.append(None);continue
        last=int(indices[-1]);segment=segments[b,last]
        indices=indices[segments[b,indices]==segment];first=int(indices[0]);length=len(indices)
        document=int(docs[b,last]); histories=[]
        for j,depth in enumerate(capacities(R,model.source_trajectory_offset)):
            histories.append(tuple(Entry(t-first,document,result.loops[j][b:b+1,t:t+1].clone())
                                   for t in range(max(first,last-depth+1),last+1)))
        kv=[]
        for r in range(R):
            lane='G' if r==0 else f'L{r+1}'
            start=first if r==0 else max(first,last-c.window+1)
            kv.append(tuple(tuple(v[b:b+1,:,start:last+1].clone() for v in pair) for pair in result.kv[lane]))
        rows.append(RequestState(tuple(histories),tuple(kv),document,length-1))
    return StreamOutput(model.head(result.hidden),BatchCache(tuple(rows),signature(model)),
                        torch.stack(result.loops,2) if return_trace else None,
                        torch.stack(result.inputs,2) if return_trace else None,
                        result.kv if return_trace else None)


@torch.no_grad()
def prefill(model,tokens,docs=None,*,valid=None,cache=None,return_trace=False):
    from .sequence import layout
    if tokens.ndim!=2 or tokens.shape[1]==0:raise ValueError('prefill expects nonempty [B,T]')
    B,T=tokens.shape
    if cache is not None and (len(cache.rows)!=B or cache.config!=signature(model)):
        raise ValueError('Incompatible request cache')
    if docs is None:
        docs=torch.tensor([s.document if s else 0 for s in cache.rows] if cache is not None else [0]*B,device=tokens.device)[:,None].expand_as(tokens)
    _,_,valid=layout(tokens,docs,valid)
    if cache is None or all(s is None for s in cache.rows):
        return parallel_prefill(model,tokens,docs,valid,return_trace)
    ys=[];hs=[];zs=[];kvs=[]
    for t in range(T):
        out=decode(model,tokens[:,t],docs[:,t],valid=valid[:,t],cache=cache,return_trace=return_trace)
        cache=out.cache;ys.append(out.logits)
        if return_trace:hs.append(out.hidden);zs.append(out.inputs);kvs.append(out.kv)
    kv=None
    if return_trace:
        kv={lane:[tuple(torch.cat([v[lane][i][k] for v in kvs],2) for k in (0,1))
                  for i in range(model.c.layers)] for lane in kvs[0]}
    return StreamOutput(torch.cat(ys,1),cache,torch.cat(hs,1) if return_trace else None,
                        torch.cat(zs,1) if return_trace else None,kv)
