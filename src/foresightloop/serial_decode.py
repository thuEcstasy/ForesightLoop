# SPDX-License-Identifier: Apache-2.0
"""Serial cached execution: finish the full stack before preparing the next loop.

The first loop publishes the same global KV; subsequent loops append only their
own local-window KV. The input cache is immutable. Only per-token loop outputs
are used by AttnRes; old.prev_hidden is never a residual input.
"""
import torch
from .checkpoint import signature


def one(model, token, document, old, order=None):
    from .decode import RequestState
    R, c = model.H, model.c
    if order is not None and tuple(order) != tuple(range(1, R + 1)):
        raise ValueError('Serial loops have a strict shallower-to-deeper dependency')
    e = model.embedding(token.reshape(1, 1))
    if old is not None and old.document != document:
        old = None
    if old is not None and old.kv[0][0][0].device != e.device:
        raise ValueError('Cache device mismatch')
    position = 0 if old is None else old.position + 1
    past = old.kv if old is not None else tuple(tuple(None for _ in model.stack) for _ in range(R))
    native = [[None] * c.layers for _ in range(R)]
    updated = [[None] * c.layers for _ in range(R)]
    outputs, inputs = [], []
    pos = torch.tensor([[position]], device=e.device)
    for r in range(R):
        if r == 0:
            h = e
        elif model.cross_loop_aggregation == 'legacy':
            # Exact Serial-4 recurrence, retained as an audit/control path.
            h = e + outputs[-1]
        else:
            message, _ = model.cross_loop.message(torch.stack(outputs, -2), r + 1)
            h = (e.float() + message).to(e.dtype)
        inputs.append(h)
        for i, layer in enumerate(model.stack):
            q, kv, gate = layer.project(h, pos)
            native[r][i] = kv
            previous = past[r][i]
            joined = kv if previous is None else tuple(torch.cat((a, b), 2) for a, b in zip(previous, kv))
            updated[r][i] = tuple(v[:, :, -c.window:] for v in joined) if r else joined
            global_kv = updated[0][i]
            local_kv = updated[r][i] if r else None
            gm = torch.ones(1, 1, global_kv[0].shape[2], dtype=torch.bool, device=e.device)
            lm = torch.ones(1, 1, local_kv[0].shape[2], dtype=torch.bool, device=e.device) if r else None
            h = layer.finish(h, q, gate, global_kv, gm, local_kv, lm)
        outputs.append(h)
    new = RequestState(tuple(outputs), tuple(tuple(v) for v in updated), document, position)
    return model.head(outputs[-1]), new, tuple(inputs), native


@torch.no_grad()
def prefill(model, tokens, docs=None, *, valid=None, cache=None, return_trace=False):
    from .decode import BatchCache, RequestState, StreamOutput, prefill as continuation
    from .sequence import layout
    from .engine import hidden
    if tokens.ndim != 2:
        raise ValueError('prefill expects [B,T]')
    if cache is not None:
        # The shared wrapper dispatches each token to the serial `one` above.
        return continuation(model, tokens, docs, valid=valid, cache=cache, return_trace=return_trace)
    if docs is None:
        docs = torch.zeros_like(tokens)
    if tokens.shape[1] == 0:
        return continuation(model, tokens, docs, valid=valid, return_trace=return_trace)
    safe, segments, valid = layout(tokens, docs, valid)
    full = bool(valid.all()) and bool((segments == segments[:, :1]).all())
    result = hidden(model, safe, None if full else segments, valid=None if full else valid,
                    single_document=full, return_details=True)
    rows = []
    for b in range(len(tokens)):
        indices = valid[b].nonzero().flatten()
        if not len(indices):
            rows.append(None)
            continue
        last = int(indices[-1]); segment = segments[b, last]
        indices = indices[segments[b, indices] == segment]
        first = int(indices[0]); size = len(indices)
        kv = []
        for r in range(model.H):
            lane = 'G' if r == 0 else f'L{r + 1}'
            start = first if r == 0 else max(first, last - model.c.window + 1)
            kv.append(tuple(tuple(v[b:b+1, :, start:last+1].clone() for v in pair) for pair in result.kv[lane]))
        outputs = tuple(h[b:b+1, last:last+1].clone() for h in result.loops)
        rows.append(RequestState(outputs, tuple(kv), int(docs[b, last]), size - 1))
    return StreamOutput(model.head(result.hidden), BatchCache(tuple(rows), signature(model)),
                        torch.stack(result.loops, 2) if return_trace else None,
                        torch.stack(result.inputs, 2) if return_trace else None,
                        result.kv if return_trace else None)
