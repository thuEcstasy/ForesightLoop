# SPDX-License-Identifier: Apache-2.0
"""Differentiable scalar-token oracle independent of all source/cache adapters.

Reuses only the unchanged backbone Layer equations. It never imports aggregation,
sequence, engine, serial_decode, trajectory, or their indexing helpers.
"""
import math
import torch


def mix(model, candidates, receiver):
    """Candidates are explicit (zero-based loop index, full-stack output) pairs."""
    m = model.cross_loop
    vals = [(index, (h.float() / (h.float().square().mean(-1, keepdim=True) + m.eps).sqrt() if model.source_rmsnorm else h.float()))
            for index, h in candidates]
    if not vals:
        raise ValueError('Caller must handle absent historical sources')
    q, bias = m.query[receiver-2].float(), m.bias[receiver-2].float()
    scores = torch.stack([(q * v).sum(-1) / math.sqrt(m.dim) + bias[index] for index, v in vals], -1)
    weights = scores.softmax(-1)
    return m.alpha[receiver-2].float() * sum(weights[..., i:i+1] * v for i, (_, v) in enumerate(vals))


def run(model, tokens, docs, valid=None):
    if model.cross_loop_aggregation=='attnres_fullstate':
        from .full_reference import run as full_reference
        return full_reference(model,tokens,docs,valid)
    if valid is None:
        valid = torch.ones_like(tokens, dtype=torch.bool)
    B, T = tokens.shape; R = model.H; c = model.c
    all_logits, all_hidden, all_inputs, all_kv = [], [], [], []
    source_map = []
    for b in range(B):
        history = [[[] for _ in model.stack] for _ in range(R)]
        archive = [[[] for _ in model.stack] for _ in range(R)]
        states_by_cycle = {}; last_document = None; position = 0
        logits, hidden, inputs = [], [], []
        for t in range(T):
            if not bool(valid[b, t]):
                z = model.embedding.weight.new_zeros(1, 1, R, c.dim)
                hidden.append(z); inputs.append(z)
                logits.append(model.embedding.weight.new_zeros(1, 1, c.vocab))
                for r in range(R):
                    for i in range(c.layers):
                        kv_zero = model.embedding.weight.new_zeros(1, c.kv_heads, 1, c.head_dim)
                        archive[r][i].append((kv_zero, kv_zero))
                continue
            doc = int(docs[b, t])
            if doc != last_document:
                history = [[[] for _ in model.stack] for _ in range(R)]
                states_by_cycle = {}; position = 0
            e = model.embedding(tokens[b:b+1, t:t+1])
            completed, prepared = [], []
            for r in range(1, R+1):
                pairs, indices = [], []
                if r == 1:
                    h = e
                elif model.cross_loop_aggregation == 'legacy':
                    if model.cross_loop_source == 'same_token':
                        h = e + completed[-1]
                        indices = [(position, r-1)]
                    else:
                        old = states_by_cycle.get(position-1)
                        h = e if old is None else e + old[r-2]
                        if old is not None:
                            indices = [(position-1, r-1)]
                else:
                    for j in range(1, r):
                        if getattr(model, 'cross_loop_source', 'previous_token') == 'same_token':
                            pairs.append((j-1, completed[j-1])); indices.append((position, j))
                        else:
                            offset = getattr(model, 'source_trajectory_offset', None)
                            cycle = position-1 if offset is None else position-r+j-offset
                            old = states_by_cycle.get(cycle)
                            if old is not None:
                                pairs.append((j-1, old[j-1])); indices.append((cycle, j))
                    h = (e.float() + mix(model, pairs, r)).to(e.dtype) if pairs else e
                prepared.append(h)
                source_map.append(dict(batch=b, token=t, cycle=position, receiver=r, sources=indices))
                p = torch.tensor([[position]], device=e.device)
                for i, layer in enumerate(model.stack):
                    q, kv, gate = layer.project(h, p)
                    history[r-1][i].append(kv)
                    if r > 1:
                        history[r-1][i] = history[r-1][i][-c.window:]
                    archive[r-1][i].append(kv)
                    gkv = tuple(torch.cat([v[k] for v in history[0][i]], 2) for k in (0, 1))
                    lkv = tuple(torch.cat([v[k] for v in history[r-1][i]], 2) for k in (0, 1)) if r > 1 else None
                    gm = torch.ones(1, 1, gkv[0].shape[2], device=e.device, dtype=torch.bool)
                    lm = torch.ones(1, 1, lkv[0].shape[2], device=e.device, dtype=torch.bool) if r > 1 else None
                    h = layer.finish(h, q, gate, gkv, gm, lkv, lm)
                completed.append(h)
            states_by_cycle[position] = completed
            last_document = doc; position += 1
            hidden.append(torch.stack(completed, 2)); inputs.append(torch.stack(prepared, 2))
            logits.append(model.head(completed[-1]))
        all_logits.append(torch.cat(logits, 1)); all_hidden.append(torch.cat(hidden, 1)); all_inputs.append(torch.cat(inputs, 1))
        all_kv.append([[tuple(torch.cat([v[k] for v in lane], 2) for k in (0, 1)) for lane in layers] for layers in archive])
    kv = {('G' if r == 0 else f'L{r+1}'):
          [tuple(torch.cat([v[r][i][k] for v in all_kv], 0) for k in (0, 1)) for i in range(c.layers)]
          for r in range(R)}
    return dict(logits=torch.cat(all_logits), hidden=torch.cat(all_hidden), inputs=torch.cat(all_inputs), kv=kv, sources=source_map)
