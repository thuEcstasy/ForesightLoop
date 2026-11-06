# SPDX-License-Identifier: Apache-2.0
"""Cycle-coordinate source gather. No current-cycle or same-loop edges."""
import math
import torch


def lag(receiver, source, offset):
    if offset not in (0, 1) or not 1 <= source < receiver:
        raise ValueError('Require offset 0/1 and a strictly shallower source')
    return receiver - source + offset


def capacities(loops, offset):
    return tuple(lag(loops, j, offset) for j in range(1, loops))


def shift_source(h, docs, valid, distance):
    """Source validity is independent for each lag, including pipeline fill."""
    if distance < 1:
        raise ValueError('Source must be from an older cycle')
    B, T, D = h.shape
    if valid is None:
        valid = torch.ones_like(docs, dtype=torch.bool)
    if distance >= T:
        return torch.zeros_like(h), torch.zeros_like(valid)
    mask = valid[:, distance:] & valid[:, :-distance]
    mask = mask & (docs[:, distance:] == docs[:, :-distance])
    # Caller supplies contiguous segment IDs, not reusable external document IDs.
    mask = torch.cat((torch.zeros_like(valid[:, :distance]), mask), 1)
    shifted = torch.cat((torch.zeros_like(h[:, :distance]), h[:, :-distance]), 1)
    return shifted.masked_fill(~mask[..., None], 0.), mask


def candidates(completed, docs, valid, offset):
    receiver = len(completed) + 1
    pairs = [shift_source(h, docs, valid, lag(receiver, j, offset))
             for j, h in enumerate(completed, 1)]
    return torch.stack([p[0] for p in pairs], -2), torch.stack([p[1] for p in pairs], -1)


def normalized_message(values, module, receiver, mask):
    """Optional norm-reuse path; math and parameter identities match aggregate."""
    x = torch.where(mask[..., None], values.float(), 0.)
    q = module.query[receiver-2].float()
    b = module.bias[receiver-2].float()
    scores = (x * q).sum(-1) / math.sqrt(module.dim) + b
    scores = scores.masked_fill(~mask, float('-inf'))
    scores = torch.where(mask.any(-1, keepdim=True), scores, torch.zeros_like(scores))
    weights = scores.softmax(-1) * mask
    return (weights[..., None] * x).sum(-2) * module.alpha[receiver-2].float(), weights


def training_input(module, embedding, completed, docs, valid=None, normalized=None):
    source, mask = candidates(completed if normalized is None else normalized,
                              docs, valid, module.source_trajectory_offset)
    receiver = len(completed) + 1
    if normalized is None:
        message, weights = module.message(source, receiver, source_mask=mask)
    else:
        message, weights = normalized_message(source, module, receiver, mask)
    collector=getattr(module,'audit_collector',None)
    if collector is not None:
        collector.append(dict(receiver=receiver,source=source,valid=mask,message=message,
                              weights=weights,normalized_sources=normalized is not None))
    return (embedding.float() + message).to(embedding.dtype)


def normalize(h, eps, source_rmsnorm=True):
    x = h.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) if source_rmsnorm else x
