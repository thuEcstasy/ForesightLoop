# SPDX-License-Identifier: Apache-2.0
"""Serial-8: all shallower full-stack outputs at the SAME input token.

No shifted token, original embedding candidate, or separate last-hidden bypass.
Only the embedding is added to the inherited normalized AttnRes message.
"""
import torch


def training_input(module, embedding, completed, docs, valid=None, normalized=None):
    receiver = len(completed) + 1
    if not 2 <= receiver <= module.loops:
        raise ValueError('Expected all completed shallower loops')
    values = completed if normalized is None else normalized
    if len(values) != len(completed):
        raise ValueError('Normalized-source count differs from completed loop count')
    source = torch.stack(values, -2)
    mask = torch.ones(source.shape[:-1], dtype=torch.bool, device=source.device)
    if valid is not None:
        mask = mask & valid[..., None]
    if normalized is None:
        message, weights = module.message(source, receiver, source_mask=mask)
    else:
        from .trajectory import normalized_message
        message, weights = normalized_message(source, module, receiver, mask)
    collector = getattr(module, 'audit_collector', None)
    if collector is not None:
        collector.append(dict(receiver=receiver, source=source, valid=mask, message=message,
                              weights=weights, normalized_sources=normalized is not None))
    return (embedding.float() + message).to(embedding.dtype)
