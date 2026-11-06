# SPDX-License-Identifier: Apache-2.0
"""One full tied output projection per microbatch; no token/head chunking."""
import torch
from torch.nn import functional as F

@torch.compile(dynamic=False, fullgraph=True)
def cross_entropy(logits, targets):
    return F.cross_entropy(logits.float(), targets, reduction='sum')

def token_loss(model,h,targets):
    flat,target=h.flatten(0,1),targets.flatten()
    def ce(x,y):return cross_entropy(model.head(x),y)
    if getattr(model,'head_checkpoint',False) and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(ce,flat,target,use_reentrant=False)
    return ce(flat,target)
