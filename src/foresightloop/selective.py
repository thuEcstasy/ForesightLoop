# SPDX-License-Identifier: Apache-2.0
"""Cache modest-sized linear outputs while recomputing wide FFN expansions."""
import torch
from torch.utils.checkpoint import CheckpointPolicy,create_selective_checkpoint_contexts

def contexts(intermediate):
    def policy(ctx,op,*args,**kwargs):
        # Do not cache mm.out: compiled buffer writes change its version.
        # Keep PyTorch's cache mutation checks enabled; cache only fresh outputs.
        if op == torch.ops.aten.mm.default:
            if args[1].shape[1]!=intermediate:
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE
    return create_selective_checkpoint_contexts(policy)
