# SPDX-License-Identifier: Apache-2.0
"""Replay the unchanged forward/backward kernels with stable inputs and gradients.

Optimizer and distributed synchronization remain outside the graph. All PLT
loops are differentiated; no activation checkpointing or truncation is added.
"""
import gc
import torch

class CapturedBackward:
    def __init__(self, model, batch, scale, *, loss_fn=None, bf16=True):
        self.model=model
        self.batch=batch.clone()
        self.scale=scale
        self.loss_fn=loss_fn or (lambda b:model(b[:,:-1],b[:,1:]))
        self.params=[p for p in model.parameters() if p.requires_grad]
        assert all(p.grad is not None for p in self.params), 'Audit unused parameters before graph capture'
        self.grads=[p.grad for p in self.params]
        self.pointers=[(p.data_ptr(),p.grad.data_ptr()) for p in self.params]
        # Warm-up on the capture stream. Stable gradient buffers accumulate.
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch._foreach_zero_(self.grads)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                loss=self.loss_fn(self.batch)
                (loss*self.scale).backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        del loss
        gc.collect()
        torch.cuda.empty_cache()
        torch._foreach_zero_(self.grads)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                self.loss=self.loss_fn(self.batch)
                (self.loss*self.scale).backward()
        torch.cuda.synchronize()
        self.zero()

    def zero(self):
        assert [(p.data_ptr(),p.grad.data_ptr() if p.grad is not None else None) for p in self.params]==self.pointers
        torch._foreach_zero_(self.grads)

    def backward(self,batch):
        assert batch.shape==self.batch.shape and batch.dtype==self.batch.dtype
        self.batch.copy_(batch)
        self.graph.replay()
        return self.loss.detach()
