# SPDX-License-Identifier: Apache-2.0
"""CUDA replay with exactly the ordinary DDP bucket order for each update."""
import gc
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .captured import CapturedBackward

class Execution:
    def __init__(self, m, opt, batch, mb=4):
        assert batch.shape == (mb, 1025) and mb in (2, 4, 8, 16)
        assert dist.get_world_size() == 4
        self.m = m; self.opt = opt; self.mb = mb; self.params = list(m.parameters())
        self.start_step = None
        from support import rng, restore
        from training import backward
        saved_rng = rng()
        # Ask an independent ordinary DDP reducer for its real rebuilt layout.
        # Two backward-only warmups change neither parameters nor Adam state.
        wrapped = DDP(m, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        repeated = batch.repeat(32 // mb, 1)
        for _ in range(2):
            m.zero_grad(set_to_none=True)
            backward(wrapped, repeated[:, :-1], repeated[:, 1:], mb, 4, True)
        ids = {id(p): i for i, p in enumerate(self.params)}
        bucket_objects = wrapped.reducer._get_zeros_like_grad_buckets()
        self.layout = [[ids[id(p)] for p in b.parameters()] for b in bucket_objects]
        assert sorted(i for b in self.layout for i in b) == list(range(len(self.params)))
        del bucket_objects, wrapped, repeated
        m.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); restore(saved_rng)
        self.flat = torch.zeros(sum(p.numel() for p in self.params), device='cuda', dtype=torch.float32)
        self.buckets = []; offset = 0
        for indices in self.layout:
            start = offset
            for index in indices:
                p = self.params[index]; assert p.dtype == torch.float32
                p.grad = self.flat[offset:offset+p.numel()].view_as(p); offset += p.numel()
            self.buckets.append(self.flat[start:offset])
        self.runner = CapturedBackward(m, batch, 4 / 131072)
        self.reserved_after_capture = torch.cuda.memory_reserved()
        self.allocated_after_capture = torch.cuda.memory_allocated()

    def update(self, x, y, step):
        assert self.m.training and x.shape == y.shape == (32, 1024) and not (y == -100).any()
        if self.start_step is None: self.start_step = step
        assert step >= self.start_step
        for group in self.opt.param_groups: group['lr'] = 3e-4 * min((step+1) / 128, 1.)
        torch.cuda.synchronize(); started = time.perf_counter(); self.runner.zero()
        total = torch.zeros((), device='cuda')
        for start in range(0, 32, self.mb):
            batch = torch.cat((x[start:start+self.mb], y[start:start+self.mb, -1:]), 1)
            assert torch.equal(batch[:, 1:], y[start:start+self.mb])
            total += self.runner.backward(batch)
        self.flat.div_(4)
        if step == self.start_step:
            # A fresh DDP reducer initially has one bucket in parameter order.
            # Choose by logical step so in-place checkpoint replay is repeatable.
            grads = [p.grad for p in self.params]
            first = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(first)
            for grad, part in zip(grads, torch._utils._unflatten_dense_tensors(first, grads)):
                grad.copy_(part)
            del first, grads, grad, part
        else:
            for bucket in self.buckets: dist.all_reduce(bucket)
        dist.all_reduce(total)
        assert torch.isfinite(total)
        norm = torch.nn.utils.clip_grad_norm_(self.params, 1., error_if_nonfinite=True)
        self.opt.step(); torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter()-started, device='cuda', dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return dict(loss=float(total)/131072, tokens=131072, grad_norm=float(norm), lr=self.opt.param_groups[0]['lr'], seconds=float(elapsed))
