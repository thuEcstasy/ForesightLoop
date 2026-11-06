# SPDX-License-Identifier: Apache-2.0
import torch
from .stable_eval_aggregation import fixed_sum

def forward(module,x):
    value=x.float()
    mean=fixed_sum(value.square())/value.shape[-1]
    out=value*torch.rsqrt(mean[...,None]+module.eps)*module.weight.float()
    return out.to(x.dtype)

def install(all_norms=False):
    from . import model
    model.OutputNorm.forward=forward
    if all_norms:torch.nn.RMSNorm.forward=forward
