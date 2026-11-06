# SPDX-License-Identifier: Apache-2.0
"""Keep native forward rounding; compile its autograd-derived VJP only."""
import torch
from torch.nn import functional as F

def definition(x,weight,eps,output):
    if output:
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*weight.float()).to(x.dtype)
    return F.rms_norm(x,(x.shape[-1],),weight,eps)

@torch.compile(dynamic=False,fullgraph=True)
def backward_norm(grad,x,weight,eps,output):
    _,vjp=torch.func.vjp(lambda a,b:definition(a,b,eps,output),x,weight)
    return vjp(grad)

class NativeForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,eps,output):
        ctx.save_for_backward(x,weight);ctx.eps=eps;ctx.output=output
        return definition(x,weight,eps,output)
    @staticmethod
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        dx,dw=backward_norm(grad,x,weight,ctx.eps,ctx.output)
        return dx,dw,None,None

def norm(module,x,output=False):
    return NativeForward.apply(x,module.weight,module.eps,output)
