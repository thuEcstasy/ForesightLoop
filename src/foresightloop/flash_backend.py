# SPDX-License-Identifier: Apache-2.0
"""Original SDPA Flash global path; optional native sliding-window Flash kernel."""
import torch,math
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
class Window(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,window):
  q,k,v=(t.transpose(1,2) for t in (q,k,v));nq,nk=q.shape[1],k.shape[1]
  out,lse,seed,offset,_=torch.ops.aten._flash_attention_forward.default(q,k,v,None,None,nq,nk,0.,True,False,window_size_left=window-1,window_size_right=0)
  ctx.save_for_backward(q,k,v,out,lse,seed,offset);ctx.window=window
  return out.transpose(1,2)
 @staticmethod
 def backward(ctx,grad):
  q,k,v,out,lse,seed,offset=ctx.saved_tensors;cu=torch.empty(0,device=q.device,dtype=torch.int32)
  dq,dk,dv=torch.ops.aten._flash_attention_backward.default(grad.transpose(1,2).contiguous(),q,k,v,out,lse,None,None,q.shape[1],k.shape[1],0.,True,seed,offset,window_size_left=ctx.window-1,window_size_right=0)
  return dq.transpose(1,2),dk.transpose(1,2),dv.transpose(1,2),None

def global_sdpa(q,k,v,attn_mask=None,is_causal=False,**kw):
 if q.dtype in (torch.bfloat16,torch.float16) and attn_mask is None:
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):return F.scaled_dot_product_attention(q,k,v,is_causal=is_causal,**kw)
 with sdpa_kernel(SDPBackend.MATH):return F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,is_causal=is_causal,**kw)

def reference_attention(q,kv,mask,groups,window=None,native_window=False):
 k,v=(t.repeat_interleave(groups,1) for t in kv)
 nq,nk=q.shape[2],k.shape[2]
 all_visible=bool(mask.all())
 if window is None:
  causal=nq==nk and bool(torch.equal(mask,torch.ones_like(mask).tril()))
  if all_visible:return global_sdpa(q,k,v,is_causal=False)
  if causal:return global_sdpa(q,k,v,is_causal=True)
 elif native_window and q.dtype==torch.bfloat16:
  if all_visible:return global_sdpa(q,k,v,is_causal=False)
  t=torch.arange(nq,device=q.device);delta=t[:,None]-t[None,:]
  if nq==nk and torch.equal(mask,((delta>=0)&(delta<window))[None].expand_as(mask)):return Window.apply(q,k,v,window)
 with sdpa_kernel(SDPBackend.MATH):return F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None])

def install_reference(native_window=False):
 from .model import Layer
 def finish(layer,x,q,gate,global_kv,gm,local_kv=None,lm=None):
  groups=layer.c.heads//layer.c.kv_heads
  out=reference_attention(q,global_kv,gm,groups)
  if local_kv is not None:
   local=reference_attention(q,local_kv,lm,groups,layer.c.window,native_window)
   out=gate*local+(1-gate)*out
  x=x+layer.gamma*layer.o(out.transpose(1,2).flatten(-2));z=layer.n2(x)
  return x+layer.gamma*layer.ffn_output_norm(layer.down(F.silu(layer.mgate(z))*layer.up(z)))
 Layer.finish=finish
