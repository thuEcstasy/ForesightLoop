# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import functional as F
from .flash_backend import global_sdpa,Window
from .model import positions, shift, rope
torch._inductor.config.emulate_precision_casts=True
torch._dynamo.config.recompile_limit=64
torch._dynamo.config.accumulated_recompile_limit=256

# Compile stateless pieces without changing the model's checkpoint/state keys.
@torch.compile(dynamic=False, fullgraph=True)
def project_base(layer, x, pos):
    z = layer.n1(x)
    def shape(y, heads):
        return y.reshape(*y.shape[:2], heads, layer.c.head_dim).transpose(1, 2)
    q = layer.q_norm(shape(layer.q(z), layer.c.heads))
    k = layer.k_norm(shape(layer.k(z), layer.c.kv_heads))
    v = shape(layer.v(z), layer.c.kv_heads)
    return rope(q, pos, layer.c.theta), (rope(k, pos, layer.c.theta), v), q

@torch.compile(dynamic=False, fullgraph=True)
def project_gate(layer, q):
    return torch.sigmoid((q*layer.gate_weight[None, :, None]).sum(-1) +
                         layer.gate_bias[None, :, None]).unsqueeze(-1)

def project(layer, x, pos):
    q, kv, before_rope = project_base(layer, x, pos)
    # Keep the gate outside the multi-output compiled projection: an unused
    # gate must keep grad=None, not materialized zeros/extra Adam state.
    return q, kv, project_gate(layer, before_rope)

@torch.compile(dynamic=False, fullgraph=True)
def residual_mlp(layer, x, out):
    x = x+layer.gamma*layer.o(out.transpose(1, 2).flatten(-2))
    z = layer.n2(x)
    return x+layer.gamma*layer.ffn_output_norm(layer.down(F.silu(layer.mgate(z))*layer.up(z)))

@torch.compile(dynamic=False, fullgraph=True)
def project_linears(layer, x):
    z=layer.n1(x)
    def shape(y, heads):
        return y.reshape(*y.shape[:2], heads, layer.c.head_dim).transpose(1,2)
    return shape(layer.q(z),layer.c.heads),shape(layer.k(z),layer.c.kv_heads),shape(layer.v(z),layer.c.kv_heads)

@torch.compile(dynamic=False, fullgraph=True)
def rotate(q,k,pos,theta):
    return rope(q,pos,theta),rope(k,pos,theta)

def project_safe(layer,x,pos):
    q,k,v=project_linears(layer,x)
    # Retain native head RMSNorm rounding and backward; fuse independent work.
    q,k=layer.q_norm(q),layer.k_norm(k)
    gate=project_gate(layer,q)
    q,k=rotate(q,k,pos,layer.c.theta)
    return q,(k,v),gate

@torch.compile(dynamic=False, fullgraph=True)
def mlp_only(layer,z):
    return layer.down(F.silu(layer.mgate(z))*layer.up(z))

@torch.compile(dynamic=False, fullgraph=True)
def add_scaled(x,a,gamma):
    return x+gamma*a

@torch.compile(dynamic=False, fullgraph=True)
def output_norm(norm,x):
    return norm(x)

def residual_split(layer,x,out):
    a=layer.o(out.transpose(1,2).flatten(-2))
    u=add_scaled(x,a,layer.gamma)
    z=layer.n2(u)
    f=mlp_only(layer,z)
    nf=layer.ffn_output_norm(f)
    return add_scaled(u,nf,layer.gamma)

@torch.compile(dynamic=False, fullgraph=True)
def project_normalized(layer,z):
    def shape(y,h):return y.reshape(*y.shape[:2],h,layer.c.head_dim).transpose(1,2)
    return layer.q_norm(shape(layer.q(z),layer.c.heads)),layer.k_norm(shape(layer.k(z),layer.c.kv_heads)),shape(layer.v(z),layer.c.kv_heads)

@torch.compile(dynamic=False, fullgraph=True)
def rotate_cached(q,k,cos,sin):
    def apply(x):
        a,b=x.chunk(2,-1)
        return torch.cat((a*cos-b*sin,a*sin+b*cos),-1).to(x.dtype)
    return apply(q),apply(k)

def project_cached(layer,x,phase):
    q,k,v=project_normalized(layer,layer.n1(x))
    gate=project_gate(layer,q)
    q,k=rotate_cached(q,k,*phase)
    return q,(k,v),gate

def project_native(layer,x,phase):
    z=layer.n1(x)
    def shape(y,h):return y.reshape(*y.shape[:2],h,layer.c.head_dim).transpose(1,2)
    q=layer.q_norm(shape(layer.q(z),layer.c.heads))
    k=layer.k_norm(shape(layer.k(z),layer.c.kv_heads))
    v=shape(layer.v(z),layer.c.kv_heads)
    gate=project_gate(layer,q)
    q,k=rotate_cached(q,k,*phase)
    return q,(k,v),gate

def project_norm_backward(layer,x,phase):
    from .norm_backward import norm
    z=norm(layer.n1,x)
    def shape(y,h):return y.reshape(*y.shape[:2],h,layer.c.head_dim).transpose(1,2)
    q=norm(layer.q_norm,shape(layer.q(z),layer.c.heads))
    k=norm(layer.k_norm,shape(layer.k(z),layer.c.kv_heads))
    v=shape(layer.v(z),layer.c.kv_heads)
    gate=project_gate(layer,q)
    q,k=rotate_cached(q,k,*phase)
    return q,(k,v),gate

def residual_norm_backward(layer,x,out):
    from .norm_backward import norm
    a=layer.o(out.transpose(1,2).flatten(-2))
    u=add_scaled(x,a,layer.gamma)
    z=norm(layer.n2,u)
    f=mlp_only(layer,z)
    return add_scaled(u,norm(layer.ffn_output_norm,f,True),layer.gamma)

@torch.compile(dynamic=False,fullgraph=True)
def mix_attention(gate,local,global_output):
    return gate*local+(1-gate)*global_output
