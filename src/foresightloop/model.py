# SPDX-License-Identifier: Apache-2.0
"""Independent dense and layer-synchronous cached PLT references; no stall paths."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int = 151936  # Qwen3-0.6B padded vocabulary
    dim: int = 1024
    layers: int = 28
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 3072
    window: int = 64
    eps: float = 1e-6
    theta: float = 1000000.0


def positions(docs):
    t = torch.arange(docs.shape[1], device=docs.device).expand_as(docs)
    starts = torch.cat((torch.ones_like(docs[:, :1], dtype=torch.bool),
                        docs[:, 1:] != docs[:, :-1]), dim=1)
    return t-torch.where(starts, t, 0).cummax(dim=1).values


def shift(x, docs):
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1] *
                      (docs[:, 1:] == docs[:, :-1]).unsqueeze(-1)), dim=1)


def rope(x, pos, theta):
    freq = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float()/x.shape[-1])
    angle = pos[:, None, :, None].float() * freq
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), -1).to(x.dtype)


def attend(q, kv, mask, groups):
    k, v = (x.repeat_interleave(groups, dim=1) for x in kv)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], dropout_p=0.0)


class OutputNorm(nn.RMSNorm):
    def forward(self, x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1, keepdim=True)+self.eps)*self.weight.float()).to(x.dtype)


class Layer(nn.Module):
    def __init__(self, c, gamma):
        super().__init__()
        self.c = c
        self.gamma = gamma
        self.ffn_output_norm = OutputNorm(c.dim, eps=c.eps)
        self.n1 = nn.RMSNorm(c.dim, eps=c.eps)
        self.n2 = nn.RMSNorm(c.dim, eps=c.eps)
        self.q_norm = nn.RMSNorm(c.head_dim, eps=c.eps)
        self.k_norm = nn.RMSNorm(c.head_dim, eps=c.eps)
        self.q = nn.Linear(c.dim, c.heads*c.head_dim, bias=False)
        self.k = nn.Linear(c.dim, c.kv_heads*c.head_dim, bias=False)
        self.v = nn.Linear(c.dim, c.kv_heads*c.head_dim, bias=False)
        self.o = nn.Linear(c.heads*c.head_dim, c.dim, bias=False)
        self.up = nn.Linear(c.dim, c.intermediate, bias=False)
        self.mgate = nn.Linear(c.dim, c.intermediate, bias=False)
        self.down = nn.Linear(c.intermediate, c.dim, bias=False)
        self.gate_weight = nn.Parameter(torch.zeros(c.heads, c.head_dim))
        self.gate_bias = nn.Parameter(torch.zeros(c.heads))

    def project(self, x, pos):
        z = self.n1(x)
        def reshape(y, heads):
            return y.reshape(*y.shape[:2], heads, self.c.head_dim).transpose(1, 2)
        q = self.q_norm(reshape(self.q(z), self.c.heads))
        k = self.k_norm(reshape(self.k(z), self.c.kv_heads))
        v = reshape(self.v(z), self.c.kv_heads)
        # Gate is applied to the projected Q before rotary positional rotation.
        gate = torch.sigmoid((q*self.gate_weight[None, :, None]).sum(-1) +
                             self.gate_bias[None, :, None]).unsqueeze(-1)
        return rope(q, pos, self.c.theta), (rope(k, pos, self.c.theta), v), gate

    def finish(self, x, q, gate, global_kv, gm, local_kv=None, lm=None):
        groups = self.c.heads // self.c.kv_heads
        out = attend(q, global_kv, gm, groups)
        if local_kv is not None:
            local = attend(q, local_kv, lm, groups)
            out = gate*local + (1-gate)*out
        x = x + self.gamma*self.o(out.transpose(1, 2).flatten(-2))
        z = self.n2(x)
        return x + self.gamma*self.ffn_output_norm(self.down(F.silu(self.mgate(z))*self.up(z)))


class Model(nn.Module):
    def __init__(self, c=Config(), H=1, lam=0.5, *, cross_loop_aggregation='legacy', cross_loop_eps=None, source_trajectory_offset=None, cross_loop_source='previous_token', source_rmsnorm=True):
        super().__init__()
        assert c.heads % c.kv_heads == 0 and c.head_dim % 2 == 0
        assert H in (1,2,4,8,16) and lam == 0.5
        from .aggregation import MODES
        if cross_loop_aggregation not in MODES:raise ValueError('Unknown cross_loop_aggregation')
        self.cross_loop_aggregation=cross_loop_aggregation
        self.source_rmsnorm=bool(source_rmsnorm)
        if cross_loop_source not in ('previous_token', 'same_token'):
            raise ValueError('Unknown cross-loop hidden source')
        if cross_loop_source=='same_token' and (source_trajectory_offset is not None or cross_loop_aggregation not in ('legacy','attnres')):
            raise ValueError('Serial source supports legacy or AttnRes, without a PLT trajectory offset')
        self.cross_loop_source=cross_loop_source
        if source_trajectory_offset not in (None,0,1):raise ValueError('Invalid trajectory offset')
        if source_trajectory_offset is not None and cross_loop_aggregation!='attnres':raise ValueError('Trajectory arms require the inherited AttnRes module')
        self.source_trajectory_offset=source_trajectory_offset
        self.cross_loop_eps=c.eps if cross_loop_eps is None else float(cross_loop_eps)
        if not math.isfinite(self.cross_loop_eps) or self.cross_loop_eps<=0:raise ValueError('Invalid cross_loop_eps')
        self.H, self.lam, self.gamma = H, lam, lam/(H*math.sqrt(c.layers))
        self.c = c
        self.embedding = nn.Embedding(c.vocab, c.dim)
        self.stack = nn.ModuleList(Layer(c, self.gamma) for _ in range(c.layers))
        for i,layer in enumerate(self.stack):layer.index=i
        self.final_norm = nn.RMSNorm(c.dim, eps=c.eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for layer in self.stack:
            with torch.no_grad():
                layer.o.weight.div_(math.sqrt(2*c.layers))
                layer.down.weight.div_(math.sqrt(2*c.layers))
        # Only enabled modes register parameters; legacy RNG/state_dict are unchanged.
        if cross_loop_aggregation=='attnres_fullstate':
            from .full_sources import SourceSetAggregation
            self.cross_loop=SourceSetAggregation(c.dim,H,self.cross_loop_eps,'colored_full',source_rmsnorm=self.source_rmsnorm)
        elif cross_loop_aggregation!='legacy':
            from .aggregation import CrossLoopAggregation
            self.cross_loop=CrossLoopAggregation(c.dim,H,cross_loop_aggregation,self.cross_loop_eps,source_trajectory_offset,cross_loop_source,source_rmsnorm=self.source_rmsnorm)

    def head(self, x):
        return F.linear(self.final_norm(x), self.embedding.weight)

    def vectorized(self, tokens, docs, return_trace=False, *, valid=None):
        # Canonical segment IDs prevent repeated external document IDs from
        # connecting noncontiguous documents, including in the plain arm.
        from .sequence import layout
        tokens,docs,valid=layout(tokens,docs,valid)
        if self.cross_loop_aggregation!='legacy' or valid is not None:
            from .sequence import dense
            out=dense(self,tokens,docs,valid);logits=self.head(out.hidden)
            return (logits,out.kv) if return_trace else logits
        e,pos=self.embedding(tokens),positions(docs)
        t=torch.arange(tokens.shape[1],device=tokens.device)
        delta=t[:,None]-t[None,:]
        gm=(docs[:,:,None]==docs[:,None,:]) & (delta>=0)
        lm=gm & (delta<self.c.window)
        trace={}; h=e; global_kv=None
        for r in range(self.H):
            if r: h=e+(h if self.cross_loop_source=='same_token' else shift(h,docs))
            kvs=[]
            for i,layer in enumerate(self.stack):
                q,kv,gate=layer.project(h,pos);kvs.append(kv)
                h=layer.finish(h,q,gate,kv if r==0 else global_kv[i],gm,None if r==0 else kv,lm)
            if r==0:global_kv=kvs
            trace['G' if r==0 else f'L{r+1}']=kvs
        logits=self.head(h)
        return (logits,trace) if return_trace else logits

    def cached(self,tokens,docs,return_trace=False, *, valid=None):
        if self.cross_loop_source=='same_token':
            out=self.prefill(tokens,docs,valid=valid,return_trace=return_trace)
            return (out.logits,out.kv) if return_trace else out.logits
        if self.cross_loop_aggregation!='legacy' or valid is not None:
            out=self.prefill(tokens,docs,valid=valid,return_trace=return_trace)
            return (out.logits,out.kv) if return_trace else out.logits
        assert tokens.shape[0]==1
        e,pos=self.embedding(tokens),positions(docs)
        archive=[[[] for _ in self.stack] for _ in range(self.H)]
        outputs=[]
        def join(items):return tuple(torch.cat([kv[j] for kv in items],2) for j in (0,1))
        for t in range(tokens.shape[1]):
            if t==0 or docs[0,t]!=docs[0,t-1]:
                history=[[[] for _ in self.stack] for _ in range(self.H)]
                previous=[torch.zeros_like(e[:,:1]) for _ in range(self.H)]
            # All inputs come from the immutable previous-position snapshot.
            old=tuple(previous)
            states=[e[:,t:t+1]]+[e[:,t:t+1]+old[r-1] for r in range(1,self.H)]
            # Publish G KV at this layer, before advancing to the next layer.
            for i,layer in enumerate(self.stack):
                for r in range(self.H):
                    q,kv,gate=layer.project(states[r],pos[:,t:t+1])
                    history[r][i]=history[r][i]+[kv]
                    if r:history[r][i]=history[r][i][-self.c.window:]
                    archive[r][i].append(kv)
                    gkv=join(history[0][i]);lkv=join(history[r][i]) if r else None
                    gm=torch.ones((1,1,gkv[0].shape[2]),device=e.device,dtype=torch.bool)
                    lm=torch.ones((1,1,lkv[0].shape[2]),device=e.device,dtype=torch.bool) if r else None
                    states[r]=layer.finish(states[r],q,gate,gkv,gm,lkv,lm)
            previous=states
            outputs.append(self.head(states[-1]))
        trace={'G' if r==0 else f'L{r+1}':[join(kvs) for kvs in layers] for r,layers in enumerate(archive)}
        result=torch.cat(outputs,1)
        return (result,trace) if return_trace else result

    def forward(self,tokens,targets):
        from .train_graph import hidden,token_loss
        docs=torch.zeros_like(tokens)
        return token_loss(self,hidden(self,tokens,docs,single_document=True),targets)

    def prefill(self,tokens,docs=None,*,valid=None,cache=None,return_trace=False):
        if self.cross_loop_aggregation=='attnres_fullstate':
            from .full_decode import prefill
        elif self.cross_loop_source=='same_token':
            from .serial_decode import prefill
        elif self.source_trajectory_offset is not None:
            from .trajectory_decode import prefill
        else:
            from .decode import prefill
        return prefill(self,tokens,docs,valid=valid,cache=cache,return_trace=return_trace)

    def decode(self,tokens,docs=None,*,valid=None,cache=None,return_trace=False):
        if self.cross_loop_aggregation=='attnres_fullstate':
            from .full_decode import decode
        elif self.source_trajectory_offset is not None:
            from .trajectory_decode import decode
        else:
            from .decode import decode
        return decode(self,tokens,docs,valid=valid,cache=cache,return_trace=return_trace)

    def sequence_loss(self,tokens,docs,valid=None):
        from .train_graph import hidden
        from .sequence import next_token_loss
        h=hidden(self,tokens,docs,valid=valid if valid is not None else torch.ones_like(tokens,dtype=torch.bool))
        return next_token_loss(self.head(h),tokens,docs,valid)
