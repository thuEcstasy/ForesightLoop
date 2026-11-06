# SPDX-License-Identifier: Apache-2.0
import torch
from torch.nn import functional as F
from .flash_backend import global_sdpa,Window
from .model import positions, shift
from . import primitives
VARIANT="eager"
FUSE_MIX=False
def set_variant(v):
 global VARIANT,FUSE_MIX
 FUSE_MIX=v.endswith("_mix")
 VARIANT=v.removesuffix("_mix")


def window_attention(q, history, docs, window, self_kv=None, block=64):
    batch, heads, length, dim = q.shape
    groups = heads // history[0].shape[1]
    nb = (length+block-1)//block
    qi = torch.arange(nb*block, device=q.device).reshape(nb, block)
    ki = torch.arange(nb, device=q.device)[:, None]*block + torch.arange(
        -window+1, block, device=q.device)[None, :]
    validq, validk = qi < length, (ki >= 0) & (ki < length)
    iq, ik = qi.clamp(0, length-1), ki.clamp(0, length-1)
    dq, dk = docs[:, iq], docs[:, ik]
    delta = qi[:, :, None]-ki[:, None, :]
    mask = (validq[:, :, None] & validk[:, None, :] &
            (delta >= (1 if self_kv is not None else 0)) & (delta < window))[None]
    mask = mask & (dq[:, :, :, None] == dk[:, :, None, :])
    def gather(x, idx):
        return x[:, :, idx, :].permute(0, 2, 1, 3, 4).reshape(batch*nb, x.shape[1], idx.shape[1], dim)
    qq = gather(q, iq)
    kk, vv = (gather(x, ik) for x in history)
    if self_kv is not None:
        sk, sv = (gather(x, iq) for x in self_kv)
        kk, vv = torch.cat((kk, sk), 2), torch.cat((vv, sv), 2)
        diagonal = torch.eye(block, dtype=torch.bool, device=q.device)[None, None]
        own = diagonal & validq[None, :, :, None]
        mask = torch.cat((mask, own.expand(batch, nb, block, block)), -1)
    # Padded queries have no visible keys; SDPA returns zero, then we truncate.
    out = F.scaled_dot_product_attention(qq, kk.repeat_interleave(groups, 1),
        vv.repeat_interleave(groups, 1), attn_mask=mask.reshape(batch*nb, 1, block, -1))
    return out.reshape(batch, nb, heads, block, dim).permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :length]


def hidden(model, tokens, docs, arm=None, *, single_document=False, return_trace=False, valid=None, return_details=False):
    from .sequence import layout,GraphOutput
    enabled=getattr(model,'cross_loop_aggregation','legacy')!='legacy'
    # Only the new forward adapter calls this with docs=None: it constructs full,
    # independent rows itself. Explicit document layouts retain all original guards.
    full_rows = docs is None
    if full_rows:
        if valid is not None or not single_document:
            raise ValueError('Full-row training requires single_document and no padding')
        docs = torch.zeros_like(tokens)
    if not full_rows:
        tokens,docs,valid=layout(tokens,docs,valid)
        if single_document and (not bool(valid.all()) or bool((docs!=docs[:,:1]).any())):
            raise ValueError('single_document requires unpadded independent rows')
    e = model.embedding(tokens)
    pos = positions(docs)
    phase=None
    if VARIANT in ("cached_project", "cached_full", "native_full", "norm_backward"):
        freq=model.c.theta ** (-torch.arange(0,model.c.head_dim,2,device=pos.device).float()/model.c.head_dim)
        angle=pos[:,None,:,None].float()*freq
        phase=(angle.cos(),angle.sin())
    length = tokens.shape[1]
    t = torch.arange(length, device=tokens.device)
    causal = t[:, None] >= t[None, :]
    gm = causal[None, None] if single_document else ((docs[:, :, None] == docs[:, None, :]) & causal)[:, None]
    if valid is not None:gm=gm & valid[:,None,:,None] & valid[:,None,None,:]
    trace = {}
    outputs=[];inputs=[];normalized=[]
    reuse_norm=getattr(model,'reuse_source_norm',False)
    if reuse_norm:
        serial=getattr(model,'cross_loop_source','previous_token')=='same_token'
        if not enabled or (getattr(model,'source_trajectory_offset',None) is None and not serial):
            raise ValueError('Norm-reuse candidate requires a trajectory AttnRes graph')
        from .trajectory import normalize
        if serial:
            from .serial_aggregation import training_input as trajectory_input
        else:
            from .trajectory import training_input as trajectory_input
    checkpoint_layers = getattr(model, 'checkpoint_layers', False) and torch.is_grad_enabled() and not return_trace
    def apply_layer(layer, x, gkv):
        if VARIANT in ("cached_project", "cached_full", "native_full", "norm_backward"):
            fn=primitives.project_norm_backward if VARIANT=="norm_backward" else primitives.project_native if VARIANT=="native_full" else primitives.project_cached
            q,kv,gate=fn(layer,x,phase)
        elif VARIANT in ("safe_project", "safe_full", "split_full"):
            q,kv,gate=primitives.project_safe(layer,x,pos)
        else:
            q, kv, gate = primitives.project(layer, x, pos) if VARIANT in ("project", "full") else layer.project(x, pos)
        y = finish(layer, x, q, gate, kv if gkv is None else gkv,
                   None if gkv is None else kv)
        return y, kv[0], kv[1]

    def finish(layer, x, q, gate, gkv, local=None, own=None):
        groups = model.c.heads // model.c.kv_heads
        k, v = (x.repeat_interleave(groups, 1) for x in gkv)
        # Flash causal path for document-separated production rows; explicit
        # masked SDPA remains the reference-compatible multi-document path.
        out = global_sdpa(q, k, v,
            attn_mask=None if single_document else gm,
            is_causal=single_document, dropout_p=0.0)
        if local is not None:
            if getattr(model,'native_window',False) and single_document and q.dtype==torch.bfloat16:
                lk,lv=(z.repeat_interleave(groups,1) for z in local)
                lo=Window.apply(q,lk,lv,model.c.window)
            else:lo = window_attention(q, local, docs, model.c.window, own)
            out = primitives.mix_attention(gate,lo,out) if FUSE_MIX else gate*lo + (1-gate)*out
        if VARIANT=="norm_backward" and getattr(model,"diagnostics",None) is None:
            return primitives.residual_norm_backward(layer,x,out)
        if VARIANT in ("split_residual", "split_full", "cached_full", "native_full") and getattr(model,"diagnostics",None) is None:
            return primitives.residual_split(layer,x,out)
        if VARIANT in ("residual", "full", "safe_full") and getattr(model,"diagnostics",None) is None:
            return primitives.residual_mlp(layer,x,out)
        a=layer.o(out.transpose(1, 2).flatten(-2))
        u=x+layer.gamma*a
        z=layer.n2(u)
        f=layer.down(F.silu(layer.mgate(z))*layer.up(z))
        nf=layer.ffn_output_norm(f)
        if getattr(model,'diagnostics',None) is not None:
            def rms(v):return float(v.detach().float().square().mean().sqrt())
            rx,ru=rms(x),rms(u)
            model.diagnostics.append(dict(loop=active_loop,layer=layer.index,residual_rms=rx,attention_rms=rms(a),ffn_before_rms=rms(f),ffn_after_rms=rms(nf),scaled_attention_identity_ratio=layer.gamma*rms(a)/max(rx,1e-30),scaled_ffn_identity_ratio=layer.gamma*rms(nf)/max(ru,1e-30),gain_rms=rms(layer.ffn_output_norm.weight),gain_change_rms=rms(layer.ffn_output_norm.weight-1)))
        return u+layer.gamma*nf
    def run(x, lane, gkv=None):
        cache = []
        for i, layer in enumerate(model.stack):
            if checkpoint_layers:
                from torch.utils.checkpoint import checkpoint
                kwargs={}
                if getattr(model,'selective_checkpoint',False):
                    from .selective import contexts
                    kwargs['context_fn']=lambda:contexts(model.c.intermediate)
                x, k, v = checkpoint(apply_layer, layer, x, None if gkv is None else gkv[i], use_reentrant=False,**kwargs)
                kv = (k, v)
            else:
                x, k, v = apply_layer(layer, x, None if gkv is None else gkv[i])
                kv = (k, v)
            if valid is not None:kv=tuple(t.masked_fill(~valid[:,None,:,None],0.) for t in kv)
            cache.append(kv)
        if return_trace or return_details:
            trace[lane] = cache
        return x, cache
    active_loop=1
    outer_grad=torch.is_grad_enabled()
    prefix=model.H-min(model.H,getattr(model,'bptt_window',model.H))
    if enabled and prefix:raise ValueError('AttnRes experiment requires the same full BPTT as PLT-8')
    if return_details:inputs.append(e if valid is None else e.masked_fill(~valid[...,None],0.))
    with torch.set_grad_enabled(outer_grad and prefix<1):
        g, gkv = run(e, 'G')
    if valid is not None:g=g.masked_fill(~valid[...,None],0.)
    if enabled or return_details:outputs.append(g)
    if reuse_norm:normalized.append(normalize(g,model.cross_loop_eps,model.source_rmsnorm))
    h = g
    for r in range(2, model.H+1):
        active_loop=r
        with torch.set_grad_enabled(outer_grad and r>prefix):
            z=trajectory_input(model.cross_loop,e,outputs,docs,valid,normalized) if reuse_norm else model.cross_loop.training_input(e,outputs,docs,valid) if enabled else e+(h if model.cross_loop_source=='same_token' else shift(h,docs))
            if return_details:inputs.append(z if valid is None else z.masked_fill(~valid[...,None],0.))
            h, _ = run(z, f'L{r}', gkv)
            if valid is not None:h=h.masked_fill(~valid[...,None],0.)
            if enabled or return_details:outputs.append(h)
            if reuse_norm and r<model.H:normalized.append(normalize(h,model.cross_loop_eps,model.source_rmsnorm))
    if return_details:return GraphOutput(h,tuple(outputs),tuple(inputs),trace)
    return (h, trace) if return_trace else h


def logits(model, tokens, docs, arm=None, **kw):
    result=hidden(model, tokens, docs, arm, **kw)
    return (model.head(result[0]),result[1]) if isinstance(result,tuple) else model.head(result)


def token_loss(model, h, targets):
    # FP32 loss with checkpointed head chunks avoids retaining full vocabulary
    # activations. Recompute head FLOPs are recorded in the execution ledger.
    from torch.utils.checkpoint import checkpoint
    flat, target = h.flatten(0, 1), targets.flatten()
    pieces = []
    for start in range(0, len(target), 256):
        def ce(x, y):
            return F.cross_entropy(model.head(x).float(), y, reduction='sum')
        if torch.is_grad_enabled():
            pieces.append(checkpoint(ce, flat[start:start+256], target[start:start+256], use_reentrant=False))
        else:
            pieces.append(ce(flat[start:start+256], target[start:start+256]))
    return torch.stack(pieces).sum()
