# SPDX-License-Identifier: Apache-2.0
"""Independent scalar aggregation and differentiable token-by-token PLT oracle.

Does not import aggregation, sequence, decode or train_graph implementation.
Reuses the unchanged backbone Layer equations, as the original cached oracle did.
"""
import math
import torch


def message(model, archive, receiver):
    # Independent trajectory coordinates; no production source-spec/gather imports.
    module=model.cross_loop;k=receiver;K=model.H;p=len(archive);target=p+K+1-k
    values=[];scores=[];q=module.query[k-2].float();bias=module.bias[k-2].float()
    for j in range(1,k):
        for source_trajectory in range(target-1,target+k-j):
            source_cycle=source_trajectory+j-K-1
            if source_cycle<0:continue
            assert source_cycle<p
            delta=p-source_cycle;idx=sum(k-jj+1 for jj in range(1,j))+delta-1
            h=archive[source_cycle][j-1].float();v=h/(h.square().mean(-1,keepdim=True)+module.eps).sqrt() if model.source_rmsnorm else h
            values.append(v);scores.append((v*q).sum(-1)/math.sqrt(q.numel())+bias[idx])
    assert values
    weights=torch.stack(scores,-1).softmax(-1)
    return module.alpha[k-2].float()*sum(weights[...,i:i+1]*v for i,v in enumerate(values))


def run(model, tokens, docs, valid=None):
    """All history remains differentiable; invalid positions do not update state."""
    if valid is None: valid = torch.ones_like(tokens, dtype=torch.bool)
    B,T = tokens.shape; R=model.H; c=model.c
    batch_logits=[]; batch_hidden=[]; batch_inputs=[]; batch_kv=[]
    def join(items):return tuple(torch.cat([kv[j] for kv in items], 2) for j in (0,1))
    for b in range(B):
        previous=None; completed_cycles=[]; last_doc=None; pos=0
        history=[[[] for _ in model.stack] for _ in range(R)]
        logits=[]; hidden=[]; inputs=[]
        archive=[[[] for _ in model.stack] for _ in range(R)]
        for t in range(T):
            if not bool(valid[b,t]):
                logits.append(model.embedding.weight.new_zeros(1,1,c.vocab))
                hidden.append(model.embedding.weight.new_zeros(1,1,R,c.dim))
                inputs.append(model.embedding.weight.new_zeros(1,1,R,c.dim))
                for r in range(R):
                    for i in range(c.layers):
                        z=model.embedding.weight.new_zeros(1,c.kv_heads,1,c.head_dim)
                        archive[r][i].append((z,z))
                continue
            document=int(docs[b,t])
            if document != last_doc:
                previous=None; completed_cycles=[]; pos=0
                history=[[[] for _ in model.stack] for _ in range(R)]
            e=model.embedding(tokens[b:b+1,t:t+1])
            frozen=completed_cycles
            states=[e]
            for r in range(2,R+1):
                if not frozen: z=e
                elif model.cross_loop_aggregation=='legacy': z=e+frozen[r-2]
                else: z=(e.float()+message(model,frozen,r)).to(e.dtype)
                states.append(z)
            inputs.append(torch.stack(states,2))
            for i,layer in enumerate(model.stack):
                for r in range(R):
                    p=torch.tensor([[pos]],device=e.device)
                    q,kv,gate=layer.project(states[r],p)
                    history[r][i]=history[r][i]+[kv]
                    if r:history[r][i]=history[r][i][-c.window:]
                    archive[r][i].append(kv)
                    gkv=join(history[0][i]);lkv=join(history[r][i]) if r else None
                    gm=torch.ones(1,1,gkv[0].shape[2],dtype=torch.bool,device=e.device)
                    lm=torch.ones(1,1,lkv[0].shape[2],dtype=torch.bool,device=e.device) if r else None
                    states[r]=layer.finish(states[r],q,gate,gkv,gm,lkv,lm)
            previous=tuple(states);completed_cycles.append(previous);last_doc=document;pos+=1
            hidden.append(torch.stack(states,2));logits.append(model.head(states[-1]))
        batch_logits.append(torch.cat(logits,1));batch_hidden.append(torch.cat(hidden,1));batch_inputs.append(torch.cat(inputs,1))
        batch_kv.append([[join(v) for v in layers] for layers in archive])
    kv={('G' if r==0 else f'L{r+1}'):[tuple(torch.cat([v[r][i][j] for v in batch_kv],0) for j in (0,1))
                                    for i in range(c.layers)] for r in range(R)}
    return dict(logits=torch.cat(batch_logits),hidden=torch.cat(batch_hidden),inputs=torch.cat(batch_inputs),kv=kv)
