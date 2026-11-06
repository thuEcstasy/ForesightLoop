# SPDX-License-Identifier: Apache-2.0
"""Independent scalar aggregation and differentiable token-by-token PLT oracle.

Does not import aggregation, sequence, decode or train_graph implementation.
Reuses the unchanged backbone Layer equations, as the original cached oracle did.
"""
import math
import torch


def message(model, previous, receiver, available=None):
    module = model.cross_loop
    available = list(range(receiver-1)) if available is None else available
    sources = [previous[j] for j in available]
    values = [h.float() / (h.float().square().mean(-1, keepdim=True)+module.eps).sqrt()
              for h in sources]
    alpha = module.alpha[receiver-2].float()
    if module.mode == 'normalized_chain':
        return alpha * values[-1]
    q, b = module.query[receiver-2].float(), module.bias[receiver-2].float()
    scores = torch.stack([(q*v).sum(-1)/math.sqrt(q.numel())+b[source_index]
                          for source_index, v in zip(available,values)], -1)
    weights = scores.softmax(-1)
    return alpha * sum(weights[..., j:j+1]*v for j, v in enumerate(values))


def run(model, tokens, docs, valid=None):
    """All history remains differentiable; invalid positions do not update state."""
    if valid is None: valid = torch.ones_like(tokens, dtype=torch.bool)
    B,T = tokens.shape; R=model.H; c=model.c
    batch_logits=[]; batch_hidden=[]; batch_inputs=[]; batch_kv=[]
    def join(items):return tuple(torch.cat([kv[j] for kv in items], 2) for j in (0,1))
    for b in range(B):
        previous=None; last_doc=None; pos=0; trajectories={}
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
                previous=None; pos=0; trajectories={}
                history=[[[] for _ in model.stack] for _ in range(R)]
            e=model.embedding(tokens[b:b+1,t:t+1])
            frozen=previous
            states=[e]
            for r in range(2,R+1):
                if getattr(model,'source_trajectory_offset',None) is not None:
                    # Independent G[origin,j] dictionary, not a cycle-coordinate shift.
                    origin=pos-r+1-model.source_trajectory_offset
                    found=[j for j in range(1,r) if (origin,j) in trajectories]
                    if found:
                        source={j-1:trajectories[(origin,j)] for j in found}
                        z=(e.float()+message(model,source,r,[j-1 for j in found])).to(e.dtype)
                    else:z=e
                elif frozen is None: z=e
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
            for r,state in enumerate(states,1):trajectories[(pos-r+1,r)]=state
            previous=tuple(states);last_doc=document;pos+=1
            hidden.append(torch.stack(states,2));logits.append(model.head(states[-1]))
        batch_logits.append(torch.cat(logits,1));batch_hidden.append(torch.cat(hidden,1));batch_inputs.append(torch.cat(inputs,1))
        batch_kv.append([[join(v) for v in layers] for layers in archive])
    kv={('G' if r==0 else f'L{r+1}'):[tuple(torch.cat([v[r][i][j] for v in batch_kv],0) for j in (0,1))
                                    for i in range(c.layers)] for r in range(R)}
    return dict(logits=torch.cat(batch_logits),hidden=torch.cat(batch_hidden),inputs=torch.cat(batch_inputs),kv=kv)
