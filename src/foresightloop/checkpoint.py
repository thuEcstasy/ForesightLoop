# SPDX-License-Identifier: Apache-2.0
"""Explicit old-weight migration; never broadly suppress missing backbone keys."""
from dataclasses import asdict
import torch


def load_legacy_state(model,state,*,allow_new_aggregation=False):
    if not allow_new_aggregation:
        model.load_state_dict(state,strict=True)
        return []
    if model.cross_loop_aggregation=='legacy':
        raise ValueError('New-parameter initialization requested for a legacy model')
    expected=model.state_dict();added=[k for k in expected if k.startswith('cross_loop.')]
    backbone=set(expected)-set(added)
    if set(state)!=backbone:
        raise ValueError(f'Backbone key mismatch: missing={sorted(backbone-set(state))}, unexpected={sorted(set(state)-backbone)}')
    for k,t in state.items():
        if t.shape!=expected[k].shape:raise ValueError(f'Backbone shape mismatch: {k}')
    merged=dict(state)
    for k in added:merged[k]=torch.ones_like(expected[k]) if k.startswith('cross_loop.alpha.') else torch.zeros_like(expected[k])
    model.load_state_dict(merged,strict=True)
    return added


def signature(model):
    result=dict(model=asdict(model.c),H=model.H,lam=model.lam,gamma=model.gamma,
                aggregation=model.cross_loop_aggregation,aggregation_eps=model.cross_loop_eps,source_rmsnorm=model.source_rmsnorm)
    if getattr(model,'source_trajectory_offset',None) is not None:
        result['source_trajectory_offset']=model.source_trajectory_offset
    if getattr(model,'cross_loop_source','previous_token')=='same_token':
        result['cross_loop_source']='same_token'
    return result


def pack(model):return dict(signature=signature(model),model=model.state_dict())


def restore(model,payload):
    if payload['signature']!=signature(model):raise ValueError('Model/aggregation configuration mismatch')
    return model.load_state_dict(payload['model'],strict=True)


def from_payload(payload, *, device='cpu'):
    """Restore architecture from saved metadata, defaulting old baselines to RMSNorm."""
    from .model import Model, Config
    sig=payload['signature']; c=payload.get('execution_config') or payload.get('config') or sig
    source_rmsnorm=c.get('source_rmsnorm',sig.get('source_rmsnorm',True))
    if 'source_rmsnorm' in sig:assert sig['source_rmsnorm']==source_rmsnorm
    kwargs=dict(cross_loop_aggregation=c.get('cross_loop_aggregation',sig.get('aggregation','legacy')),
        cross_loop_eps=c.get('cross_loop_eps',sig.get('aggregation_eps')),
        source_trajectory_offset=c.get('source_trajectory_offset',sig.get('source_trajectory_offset')),
        cross_loop_source=c.get('cross_loop_source',sig.get('cross_loop_source','previous_token')),
        source_rmsnorm=source_rmsnorm)
    with torch.device('meta'):m=Model(Config(**sig['model']),sig['H'],sig.get('lambda',sig.get('lam',.5)),**kwargs)
    m.load_state_dict(payload['model'],strict=True,assign=True)
    return m.to(device)
