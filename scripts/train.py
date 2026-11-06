"""Portable training example for prepacked token rows; no cluster-specific runtime."""
import argparse,contextlib,json,os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from foresightloop import ForesightLoop,Config
p=argparse.ArgumentParser()
p.add_argument('--tokens',required=True)
p.add_argument('--loops',type=int,choices=[4,8],default=4)
p.add_argument('--sequence-length',type=int,default=1024)
p.add_argument('--global-batch-size',type=int,default=128)
p.add_argument('--micro-batch-size',type=int,default=1)
p.add_argument('--steps',type=int,default=45777)
p.add_argument('--output',required=True)
p.add_argument('--config',help='Optional JSON backbone configuration for smoke tests')
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
if world>1:dist.init_process_group('nccl' if a.device=='cuda' else 'gloo')
if a.device=='cuda':torch.cuda.set_device(local)
device=torch.device('cuda',local) if a.device=='cuda' else torch.device(a.device)
assert a.global_batch_size%(world*a.micro_batch_size)==0
accum=a.global_batch_size//(world*a.micro_batch_size)
torch.manual_seed(20261017)
config=Config(**json.loads(Path(a.config).read_text())) if a.config else Config()
m=ForesightLoop(config,a.loops).to(device);m.checkpoint_layers=True
train=DistributedDataParallel(m,device_ids=[local] if a.device=='cuda' else None) if world>1 else m
optimizer=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1)
raw=np.memmap(a.tokens,dtype='<u4',mode='r')
assert len(raw)%(a.sequence_length+1)==0
rows=raw.reshape(-1,a.sequence_length+1)
assert len(rows)>=a.global_batch_size
for step in range(a.steps):
 optimizer.zero_grad(set_to_none=True);loss_value=0.
 for micro in range(accum):
  begin=step*a.global_batch_size+rank*(a.global_batch_size//world)+micro*a.micro_batch_size
  ids=np.arange(begin,begin+a.micro_batch_size)%len(rows)
  x=torch.from_numpy(np.array(rows[ids],dtype=np.int64)).to(device)
  synchronize=not (world>1 and micro+1<accum)
  sync=contextlib.nullcontext() if synchronize else train.no_sync()
  amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
  with sync,amp:
   loss=train(x[:,:-1],x[:,1:])/(a.micro_batch_size*a.sequence_length*accum)
  assert torch.isfinite(loss).item(),'Nonfinite loss; update aborted'
  loss.backward();loss_value+=float(loss.detach())
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
 optimizer.step()
 if rank==0 and (step%10==0 or step+1==a.steps):print(json.dumps({'step':step+1,'loss_local':loss_value,'grad_norm':float(norm)}),flush=True)
if rank==0:
 m.save_pretrained(a.output)
 Path(a.output,'training-example-config.json').write_text(json.dumps(vars(a),indent=2))
if world>1:dist.barrier();dist.destroy_process_group()
