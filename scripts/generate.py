"""Greedy continuation with the public reference model."""
import argparse
import torch
from transformers import AutoTokenizer
from foresightloop import ForesightLoop
p=argparse.ArgumentParser()
p.add_argument('--model',default='thuEcstasy1/ForesightLoop-8')
p.add_argument('--prompt',required=True)
p.add_argument('--max-new-tokens',type=int,default=32)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
a=p.parse_args()
t=AutoTokenizer.from_pretrained(a.model)
m=ForesightLoop.from_pretrained(a.model,device=a.device)
x=t(a.prompt,return_tensors='pt',add_special_tokens=False).input_ids.to(a.device)
y=m.generate(x,max_new_tokens=a.max_new_tokens,eos_token_id=t.eos_token_id)
print(t.decode(y[0,x.shape[1]:],skip_special_tokens=True))
