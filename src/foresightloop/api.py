# SPDX-License-Identifier: Apache-2.0
"""Public checkpoint API for the validated ForesightLoop reference graph."""
from dataclasses import asdict
from pathlib import Path
import json
import torch
from .model import Model,Config


class ForesightLoop(Model):
    """Shared-backbone causal language model with previous-cycle aggregation.

    This is the readable PyTorch reference, not the FA4/Triton benchmark engine.
    Public weights retain their original FP32 precision. For pretrained models,
    use ``from_pretrained`` rather than constructing a fresh random model.
    """
    def __init__(self,backbone_config=None,num_loops=8,residual_lambda=0.5,source_rmsnorm=True):
        config=Config(**backbone_config) if isinstance(backbone_config,dict) else (backbone_config or Config())
        super().__init__(config,num_loops,residual_lambda,cross_loop_aggregation='attnres',
                         cross_loop_eps=config.eps,source_rmsnorm=source_rmsnorm)

    @classmethod
    def from_pretrained(cls,model_id,*,device='cpu',torch_dtype=None,revision=None,
                        local_files_only=False,token=None):
        """Load a local exported directory or a Hugging Face model repository."""
        from safetensors.torch import load_file
        root=Path(model_id)
        def get(name):
            if root.is_dir():
                path=root/name
                if not path.is_file():raise FileNotFoundError(path)
                return path
            from huggingface_hub import hf_hub_download
            return Path(hf_hub_download(str(model_id),name,revision=revision,
                        local_files_only=local_files_only,token=token))
        config=json.loads(get('config.json').read_text())
        if config.get('checkpoint_format')!='foresightloop-safetensors-v1':
            raise ValueError('Expected an exported ForesightLoop checkpoint, not a standard Qwen checkpoint')
        if config.get('cross_loop_source','previous_token')!='previous_token' or config.get('source_trajectory_offset') is not None:
            raise ValueError('This public API loads previous-cycle ForesightLoop checkpoints')
        with torch.device('meta'):
            model=cls(config['backbone_config'],config['num_loops'],config.get('residual_lambda',.5),config.get('source_rmsnorm',True))
        model.load_state_dict(load_file(str(get('model.safetensors')),device='cpu'),strict=True,assign=True)
        model=model.to(device=device,dtype=torch_dtype) if torch_dtype is not None else model.to(device=device)
        model.release_config=config
        return model.eval()

    def save_pretrained(self,directory):
        """Save parameters only. Optimizer, training data, and caches are excluded."""
        from safetensors.torch import save_file
        out=Path(directory);out.mkdir(parents=True,exist_ok=True)
        config=dict(getattr(self,'release_config',{}))
        config.update(model_type='foresightloop',architectures=['ForesightLoop'],
            checkpoint_format='foresightloop-safetensors-v1',backbone_config=asdict(self.c),
            num_loops=self.H,residual_lambda=self.lam,cross_loop_aggregation='attnres',
            source_rmsnorm=self.source_rmsnorm,cross_loop_source='previous_token',source_trajectory_offset=None)
        (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        temporary=out/'.model.safetensors.incomplete'
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(temporary),metadata={'format':'pt'})
        temporary.replace(out/'model.safetensors')

    @torch.inference_mode()
    def generate(self,input_ids,*,max_new_tokens=32,eos_token_id=None):
        """Greedy completion for equally sized, unpadded prompts; returns prompt + output.

        The pretrained models are base language models, not chat-tuned models.
        Use ``vectorized`` for token-parallel full-sequence logits, or the
        lower-level ``prefill`` / ``decode`` API for explicit cache management.
        """
        if input_ids.ndim!=2 or input_ids.shape[1]==0:raise ValueError('Expected nonempty [batch,length] input IDs')
        if max_new_tokens<0:raise ValueError('max_new_tokens must be nonnegative')
        if not max_new_tokens:return input_ids.clone()
        result=input_ids.clone()
        output=self.prefill(input_ids)
        cache=output.cache;logits=output.logits[:,-1]
        finished=torch.zeros(input_ids.shape[0],device=input_ids.device,dtype=torch.bool)
        for index in range(max_new_tokens):
            next_token=logits.argmax(-1)
            if eos_token_id is not None:next_token=next_token.masked_fill(finished,int(eos_token_id))
            result=torch.cat((result,next_token[:,None]),1)
            if eos_token_id is not None:
                finished|=next_token==int(eos_token_id)
                if bool(finished.all()):break
            if index+1<max_new_tokens:
                output=self.decode(next_token,cache=cache,valid=~finished)
                cache=output.cache;logits=output.logits[:,-1]
        return result
