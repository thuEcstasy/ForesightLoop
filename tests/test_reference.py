import torch
import pytest
from foresightloop import ForesightLoop,Config
from foresightloop.oracle import run

@pytest.mark.parametrize('loops',[4,8])
def test_dense_incremental_and_independent_oracle(loops):
    torch.manual_seed(17)
    c=Config(vocab=37,dim=16,layers=2,heads=2,kv_heads=1,head_dim=8,intermediate=24,window=3)
    m=ForesightLoop(c,loops).eval()
    with torch.no_grad():
        for q,b in zip(m.cross_loop.query,m.cross_loop.bias):
            q.normal_(0,.2);b.normal_(0,.3)
    tokens=torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
    docs=torch.tensor([[0,0,1,1,0],[0,0,0,0,0]])
    valid=torch.tensor([[1,1,1,1,1],[0,1,1,1,0]],dtype=torch.bool)
    dense=m.vectorized(tokens,docs,valid=valid)
    oracle=run(m,tokens,docs,valid)['logits']
    cached=m.prefill(tokens,docs,valid=valid).logits
    torch.testing.assert_close(dense,oracle,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(cached,oracle,atol=2e-6,rtol=2e-5)
    short=m.vectorized(tokens[:,:3],docs[:,:3],valid=valid[:,:3])
    torch.testing.assert_close(dense[:,:3],short,atol=2e-6,rtol=2e-5)


def test_gradient_and_export(tmp_path):
    torch.manual_seed(23)
    c=Config(vocab=19,dim=8,layers=1,heads=1,kv_heads=1,head_dim=8,intermediate=12,window=2)
    m=ForesightLoop(c,4).eval()
    tokens=torch.tensor([[1,2,3]])
    docs=torch.zeros_like(tokens)
    y=m.vectorized(tokens,docs);(y.square().sum()).backward()
    grad={k:p.grad.clone() for k,p in m.named_parameters()}
    m.zero_grad(set_to_none=True)
    (run(m,tokens,docs)['logits'].square().sum()).backward()
    for k,p in m.named_parameters():torch.testing.assert_close(p.grad,grad[k],atol=2e-5,rtol=2e-4)
    m.save_pretrained(tmp_path)
    restored=ForesightLoop.from_pretrained(tmp_path)
    for k,p in m.state_dict().items():assert torch.equal(p,restored.state_dict()[k])
    generated=restored.generate(tokens,max_new_tokens=3)
    expected=tokens.clone()
    for _ in range(3):
        y=restored.vectorized(expected,torch.zeros_like(expected))
        expected=torch.cat([expected,y[:,-1].argmax(-1)[:,None]],1)
    assert torch.equal(generated,expected)
