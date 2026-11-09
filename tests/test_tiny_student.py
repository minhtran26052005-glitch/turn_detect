import torch
from turn_v2.tiny_student import TinyTurnGRU


def test_right_padding_does_not_change_logits():
    torch.manual_seed(42)
    model=TinyTurnGRU(40,16,16).eval()
    x=torch.tensor([[1,2,8,9,2]]);mask=torch.ones_like(x)
    with torch.no_grad():
        a=model(x,mask)
        b=model(torch.nn.functional.pad(x,(0,11)),torch.nn.functional.pad(mask,(0,11)))
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


def test_batched_short_sequence_matches_single():
    torch.manual_seed(42)
    model=TinyTurnGRU(40,16,16,layers=2).eval()
    x=torch.tensor([[1,2,3,2,0,0],[1,2,5,6,7,2]]);mask=(x!=0).long()
    with torch.no_grad():
        batch=model(x,mask)
        single=model(x[:1,:4],mask[:1,:4])
    torch.testing.assert_close(batch[:1],single,rtol=1e-5,atol=1e-6)


def test_distillation_heads_receive_finite_gradients():
    torch.manual_seed(42)
    model=TinyTurnGRU(40,16,16)
    x=torch.tensor([[1,2,8,9,2],[1,2,7,2,0]]);mask=(x!=0).long()
    logits,features=model(x,mask,return_features=True)
    loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1,0]))+.05*features.square().mean()
    loss.backward()
    assert model.embedding.weight.grad is not None
    assert model.teacher_projection.weight.grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
