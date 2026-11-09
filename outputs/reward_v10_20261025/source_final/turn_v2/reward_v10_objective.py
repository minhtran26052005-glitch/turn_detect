"""Offline one-step REINFORCE; automatic labels define imperfect rewards."""
import torch
import torch.nn.functional as F


def rewards_for_actions(labels):
    # Actions0=wait,1=stop. No true latency measurement is available: the
    # -0.5 late cost is a fixed preference, not milliseconds or a simulator.
    wait=torch.where(labels==0,torch.ones_like(labels,dtype=torch.float32),torch.full_like(labels,-.5,dtype=torch.float32))
    stop=torch.where(labels==1,torch.ones_like(labels,dtype=torch.float32),torch.full_like(labels,-2.,dtype=torch.float32))
    return torch.stack([wait,stop],dim=-1)


def actor_loss(logits,labels,reference_probs,actions=None):
    probs=logits.softmax(-1);table=rewards_for_actions(labels)
    if actions is None:actions=torch.distributions.Categorical(probs=probs).sample()
    actual=table.gather(1,actions[:,None]).squeeze(1)
    # All-action rewards are inferable from a binary training label; the
    # detached exact expectation is a variance-reduction baseline only.
    baseline=(probs.detach()*table).sum(-1)
    policy=-(F.log_softmax(logits,-1).gather(1,actions[:,None]).squeeze(1)*(actual-baseline)).mean()
    reference=reference_probs.clamp_min(1e-6);kl=(probs*(probs.clamp_min(1e-6).log()-reference.log())).sum(-1).mean()
    ce=F.cross_entropy(logits,labels,label_smoothing=.02)
    return .5*ce+.5*policy+.05*kl,{'sampled_reward':actual.mean().detach(),'expected_reward':(probs*table).sum(-1).mean().detach(),'kl':kl.detach()}
