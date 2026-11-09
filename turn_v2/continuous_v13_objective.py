"""A frozen inference-mode teacher target must be cloned before KL backward."""
import torch
import torch.nn.functional as F


def supervised_with_anchor(logits,labels,reference,cost_sensitive=False):
    target=reference.detach().clone()
    losses=F.cross_entropy(logits,labels,reduction='none',label_smoothing=.02)
    if cost_sensitive:losses=losses*torch.where(labels==0,2.,1.)
    return losses.mean()+.05*F.kl_div(F.log_softmax(logits,-1),target,reduction='batchmean')
