import torch
from turn_v2.reward_v10_objective import rewards_for_actions,actor_loss
from turn_v2.reward_v10_data import add_unique


def test_reward_penalizes_premature_stop_more_than_delayed_stop():
    assert rewards_for_actions(torch.tensor([0,1])).tolist()==[[1.,-2.],[-.5,1.]]


def test_actor_gradient_moves_to_rewarded_action():
    logits=torch.zeros(2,2,requires_grad=True);labels=torch.tensor([0,1]);ref=torch.full((2,2),.5)
    loss,_=actor_loss(logits,labels,ref,actions=torch.tensor([1,0]));loss.backward()
    assert logits.grad[0,1]>0 and logits.grad[0,0]<0
    assert logits.grad[1,0]>0 and logits.grad[1,1]<0


def test_protected_visible_inputs_are_never_added_to_train():
    from turn_v2.common import input_id
    a={'text':'known words','context':'past','label':'EOT'};b={**a,'label':'NEOT'};c={'text':'new words','label':'EOT'}
    values,removed=add_unique([a,b,c],{input_id(a)})
    assert values==[c] and removed==2
