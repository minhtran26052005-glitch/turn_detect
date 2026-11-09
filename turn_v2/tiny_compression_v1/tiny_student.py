"""Small causal GRU classifier; right padding cannot affect valid timesteps."""
from __future__ import annotations
import torch
from torch import nn


class TinyTurnGRU(nn.Module):
    def __init__(self,vocab_size,embedding_size=64,hidden_size=64,layers=1,teacher_size=768,**kwargs):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_size,padding_idx=0)
        self.input_norm=nn.LayerNorm(embedding_size)
        self.gru=nn.GRU(embedding_size,hidden_size,num_layers=layers,batch_first=True,dropout=.1 if layers>1 else 0.)
        self.dropout=nn.Dropout(.1)
        feature_size=hidden_size*3+embedding_size+2
        self.head=nn.Sequential(nn.Linear(feature_size,hidden_size),nn.GELU(),nn.Dropout(.1),nn.Linear(hidden_size,2))
        self.teacher_projection=nn.Linear(feature_size,teacher_size)

    def forward(self,input_ids,attention_mask,return_features=False):
        embedded=self.input_norm(self.embedding(input_ids))
        outputs,_=self.gru(embedded)
        length=attention_mask.sum(dim=1).clamp(min=1)
        last=outputs.gather(1,(length-1)[:,None,None].expand(-1,1,outputs.shape[-1])).squeeze(1)
        mask=attention_mask.unsqueeze(-1).to(outputs.dtype)
        mean=(outputs*mask).sum(1)/length[:,None].to(outputs.dtype)
        maximum=outputs.masked_fill(attention_mask.unsqueeze(-1)==0,-1e4).max(1).values
        final_piece=embedded.gather(1,(length-2).clamp(min=0)[:,None,None].expand(-1,1,embedded.shape[-1])).squeeze(1)
        count=length.to(outputs.dtype)
        features=torch.cat([last,mean,maximum,final_piece,(count/64)[:,None],(torch.log1p(count)/5)[:,None]],dim=1)
        logits=self.head(self.dropout(features))
        return (logits,self.teacher_projection(features)) if return_features else logits


def load_student(path,device='cpu'):
    import json
    from pathlib import Path
    path=Path(path)
    config=json.loads((path/'config.json').read_text())
    model=TinyTurnGRU(**config)
    model.load_state_dict(torch.load(path/'student.pt',map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval()
