"""Portable ONNX runner: numpy + tokenizers + onnxruntime, no PyTorch dependency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def normalize(text):
    value=unicodedata.normalize('NFKC',text).lower()
    value=''.join(' ' if unicodedata.category(c).startswith('P') else c for c in value)
    return re.sub(r'\s+',' ',value).strip()


class TinyOnnxDetector:
    def __init__(self,path,threads=None):
        self.path=Path(path)
        self.contract=json.loads((self.path/'turn_detector_config.json').read_text(encoding='utf-8'))
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'))
        settings=json.loads((self.path/'tokenizer_config.json').read_text(encoding='utf-8'))
        def special(key):
            token=settings[key]
            if isinstance(token,dict):token=token['content']
            value=self.tokenizer.token_to_id(token)
            if value is None:raise ValueError('Tokenizer special token missing: '+key)
            return value
        self.cls_id,self.sep_id,self.pad_id=[special(k) for k in ['cls_token','sep_token','pad_token']]
        options=ort.SessionOptions();options.intra_op_num_threads=threads or self.contract.get('intra_op_threads',2);options.inter_op_num_threads=1
        for key,value in self.contract.get('onnx_session_options',{}).items():options.add_session_config_entry(key,str(value))
        self.session=ort.InferenceSession(str(self.path/'model.onnx'),options,providers=['CPUExecutionProvider'])

    def encode(self,text,context=''):
        maximum=self.contract['max_length']
        current=self.tokenizer.encode(normalize(text),add_special_tokens=False).ids[:maximum-3]
        history=self.tokenizer.encode(normalize(context),add_special_tokens=False).ids
        remaining=maximum-3-len(current)
        history=history[-remaining:] if remaining else []
        ids=[self.cls_id]+history+[self.sep_id]+current+[self.sep_id]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}

    def predict(self,text,context=''):
        encoded=self.encode(text,context);size=len(encoded['input_ids']);multiple=self.contract.get('padding_to_multiple_of',8)
        target=min(self.contract['max_length'],((size+multiple-1)//multiple)*multiple)
        arrays={key:np.array([value+[self.pad_id if key=='input_ids' else 0]*(target-size)],dtype=np.int64) for key,value in encoded.items()}
        logits=self.session.run(['logits'],arrays)[0].astype(np.float64)
        exp=np.exp(logits-logits.max(-1,keepdims=True));prob=float((exp/exp.sum(-1,keepdims=True))[0,1])
        return {'prediction':'EOT' if prob>=self.contract['threshold'] else 'NEOT','prob_EOT':prob,'threshold':self.contract['threshold'],'smoke_only':self.contract.get('smoke_only',False)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--text',required=True);parser.add_argument('--context',default='');parser.add_argument('--threads',type=int)
    args=parser.parse_args()
    print(json.dumps(TinyOnnxDetector(args.model,args.threads).predict(args.text,args.context),ensure_ascii=False,indent=2))
