"""Portable single-language ONNX runner; NumPy/Tokenizers/ORT only."""
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
    return re.sub(r'\s+',' ',''.join(' ' if unicodedata.category(c).startswith('P') else c for c in value)).strip()


class TinyOnnxDetector:
    def __init__(self,path,threads=2):
        self.path=Path(path);self.contract=json.loads((self.path/'turn_detector_config.json').read_text(encoding='utf-8'))
        self.tokenizer=Tokenizer.from_file(str(self.path/'tokenizer.json'));settings=json.loads((self.path/'tokenizer_config.json').read_text(encoding='utf-8'))
        def special(key):
            value=settings[key];value=value['content'] if isinstance(value,dict) else value
            result=self.tokenizer.token_to_id(value)
            if result is None:raise ValueError('Missing special token: '+key)
            return result
        self.cls_id,self.sep_id,self.pad_id=[special(key) for key in ['cls_token','sep_token','pad_token']]
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        for key,value in self.contract.get('onnx_session_options',{}).items():options.add_session_config_entry(key,str(value))
        self.session=ort.InferenceSession(str(self.path/'model.onnx'),options,providers=['CPUExecutionProvider'])

    def encode(self,text,context=''):
        maximum=self.contract['max_length'];original=self.tokenizer.encode(normalize(text),add_special_tokens=False).ids
        current=original[-(maximum-3):] if self.contract.get('current_truncation')=='tail' else original[:maximum-3]
        budget=maximum-3-len(current);history=self.tokenizer.encode(normalize(context),add_special_tokens=False).ids[-budget:] if budget else []
        ids=[self.cls_id]+history+[self.sep_id]+current+[self.sep_id]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}

    def predict(self,text,context=''):
        encoded=self.encode(text,context);length=len(encoded['input_ids']);multiple=self.contract.get('padding_to_multiple_of',8)
        width=min(self.contract['max_length'],((length+multiple-1)//multiple)*multiple)
        arrays={key:np.asarray([value+[self.pad_id if key=='input_ids' else 0]*(width-length)],np.int64) for key,value in encoded.items()}
        logits=self.session.run(['logits'],arrays)[0].astype(np.float64);exp=np.exp(logits-logits.max(-1,keepdims=True));p=float((exp/exp.sum(-1,keepdims=True))[0,1])
        return {'prediction':'EOT' if p>=self.contract['threshold'] else 'NEOT','prob_EOT':p,'threshold':self.contract['threshold'],
            'language':self.contract.get('language'),'quality_status':self.contract.get('quality_status'),'smoke_only':self.contract.get('smoke_only',False)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--text',required=True);p.add_argument('--context',default='')
    a=p.parse_args();print(json.dumps(TinyOnnxDetector(a.model).predict(a.text,a.context),ensure_ascii=False,indent=2))
