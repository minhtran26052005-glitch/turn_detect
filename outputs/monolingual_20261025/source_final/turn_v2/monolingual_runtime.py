"""Explicit language routing to three independent model files; no fallback."""
import argparse
import json
from pathlib import Path
from .common import file_hash
from .tiny_runtime import TinyOnnxDetector,normalize


class LanguageDetector(TinyOnnxDetector):
    def encode(self,text,context=''):
        if self.contract.get('current_truncation')!='tail':return super().encode(text,context)
        maximum=self.contract['max_length'];current=self.tokenizer.encode(normalize(text),add_special_tokens=False).ids[-(maximum-3):]
        remaining=maximum-3-len(current);history=self.tokenizer.encode(normalize(context),add_special_tokens=False).ids[-remaining:] if remaining else []
        ids=[self.cls_id]+history+[self.sep_id]+current+[self.sep_id]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}


class SeparateLanguageModels:
    def __init__(self,registry,factory=LanguageDetector):
        registry=Path(registry);record=json.loads(registry.read_text(encoding='utf-8'));self.root=(registry.parent/record['project_root_relative']).resolve()
        self.entries=record['models'];self.loaded={};self.factory=factory
        if set(self.entries)!={'vi','en','zh'}:raise ValueError('Registry requires vi/en/zh entries')
        paths=[(self.root/r['path']).resolve() for r in self.entries.values()]
        if len(set(paths))!=3:raise ValueError('Each language must have its own model directory')

    def predict(self,text,context='',language=None):
        if language not in self.entries:raise ValueError('Pass an explicit vi/en/zh language')
        entry=self.entries[language];path=self.root/entry['path']
        if entry['status']=='PENDING_TRAINING':raise RuntimeError(language+' model is pending; no other-language fallback')
        if language not in self.loaded:
            if file_hash(path/'model.onnx')!=entry['model_sha256']:raise ValueError('Model hash changed: '+language)
            self.loaded[language]=self.factory(path)
        return {'language':language,'model_path':str(path),**self.loaded[language].predict(text,context)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,default=Path('outputs/monolingual_20261025/registry.json'))
    p.add_argument('--language',choices=['vi','en','zh'],required=True);p.add_argument('--text',required=True);p.add_argument('--context',default='');a=p.parse_args()
    print(json.dumps(SeparateLanguageModels(a.registry).predict(a.text,a.context,a.language),ensure_ascii=False,indent=2))
