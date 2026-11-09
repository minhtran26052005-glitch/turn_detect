"""Exact bounded caches for the official left-tail / future encoder.

No approximate tokenizer or changed truncation: only reuse deterministic
normalization/tokenization for repeated left contexts and short future strings.
"""
from collections import OrderedDict
from .common import normalize


class BoundedNormalize:
    def __init__(self,entries=2048,max_chars=8192):
        self.cache=OrderedDict();self.entries=entries;self.max_chars=max_chars

    def __call__(self,text):
        if len(text)>self.max_chars:return normalize(text)
        if text in self.cache:
            self.cache.move_to_end(text);return self.cache[text]
        value=normalize(text);self.cache[text]=value
        if len(self.cache)>self.entries:self.cache.popitem(last=False)
        return value


class ExactTailEncoder:
    def __init__(self,tokenizer,entries=2048):
        self.tokenizer=tokenizer;self.contexts=OrderedDict();self.futures=OrderedDict();self.entries=entries
        self.normalizer=BoundedNormalize();self.calls=0;self.context_hits=0;self.future_hits=0

    def encode(self,context,text,max_length=256):
        if max_length<3:raise ValueError('max_length must be >=3')
        self.calls+=1;limit=max_length-3
        key=(text,max_length)
        if key in self.futures:
            self.future_hits+=1;self.futures.move_to_end(key);current=self.futures[key]
        else:
            current=self.tokenizer.encode(self.normalizer(text),add_special_tokens=False)[-limit:] if limit else []
            if len(text)<=256:
                self.futures[key]=current
                if len(self.futures)>self.entries:self.futures.popitem(last=False)
        budget=limit-len(current);history=[]
        if budget:
            key=(context,max_length)
            if key in self.contexts:
                self.context_hits+=1;self.contexts.move_to_end(key);encoded=self.contexts[key]
            else:
                encoded=self.tokenizer.encode(self.normalizer(context),add_special_tokens=False)[-limit:]
                if len(context)<=8192:
                    self.contexts[key]=encoded
                    if len(self.contexts)>self.entries:self.contexts.popitem(last=False)
            history=encoded[-budget:]
        ids=[self.tokenizer.cls_token_id]+history+[self.tokenizer.sep_token_id]+current+[self.tokenizer.sep_token_id]
        return {'input_ids':ids,'attention_mask':[1]*len(ids)}
