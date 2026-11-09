"""Prepared future-data recipe: contiguous future words and VI abbreviations.

It is not installed into the running crawler. Existing scored examples and
their labels remain immutable; changing a recipe requires separate data/dev.
"""
import hashlib
import re
from . import translation_segmenter_causal_extraction_v1 as base
from .common import normalize
from .translation_segmenter_v15c import ABBREVIATIONS,LOOKAHEAD,TERMINATORS,make_example,words

RECIPE='CAUSAL_CONTIGUOUS_SENTENCE_EXTRACTION_V2_LANG_ABBREVIATIONS'
VI_ABBREVIATIONS={'tp','ts','ths','pgs','gs','bs','ks','th.s','pgs.ts','gs.ts'}


def all_sentence_spans(text,language='en'):
    abbreviations=ABBREVIATIONS | (VI_ABBREVIATIONS if language=='vi' else set())
    start=0
    for i,char in enumerate(text):
        if char not in TERMINATORS:continue
        before=text[i-1] if i else '';after=text[i+1] if i+1<len(text) else ''
        if char=='.':
            if before=='.' or after=='.' or (before.isdigit() and after.isdigit()):continue
            if after and not after.isspace() and after not in base.CLOSERS and not base.is_han(after):continue
            previous=re.search(r'([\w.]+)$',text[start:i],re.UNICODE)
            if previous and previous.group(1).strip('.').lower() in abbreviations:continue
        end=i+1
        while end<len(text) and text[end] in base.CLOSERS:end+=1
        following=text[end] if end<len(text) else ''
        if end==len(text) or following.isspace() or char in '。！？' or (char=='.' and base.is_han(after)):
            raw=text[start:end]
            if raw.strip():yield base.Span(raw.strip(),start,end,char)
            start=end
            while start<len(text) and text[start].isspace():start+=1
    if text[start:].strip():yield base.Span(text[start:].strip(),start,len(text),'')


def examples_from_page_causal(language,page_id,title,url,text,split,jieba):
    rows=[]
    for paragraph_index,paragraph in enumerate(re.split(r'\n+',text)):
        spans=list(all_sentence_spans(paragraph,language));tokens=[base.span_tokens(s,language,jieba) for s in spans]
        for index,(span,left) in enumerate(zip(spans,tokens)):
            if span.terminal and left:
                for k in LOOKAHEAD:
                    future=base.causal_future(tokens,index,k)
                    if len(future)==k:rows.append(make_example(language,page_id,paragraph_index,index,'sentence_boundary',k,len(left),left,future,split,title,url,target_mark=span.terminal))
            for match in re.finditer(r'[,，;；:：、]',span.text):
                before=words(span.text[:match.start()],language,jieba);after=words(span.text[match.end():],language,jieba)
                if not before:continue
                for k in LOOKAHEAD:
                    if len(after)>=k:rows.append(make_example(language,page_id,paragraph_index,index,'internal_punctuation',k,len(before),before,after[:k],split,title,url,target_mark=match.group()))
    for row in rows:
        row['example_id']=hashlib.sha256((RECIPE+'|'+row['example_id']).encode()).hexdigest();row['extraction_recipe']=RECIPE
        row['context']=normalize(row['context']);row['text']=normalize(row['text'])
    return rows
