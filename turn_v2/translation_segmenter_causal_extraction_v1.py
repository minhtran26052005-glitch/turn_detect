"""Versioned future-data extractor: never skip source words in lookahead.

This module does not alter the running crawler or its SQLite/checkpoints.
New examples have a separate recipe identity; old records are not overwritten.
"""
from dataclasses import dataclass
import hashlib
import re

from .common import normalize
from .translation_segmenter_v15c import ABBREVIATIONS, LOOKAHEAD, TERMINATORS, make_example, terminal_mark, words

RECIPE='CAUSAL_CONTIGUOUS_SENTENCE_EXTRACTION_V1'
CLOSERS='"”’»)]}'


def is_han(char):
    return bool(char) and (0x3400 <= ord(char) <= 0x9fff or 0xf900 <= ord(char) <= 0xfaff or 0x20000 <= ord(char) <= 0x2fa1f)


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    terminal: str


def all_sentence_spans(text):
    """Keep short sentences and trailing fragments as part of the source stream."""
    start=0
    for i,char in enumerate(text):
        if char not in TERMINATORS:continue
        before=text[i-1] if i else '';after=text[i+1] if i+1<len(text) else ''
        if char=='.':
            if before=='.' or after=='.' or (before.isdigit() and after.isdigit()):continue
            if after and not after.isspace() and after not in CLOSERS and not is_han(after):continue
            prior=re.search(r'([\w.]+)$',text[start:i],re.UNICODE)
            if prior and prior.group(1).strip('.').lower() in ABBREVIATIONS:continue
        end=i+1
        while end<len(text) and text[end] in CLOSERS:end+=1
        following=text[end] if end<len(text) else ''
        if end==len(text) or following.isspace() or char in '。！？' or (char=='.' and is_han(after)):
            raw=text[start:end]
            if raw.strip():yield Span(raw.strip(),start,end,char)
            start=end
            while start<len(text) and text[start].isspace():start+=1
    if text[start:].strip():yield Span(text[start:].strip(),start,len(text),'')


def span_tokens(span,language,jieba):
    body=re.sub(r'([.!?。！？])["”’»)\]}]*$','',span.text).strip() if span.terminal else span.text
    return words(body,language,jieba)


def causal_future(tokenized,index,k):
    result=[]
    for following in tokenized[index+1:]:
        result.extend(following)
        if len(result)>=k:return result[:k]
    return result


def examples_from_page_causal(language,page_id,title,url,text,split,jieba):
    rows=[]
    for paragraph_index,paragraph in enumerate(re.split(r'\n+',text)):
        spans=list(all_sentence_spans(paragraph));tokenized=[span_tokens(s,language,jieba) for s in spans]
        for index,(span,left) in enumerate(zip(spans,tokenized)):
            if span.terminal and left:
                for k in LOOKAHEAD:
                    future=causal_future(tokenized,index,k)
                    if len(future)==k:
                        rows.append(make_example(language,page_id,paragraph_index,index,'sentence_boundary',k,len(left),left,future,split,title,url,target_mark=span.terminal))
            for match in re.finditer(r'[,，;；:：、]',span.text):
                before=words(span.text[:match.start()],language,jieba);after=words(span.text[match.end():],language,jieba)
                if not before:continue
                for k in LOOKAHEAD:
                    if len(after)>=k:
                        rows.append(make_example(language,page_id,paragraph_index,index,'internal_punctuation',k,len(before),before,after[:k],split,title,url,target_mark=match.group()))
    for row in rows:
        row['example_id']=hashlib.sha256((RECIPE+'|'+row['example_id']).encode()).hexdigest()
        row['extraction_recipe']=RECIPE
        row['context']=normalize(row['context']);row['text']=normalize(row['text'])
    return rows


def audit_legacy_eot(row,paragraph,language,jieba):
    """Align old eligible sentences, then inspect actual adjacent source tokens."""
    if row['label']!='EOT':return {'applicable':False}
    spans=list(all_sentence_spans(paragraph));tokenized=[span_tokens(s,language,jieba) for s in spans]
    legacy=[i for i,s in enumerate(spans) if s.terminal and len(s.text)>=25 and len(tokenized[i])>=5]
    index=row.get('sentence_index',-1)
    if not 0<=index<len(legacy):return {'applicable':True,'aligned':False}
    original=legacy[index]
    if normalize(' '.join(tokenized[original]))!=normalize(row['context']):return {'applicable':True,'aligned':False}
    expected=normalize(' '.join(causal_future(tokenized,original,row['lookahead_words'])))
    actual=normalize(row['text'])
    return {'applicable':True,'aligned':True,'source_contiguous_lookahead':actual==expected,
        'expected_future':expected,'recorded_future':actual,
        'short_or_unterminated_source_content_skipped':actual!=expected}
