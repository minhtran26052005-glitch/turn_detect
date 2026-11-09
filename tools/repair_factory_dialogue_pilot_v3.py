"""Conservatively split generated sentence fields for manual pilot review."""
import collections
import hashlib
import html
import json
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.factory_dialogue_pilot_v2 import make_examples, validate_dialogue

ROOT=Path(__file__).resolve().parents[1]/'outputs/factory_dialogue_pilot_20261030'
SRC=ROOT/'qwen_v2';DEST=ROOT/'qwen_v3_review'
ABBREVIATION=re.compile(r'(?i)(?:\b(?:tp|ts|th\.s|pgs|gs|mr|ms|dr|stt|v\.v)|\b[A-Z])\.$')


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def split_explicit(text):
    # Only split punctuation followed by whitespace; avoid decimal points, and
    # reject likely abbreviated names rather than assert a sentence boundary.
    pieces=[];start=0
    for match in re.finditer(r'[.!?…]+(?=\s+\S)',text):
        prefix=text[start:match.end()].strip()
        if ABBREVIATION.search(prefix): raise ValueError('ambiguous abbreviation')
        pieces.append(prefix);start=match.end()
    tail=text[start:].strip()
    if tail:pieces.append(tail)
    return pieces


def main():
    DEST.mkdir(parents=True,exist_ok=False)
    generated=read_jsonl(SRC/'dialogues.jsonl');rejected=read_jsonl(SRC/'rejected.jsonl')
    accepted=[];failed=[]
    for x in generated:
        x['boundary_provenance']='qwen_explicit_sentence_list'
        accepted.append(x)
    for item in rejected:
        try:
            obj=json.loads(item['raw']);splits=0
            for t in obj['turns']:
                parts=[]
                for s in t['sentences']:
                    segments=split_explicit(s);splits+=max(0,len(segments)-1);parts.extend(segments)
                t['sentences']=parts
            d=validate_dialogue(obj,item['scenario_id'])
            d['boundary_provenance']='qwen_punctuation_conservative_split';d['split_count']=splits
            accepted.append(d)
        except Exception as e:
            failed.append({'scenario_id':item['scenario_id'],'reason':repr(e),'raw':item['raw']})
    examples=[]
    for d in accepted:
        for row in make_examples(d):
            row['label_source']='QWEN_SENTENCE_LIST_OR_GENERATED_PUNCTUATION_NOT_HUMAN_GOLD'
            row['boundary_provenance']=d['boundary_provenance']
            row['review_state']='UNREVIEWED'
            examples.append(row)
    for filename,rows in [('dialogues.jsonl',accepted),('examples.jsonl',examples),('rejected.jsonl',failed)]:
        with (DEST/filename).open('w',encoding='utf-8') as out:
            for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts=collections.Counter((x['label'],x['lookahead_words']) for x in examples)
    manifest={'source':'QWEN_SYNTHETIC','source_run':'qwen_v2','human_reviewed':False,'training_eligible':False,
        'accepted_dialogues':len(accepted),'rejected_dialogues':len(failed),'examples':len(examples),
        'boundary_provenance':dict(collections.Counter(x['boundary_provenance'] for x in accepted)),
        'counts':{f'{a}_k{b}':n for (a,b),n in counts.items()},
        'note':'Structured format checks only; sentence boundaries are model-provided/punctuation-derived; naturalness, speaker consistency, safety and labels require review.'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    blocks=[]
    for d in accepted:
        turns=''.join('<p><strong>'+html.escape(t['speaker'])+':</strong> '+html.escape(' '.join(s['display'] for s in t['sentences']))+'</p>' for t in d['turns'])
        candidate=[x for x in examples if x['dialogue_id']==d['dialogue_id'] and x['lookahead_words']==2]
        chosen=next((x for x in candidate if x['label']=='EOT'),None)
        row='' if not chosen else '<p>Ví dụ EOT: <code>'+html.escape(chosen['context'])+' | '+html.escape(chosen['text'])+'</code></p>'
        blocks.append('<article><h2>'+html.escape(d['dialogue_id'])+'</h2>'+turns+row+'<small>'+html.escape(d['boundary_provenance'])+' · chưa duyệt</small></article>')
    page='<!doctype html><meta charset="utf-8"><title>Pilot hội thoại nhà máy</title><style>body{font:16px system-ui;max-width:960px;margin:3rem auto;padding:0 1rem;line-height:1.6;background:#f8f9fc;color:#182537}article{background:white;border:1px solid #dbe1e8;border-radius:12px;padding:1rem 1.3rem;margin:1rem 0}code{background:#eef2f7;padding:.2rem}</style><h1>Hội thoại nhà máy do Qwen sinh</h1><p>Chưa duyệt: '+str(len(accepted))+' hội thoại, '+str(len(examples))+' mẫu điểm xét. Các câu gộp được tách thận trọng theo dấu câu Qwen sinh. Đây không phải human gold và chưa đưa vào train.</p>'+''.join(blocks)
    (DEST/'review.html').write_text(page,encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':main()
