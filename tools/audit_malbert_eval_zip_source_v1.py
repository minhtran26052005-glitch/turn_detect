"""Read-only source-order/coverage audit, independent of the inference loop."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

ROOT=Path('/home/edabk/hoangbpm/turn_detect/outputs/malbert_eval_zip_20261104_v1')


def words(text):
    text=unicodedata.normalize('NFKC',text).lower()
    return ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split()


def run():
    streams=json.load(gzip.open(ROOT/'prepared_streams.json.gz','rt',encoding='utf-8'))
    sources={}
    with zipfile.ZipFile(ROOT/'inputs/eval.zip') as archive:
        for name in archive.namelist():
            if name.endswith('/labels_checked.jsonl'):
                domain='/'.join(name.split('/')[1:-1])
                sources[domain]={i:json.loads(line) for i,line in enumerate(archive.open(name)) if line.strip()}
    seen=Counter();counts=Counter();label_points=Counter()
    for stream in streams:
        domain=stream['domain'];previous=(-1,-1);joined=[]
        for segment in stream['segments']:
            rows=[sources[domain][i] for i in segment['source_rows']]
            assert all(r['split']==stream['split'] and r['src_lang']==segment['language'] for r in rows)
            assert all(stream['dialogue_id']==r['dialogue_id'] or stream['dialogue_id']==r['dialogue_id']+':item:'+str(i) for i,r in zip(segment['source_rows'],rows))
            assert len({(r.get('turn_idx'),r.get('speaker'),r['src_lang']) for r in rows})==1
            if len(rows)>1:assert [r['chunk_in_turn'] for r in rows]==list(range(len(rows)))
            order=min((r.get('chunk_idx',0),i) for i,r in zip(segment['source_rows'],rows))
            assert order>=previous,(stream['key'],order,previous)
            previous=order
            text=' '.join(r['source_raw'].strip() for r in rows)
            assert segment['display']==text
            actual=words(text)
            assert actual==stream['words'][segment['start']:segment['end']]
            assert segment['start']==len(joined)
            joined.extend(actual)
            for i in segment['source_rows']:seen[(domain,i)]+=1
            counts['source_rows']+=len(rows);counts['keep_false_rows']+=sum(r.get('keep') is False for r in rows)
            counts['source_words']+=len(actual);counts['segments']+=1
            for k in (1,2):
                if not segment['punctuation_present']:continue
                for b in range(segment['start']+1,segment['end']-k+1):
                    label_points[(domain,stream['split'],segment['language'],k,'eligible')]+=1
                    if b in stream['unknown']:continue
                    label_points[(domain,stream['split'],segment['language'],k,'eot' if b in stream['eots'] else 'neot')]+=1
        assert joined==stream['words']
    for domain,rows in sources.items():
        for i,row in rows.items():
            eligible=row.get('src_lang') in ('vi','en') and isinstance(row.get('source_raw'),str) and bool(words(row['source_raw']))
            assert seen[(domain,i)]==int(eligible),(domain,i,seen[(domain,i)],eligible)
    result={'complete':True,'source_rows_included_once':counts['source_rows'],'keep_false_source_rows_included':counts['keep_false_rows'],
        'source_words':counts['source_words'],'segments':counts['segments'],'streams':len(streams),
        'all_nonempty_vi_en_rows_accounted_for':True,'raw_source_text_word_identity_verified':True,
        'source_chronological_order_and_chunk_continuity_verified':True,'cross_speaker_chunk_stitching':False,
        'expected_points':[dict(domain=key[0],split=key[1],language=key[2],lookahead_words=key[3],kind=key[4],count=value) for key,value in sorted(label_points.items())],
        'prepared_streams_sha256':hashlib.sha256((ROOT/'prepared_streams.json.gz').read_bytes()).hexdigest()}
    (ROOT/'input_source_order_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key!='expected_points'}))


if __name__=='__main__':run()
