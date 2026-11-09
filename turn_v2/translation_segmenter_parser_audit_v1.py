"""CPU-only alignment audit of scored diagnostic EOT examples, never reserve."""
from collections import Counter
from functools import lru_cache
import gzip
import json
from pathlib import Path
import re
import sqlite3
import time
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_v15d_test_v2 as ev
from .translation_segmenter_causal_extraction_v1 import audit_legacy_eot

JOB=diag.JOB
CONTROL=JOB/'parser_audit_v1'


def state(stage,**fields):
    diag.disk_guard()
    ev.atomic_json(CONTROL/'status.json',{'at_utc':diag.now(),'stage':stage,**fields})


def audit_language(db,lang,trace,jieba):
    @lru_cache(maxsize=128)
    def paragraph(source_id,number):
        language,page=source_id.split(':',1)
        row=db.execute('SELECT text FROM sources WHERE language=? AND page_id=?',(language,page)).fetchone()
        if row is None:return None
        values=re.split(r'\n+',row[0])
        return values[number] if 0<=number<len(values) else None
    counts=Counter();by_mark={};examples=[]
    with gzip.open(trace,'rt',encoding='utf-8') as stream:
        for raw in stream:
            row=json.loads(raw)
            if not row.get('included_in_clean_test') or row['label']!='EOT':continue
            counts['clean_EOT_rows']+=1
            source=paragraph(row['source_id'],row.get('paragraph_index',-1))
            result=audit_legacy_eot(row,source,lang,jieba) if source is not None else {'aligned':False}
            if not result.get('aligned'):
                counts['could_not_align_to_legacy_extraction']+=1
                continue
            counts['aligned_EOT_rows']+=1
            bad=result['short_or_unterminated_source_content_skipped']
            group='non_contiguous_lookahead' if bad else 'contiguous_lookahead'
            counts[group+'_rows']+=1
            counts[group+'_model_errors']+=not row['correct']
            mark=row.get('target_punctuation','')
            c=by_mark.setdefault(mark,Counter());c['aligned_rows']+=1;c['non_contiguous_rows']+=bad
            if bad and len(examples)<24:
                examples.append({**row,**result,'original_source_paragraph_for_posthoc_label_audit':source[:2500]})
            if counts['clean_EOT_rows']%2000==0:
                state('ALIGNING_DIAGNOSTIC_SOURCE_PARAGRAPHS_CPU',language=lang,counts=dict(counts))
    for name in ('non_contiguous_lookahead','contiguous_lookahead'):
        n=counts[name+'_rows']
        if n:counts[name+'_error_rate']=counts[name+'_model_errors']/n
    return {'counts':dict(counts),'by_terminal_mark':{k:dict(v) for k,v in by_mark.items()},
        'examples':examples,
        'limitation':'Reconstruction only where legacy sentence index and normalized full context align exactly. Unknown cases excluded from the causal-input defect rate. Error-rate association does not prove the effect of fixing the extractor.'}


def run():
    import fcntl
    import jieba
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'parser_audit_v1.json').exists():raise RuntimeError('Parser audit already completed; preserve it')
        while not (JOB/'manifest.json').exists():
            if (JOB/'error.json').exists():raise RuntimeError('Diagnostic failed; audit state preserved')
            state('WAITING_FOR_FROZEN_MANIFEST');time.sleep(10)
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        report={'created_at_utc':diag.now(),'scope':'CPU_SOURCE_ALIGNMENT_OF_CLEAN_SCORED_DIAGNOSTIC_ONLY',
            'reserve_read_or_scored':False,'languages':{},'complete':False,
            'synthetic_reproduction':'Old extractor turns a long sentence. Yes. Another long sentence. into future=another rather than future=yes.',
            'audit_source_sha256':ev.file_sha(Path(__file__)),
            'corrected_extractor_source_sha256':ev.file_sha(Path('turn_v2/translation_segmenter_causal_extraction_v1.py'))}
        try:
            for lang in ev.LANGUAGES:
                while True:
                    path=JOB/'partial_report.json'
                    partial=ev.read_json(path) if path.exists() else {}
                    if lang in partial.get('languages',{}):break
                    if (JOB/'error.json').exists():raise RuntimeError('Diagnostic failed; audit state preserved')
                    state('WAITING_FOR_SCORED_DIAGNOSTIC_LANGUAGE',language=lang);time.sleep(15)
                trace=JOB/f'test_predictions_{lang}.jsonl.gz'
                if ev.file_sha(trace)!=partial['languages'][lang]['full_trace_sha256']:
                    raise RuntimeError('Scored trace changed before source audit: '+lang)
                report['languages'][lang]=audit_language(db,lang,trace,jieba)
                ev.atomic_json(CONTROL/'partial_report.json',report)
        finally:db.close()
        report['complete']=True;report['completed_at_utc']=diag.now()
        ev.atomic_json(JOB/'parser_audit_v1.json',report)
        ev.atomic_json(JOB/'causal_extractor_source_receipt_v1.json',{'source_utf8':Path('turn_v2/translation_segmenter_causal_extraction_v1.py').read_text(encoding='utf-8'),
            'audit_source_utf8':Path(__file__).read_text(encoding='utf-8')})
        state('COMPLETE',report='parser_audit_v1.json')


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True)
        ev.atomic_json(CONTROL/'error.json',{'at_utc':diag.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('FAILED_PRESERVED',error=repr(exc))
        raise
