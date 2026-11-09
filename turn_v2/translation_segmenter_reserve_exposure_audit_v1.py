"""Conservatively remove reserve pages matching ANY inferred diagnostic input.

Legacy scorer infers removed rows for raw diagnostic metrics as well. None of
those probabilities are allowed to turn a matching reserved input into future
blind gold. This audit uses source IDs/input hashes only, never model scores to
choose thresholds/checkpoints and never reserve labels to tune a model.
"""
from collections import Counter
from pathlib import Path
import gzip
import hashlib
import json
import sqlite3
import time
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_v15d_test_v2 as ev

JOB=diag.JOB
CONTROL=JOB/'reserve_audit_v1'


def state(stage,**fields):
    diag.disk_guard();ev.atomic_json(CONTROL/'status.json',{'at_utc':diag.now(),'stage':stage,**fields})


def run():
    import fcntl
    from transformers import AutoTokenizer
    from .monolingual_data import encode_tail
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'reserve_exposure_audit_v1.json').exists():raise RuntimeError('Reserve audit already complete')
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        report={'created_at_utc':diag.now(),'complete':False,'languages':{},
            'policy':'Drop an entire reserved source page if any actual256-token input appears in any inferred diagnostic trace, including rows removed from clean metrics.',
            'reserve_model_inference_performed':False,'reserve_labels_used_for_tuning':False,
            'new_future_train_dev_overlap_audit_still_required':True}
        remaining={}
        try:
            for lang in ev.LANGUAGES:
                while True:
                    partial=ev.read_json(JOB/'partial_report.json') if (JOB/'partial_report.json').exists() else {}
                    if lang in partial.get('languages',{}):break
                    if (JOB/'continuation_v2_error.json').exists():raise RuntimeError('Scoring failed; reserve state preserved')
                    state('WAITING_FOR_INFERRED_LANGUAGE_TRACE',language=lang);time.sleep(15)
                trace=JOB/f'test_predictions_{lang}.jsonl.gz'
                if ev.file_sha(trace)!=partial['languages'][lang]['full_trace_sha256']:raise RuntimeError('Trace SHA mismatch')
                tokenizer=AutoTokenizer.from_pretrained(JOB/'frozen'/lang,local_files_only=True)
                observed=set();raw_rows=0;filtered_rows=0
                state('HASHING_ALL_ACTUALLY_INFERRED_INPUTS_CPU',language=lang)
                with gzip.open(trace,'rt',encoding='utf-8') as stream:
                    for raw in stream:
                        row=json.loads(raw);observed.add(ev.token_hash(encode_tail(tokenizer,row['context'],row['text'],256)['input_ids']))
                        raw_rows+=1;filtered_rows+=not row['included_in_clean_test']
                reserved=set(ev.read_json(JOB/(lang+'_partition.json'))['unscored_reserve_source_ids'])
                collisions=set();counts=Counter();rows_scanned=0
                for (raw,) in db.execute("SELECT record_json FROM examples WHERE language=? AND split='test'",(lang,)):
                    row=json.loads(raw);sid=row['source_id']
                    if sid not in reserved:continue
                    counts[sid]+=1;rows_scanned+=1
                    if ev.token_hash(encode_tail(tokenizer,row['context'],row['text'],256)['input_ids']) in observed:collisions.add(sid)
                kept=sorted(reserved-collisions)
                if not kept:raise RuntimeError('No untouched reserve pages survive actual-input exposure audit')
                remaining[lang]=kept
                report['languages'][lang]={'raw_inferred_rows':raw_rows,'inferred_rows_excluded_from_clean_metrics':filtered_rows,
                    'initial_reserve_pages':len(reserved),'initial_reserve_rows':rows_scanned,
                    'entire_reserve_pages_removed_for_actual_input_exposure':len(collisions),
                    'reserve_rows_removed_with_those_pages':sum(counts[sid] for sid in collisions),
                    'remaining_unscored_pages':len(kept),'remaining_unscored_rows':sum(counts[sid] for sid in kept),
                    'remaining_source_ids_sha256':hashlib.sha256('\n'.join(kept).encode()).hexdigest()}
                ev.atomic_json(CONTROL/'partial_report.json',report)
        finally:db.close()
        ev.atomic_json(JOB/'protected_remaining_sources_v1.json',{'source_ids_by_language':remaining,'labels_or_predictions_included':False})
        report.update(complete=True,completed_at_utc=diag.now(),audit_source_sha256=ev.file_sha(Path(__file__)))
        ev.atomic_json(JOB/'reserve_exposure_audit_v1.json',report)
        ev.atomic_json(JOB/'reserve_audit_source_receipt_v1.json',{'sha256':ev.file_sha(Path(__file__)),'source_utf8':Path(__file__).read_text(encoding='utf-8')})
        state('COMPLETE',remaining_pages={l:len(v) for l,v in remaining.items()})


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True);ev.atomic_json(CONTROL/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()});state('FAILED_PRESERVED',error=repr(exc));raise
