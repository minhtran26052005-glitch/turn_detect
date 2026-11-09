"""Train-fit control and coverage for the frozen c75 diagnostic; Colab only."""
from collections import Counter
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_v15d_test_v2 as ev

JOB = diag.JOB
CONTROL = JOB / 'analysis_v1'


def state(stage, **fields):
    diag.disk_guard()
    ev.atomic_json(CONTROL / 'status.json', {'stage': stage, 'at_utc': diag.now(), **fields})


def coverage(db, manifest):
    eligible = {lang + ':' + str(page) for lang, page in db.execute('SELECT language,page_id FROM sources WHERE added_cycle<=75')}
    output = {}
    for lang in ev.LANGUAGES:
        state('COUNTING_C75_TRAIN_COVERAGE_CPU', language=lang)
        counts = Counter()
        for sid, n in db.execute("SELECT source_id,COUNT(*) FROM examples WHERE language=? AND split='train' GROUP BY source_id", (lang,)):
            if sid in eligible:
                counts['rows'] += n
                counts['source_pages'] += 1
        steps = manifest['checkpoints'][lang]['total_steps']
        draws = steps * 32 * 2
        output[lang] = {**counts, 'total_optimizer_steps': steps, 'microbatch':32,
            'gradient_accumulation':2, 'total_training_row_draws':draws,
            'upper_bound_unique_current_rows_seen_fraction':min(1,draws/counts['rows']),
            'interpretation':'Upper bound only: draws include repetitions and earlier smaller corpora; this is not an epoch completion count.'}
    return eligible, output


def train_control(db, lang, eligible, manifest):
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .translation_segmenter_v15c import encode
    from .metrics import classification_metrics
    rows = []
    for label in ('NEOT','EOT'):
        for horizon in (1,2):
            selected = []
            for (raw,) in db.execute("SELECT record_json FROM examples WHERE language=? AND split='train' AND label=? AND lookahead_words=? ORDER BY example_id LIMIT 5000", (lang,label,horizon)):
                row = json.loads(raw)
                if row['source_id'] in eligible:
                    selected.append(row)
                if len(selected)==500:
                    break
            if len(selected)!=500:
                raise RuntimeError('Insufficient frozen train control rows: '+lang)
            rows.extend(selected)
    state('ENCODING_FIXED_TRAIN_FIT_CONTROL_CPU', language=lang, rows=len(rows))
    folder = JOB / 'frozen' / lang
    expected = manifest['checkpoints'][lang]['weights_sha256']
    if ev.file_sha(folder/'model.safetensors') != expected:
        raise RuntimeError('Frozen weights changed before train control')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    encoded = encode(rows,tokenizer)
    ev.wait_for_gpu(language=lang)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).cuda()
    probabilities = ev.infer(model,encoded,lang)
    threshold = manifest['checkpoints'][lang]['threshold_from_dev']
    metrics = classification_metrics(encoded[2],probabilities,threshold)
    path = JOB / ('train_fit_control_'+lang+'.jsonl.gz')
    digest=hashlib.sha256()
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for row,prob in zip(rows,probabilities):
            digest.update(row['example_id'].encode());digest.update(b'\n')
            stream.write(json.dumps({**row,'p_eot':float(prob),'threshold_from_dev':threshold,'predicted':'EOT' if prob>=threshold else 'NEOT'},ensure_ascii=False)+'\n')
    result={'metrics':metrics,'control_rows':len(rows),'sampling':'500 SHA-ordered train rows per label/lookahead stratum, source added<=75',
        'rows_id_sha256':digest.hexdigest(),'weights_sha256':expected,'threshold_from_dev':threshold,
        'by_lookahead_words':{str(k):classification_metrics(encoded[2][encoded[3]==k],probabilities[encoded[3]==k],threshold) for k in (1,2)},
        'scope':'TRAIN_POOL_FIT_CONTROL; not proof these sampled rows received an optimizer update',
        'full_trace_sha256':ev.file_sha(path)}
    del model,tokenizer,encoded,rows,probabilities
    gc.collect();torch.cuda.empty_cache()
    return result


def run():
    import fcntl
    CONTROL.mkdir(exist_ok=True)
    ev.JOB=CONTROL
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'error_analysis_v1.json').exists():raise RuntimeError('Analysis already complete; no rescore')
        while not (JOB/'manifest.json').exists():
            if (JOB/'error.json').exists():raise RuntimeError('Main diagnostic failed; preserve state')
            state('WAITING_FOR_FROZEN_INPUT_MANIFEST');time.sleep(10)
        manifest=ev.read_json(JOB/'manifest.json')
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        try:
            eligible,counts=coverage(db,manifest)
            ev.atomic_json(JOB/'training_coverage_v1.json',counts)
            while not (JOB/'report.json').exists() or not ev.read_json(JOB/'report.json').get('complete'):
                if (JOB/'error.json').exists():raise RuntimeError('Main diagnostic failed; preserve state')
                state('WAITING_FOR_DIAGNOSTIC_TEST_COMPLETION',coverage_ready=True);time.sleep(20)
            import torch
            import transformers.utils as utils
            import transformers.utils.import_utils as imports
            utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
            utils.is_librosa_available=imports.is_librosa_available=lambda:False
            torch.set_num_threads(1)
            report={'scope':'EXPLORATORY_ERROR_ANALYSIS_PLUS_FIXED_TRAIN_POOL_CONTROL',
                'created_at_utc':diag.now(),'diagnostic_report_sha256':ev.file_sha(JOB/'report.json'),
                'training_coverage':counts,'train_fit_controls':{},'quality_target_certified':False,
                'causal_limit':'Underexposure is measured; underfitting, label quality and distribution explanations require joint inspection of control metrics and source examples.'}
            for lang in ev.LANGUAGES:
                report['train_fit_controls'][lang]=train_control(db,lang,eligible,manifest)
                ev.atomic_json(CONTROL/'partial_report.json',report)
        finally:db.close()
        report['complete']=True;report['completed_at_utc']=diag.now()
        ev.atomic_json(JOB/'error_analysis_v1.json',report)
        ev.atomic_json(JOB/'analysis_source_receipt_v1.json',{'sha256':ev.file_sha(Path(__file__)),'source_utf8':Path(__file__).read_text(encoding='utf-8')})
        diag.drive_publish(complete=True)
        state('COMPLETE',report='error_analysis_v1.json',drive_readback_verified=True)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True)
        ev.atomic_json(CONTROL/'error.json',{'at_utc':diag.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('FAILED_PRESERVED',error=repr(exc))
        raise
