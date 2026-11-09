"""Replay cycle75 sampler and verify saved RNG, then score actual drawn rows.

This is a TRAIN-only fit control on Colab, not another test evaluation. Replay
must match the frozen training fingerprint AND checkpoint's final NumPy RNG.
"""
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_diagnostic_analysis_v1 as fit
from . import translation_segmenter_v15d_test_v2 as ev

JOB=diag.JOB
CONTROL=JOB/'actual_seen_control_v1'


def state(stage,**fields):
    diag.disk_guard();ev.atomic_json(CONTROL/'status.json',{'stage':stage,'at_utc':diag.now(),**fields})


def replay(db,lang,manifest,eligible):
    import numpy as np
    import torch
    state('REPLAYING_C75_OPTIMIZER_ROW_SAMPLER_CPU',language=lang)
    ids=[];pools={(label,k):[] for label in ('NEOT','EOT') for k in (1,2)};digest=hashlib.sha256()
    for example_id,label,k,sid,raw in db.execute("SELECT example_id,label,lookahead_words,source_id,record_json FROM examples WHERE language=? AND split='train' ORDER BY example_id",(lang,)):
        if sid not in eligible:continue
        pools[(label,k)].append(len(ids));ids.append(example_id);digest.update(raw.encode());digest.update(b'\n')
    record=manifest['checkpoints'][lang]
    if digest.hexdigest()!=record['training_fingerprint']:raise RuntimeError('Cannot reconstruct exact cycle75 train pool: '+lang)
    pools={key:np.asarray(values,dtype=np.int64) for key,values in pools.items()}
    rng=np.random.default_rng(20261026+75+ev.LANGUAGES.index(lang));drawn=[]
    for step in range(100):
        for accumulation in range(2):
            indices=np.concatenate([rng.choice(values,8,replace=True) for values in pools.values()]);rng.shuffle(indices);drawn.extend(indices.tolist())
    saved=torch.load(JOB/'frozen'/(lang+'_resume_pin.pt'),map_location='cpu',weights_only=False)
    expected=saved['numpy_rng'];del saved;gc.collect()
    if json.dumps(rng.bit_generator.state,sort_keys=True,default=int)!=json.dumps(expected,sort_keys=True,default=int):
        raise RuntimeError('Sampler replay RNG does not match frozen checkpoint: '+lang)
    chosen={};raws={};drawn_set=set(drawn)
    for key,pool in pools.items():
        candidates=sorted(ids[i] for i in pool if int(i) in drawn_set)
        if len(candidates)<500:raise RuntimeError('Too few actual unique drawn rows for balanced control')
        chosen[key]=candidates[:500]
    all_ids=[i for values in chosen.values() for i in values]
    for start in range(0,len(all_ids),900):
        batch=all_ids[start:start+900]
        for eid,raw in db.execute('SELECT example_id,record_json FROM examples WHERE example_id IN ('+','.join('?' for _ in batch)+')',batch):raws[eid]=raw
    selected={key:[raws[eid] for eid in values] for key,values in chosen.items()}
    receipt={'training_fingerprint_verified':True,'saved_numpy_rng_exactly_replayed':True,
        'optimizer_cycle':75,'optimizer_steps_replayed':100,'row_draws':len(drawn),
        'unique_rows_actually_drawn_in_cycle75':len(drawn_set),'control_rows':len(all_ids),
        'control_ids_sha256':hashlib.sha256('\n'.join(all_ids).encode()).hexdigest()}
    del ids,pools,drawn,drawn_set,raws;gc.collect()
    return selected,receipt


class ActualDrawnDB:
    def __init__(self,selected):self.selected=selected
    def execute(self,sql,params=()):
        lang,label,k=params
        return [(raw,) for raw in self.selected[(label,k)]]


def run():
    import fcntl
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'actual_seen_fit_control_v1.json').exists():raise RuntimeError('Actual-seen control already complete')
        manifest=ev.read_json(JOB/'manifest.json')
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        eligible={lang+':'+str(page) for lang,page in db.execute('SELECT language,page_id FROM sources WHERE added_cycle<=75')}
        selected={};receipts={}
        try:
            for lang in ev.LANGUAGES:selected[lang],receipts[lang]=replay(db,lang,manifest,eligible)
        finally:db.close()
        ev.atomic_json(CONTROL/'sampler_replay_receipts.json',receipts)
        (CONTROL/'frozen').mkdir(exist_ok=True)
        for lang in ev.LANGUAGES:
            os.symlink((JOB/'frozen'/lang).resolve(),CONTROL/'frozen'/lang,target_is_directory=True)
        import torch
        import transformers.utils as utils
        import transformers.utils.import_utils as imports
        utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
        utils.is_librosa_available=imports.is_librosa_available=lambda:False
        torch.set_num_threads(1);fit.JOB=CONTROL;fit.CONTROL=CONTROL;ev.JOB=CONTROL
        report={'scope':'ACTUAL_CYCLE75_OPTIMIZER_DRAWN_TRAIN_CONTROL_NOT_TEST',
            'created_at_utc':diag.now(),'languages':{},'complete':False,'sampler_replay':receipts}
        for lang in ev.LANGUAGES:
            result=fit.train_control(ActualDrawnDB(selected[lang]),lang,eligible,manifest)
            result['sampling']='500 SHA-ordered actual cycle75 optimizer-drawn unique rows per label/lookahead; exact train-pool fingerprint and final NumPy RNG verified'
            result['scope']='ACTUALLY_OPTIMIZED_TRAIN_FIT_CONTROL'
            report['languages'][lang]=result
            ev.atomic_json(CONTROL/'partial_report.json',report)
            os.link(CONTROL/('train_fit_control_'+lang+'.jsonl.gz'),JOB/('actual_seen_fit_control_'+lang+'.jsonl.gz'))
        report.update(complete=True,completed_at_utc=diag.now())
        ev.atomic_json(JOB/'actual_seen_fit_control_v1.json',report)
        ev.atomic_json(JOB/'actual_seen_control_source_receipt_v1.json',{'sha256':ev.file_sha(Path(__file__)),'source_utf8':Path(__file__).read_text(encoding='utf-8')})
        state('COMPLETE',report='actual_seen_fit_control_v1.json')


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True);ev.atomic_json(CONTROL/'error.json',{'error':repr(exc),'traceback':traceback.format_exc()});state('FAILED_NO_CONTROL_RESCORE',error=repr(exc));raise
