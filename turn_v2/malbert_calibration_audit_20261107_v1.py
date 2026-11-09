"""Bounded DEV-only calibration audit of frozen paused v7, without training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import numpy as np

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
MAIN=ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
BASE=ROOT/'outputs/sentence_mix_end_only_test_20261107_v1'
OUT=ROOT/'outputs/malbert_calibration_audit_20261107_v1'
LOW=[.00001,.00003,.0001,.0003,.001,.003,.01,.02,.03,.05,.075,.1]
MODULES=['malbert_calibration_audit_20261107_v1.py','uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py',
         'uploaded_vi_en_sentence_mix_data_v7.py','uploaded_vi_en_sentence_mix_evaluation_v7.py',
         'malbert_turn_student_v1.py','tiny_compression_v1/common.py']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def save(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)

def status(stage,**kw):
    raw=Path('/proc/self/stat').read_text();v=dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),at_utc=datetime.now(timezone.utc).isoformat(),**kw)
    save('status.json',v);print(json.dumps(v),flush=True)

def guard():
    mount=next(v.split() for v in Path('/proc/mounts').read_text().splitlines() if v.split()[1]=='/mnt/hdd500')
    assert mount[0]=='/dev/sda1' and 'rw' in mount[3].split(',')
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
    assert (MAIN/'STOP_REQUESTED').exists() and json.loads((MAIN/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
    launch=json.loads((MAIN/'launcher.json').read_text());assert not Path('/proc',str(launch['pid'])).exists()
    assert launch['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def verify_trace(trace,result,streams,tok,predict):
    """Independent boundary population, reset, encoding, metric and parity check."""
    from .tiny_compression_v1.common import encode_pair
    from .malbert_eval_zip_v2 import kind,metric
    expected={};starts=defaultdict(int);offset=defaultdict(int);counts=defaultdict(Counter);domains=defaultdict(Counter)
    for i,s in enumerate(streams):
        expected[i]=[(b,p['language']) for p in s['segments'] if p.get('punctuation_present',True) for b in range(p['start']+1,p['end']-1)]
    digest=hashlib.sha256();n=0;cases={}
    with gzip.open(OUT/trace,'rt',encoding='utf-8') as f:
        for line in f:
            digest.update(line.rstrip('\n').encode());v=json.loads(line);i=v['stream'];s=streams[i]
            b,lang=expected[i][offset[i]];offset[i]+=1
            assert v['boundary']==b and v['language']==lang and v['context_start']==starts[i]
            label=int(b in s['_eots']);scored=b not in s['_unknown']
            assert v['label']==label and v['scored']==scored
            p=v['p_eot'];assert math.isfinite(p) and 0<=p<=1
            pred=p>=result['threshold'];assert pred==v['predicted_eot']
            inp=dict(context=' '.join(s['words'][starts[i]:b]),text=' '.join(s['words'][b:b+2]))
            seq=encode_pair(tok,inp['context'],inp['text'],256)['input_ids']
            assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==v['encoded_sha256']
            if scored:
                c=kind(bool(label),pred);counts[lang][c]+=1;domains[s['domain'],lang][c]+=1
                bucket=(lang,c,'short' if len(seq)<=32 else 'long')
                cases.setdefault(bucket,(inp,p,pred))
            if pred:starts[i]=b
            n+=1
    assert all(offset[i]==len(seq) for i,seq in expected.items())
    assert n==result['inferences'] and digest.hexdigest()==result['causal_input_trace_digest']
    assert {l:metric(c) for l,c in counts.items()}==result['by_language']
    assert [dict(domain=d,language=l,**metric(c)) for (d,l),c in sorted(domains.items())]==result['by_domain']
    delta=0.
    for inp,p,pred in cases.values():
        values,_=predict([inp]);q=values[0];delta=max(delta,abs(q-p))
        assert (q>=result['threshold'])==pred
    assert delta<1e-4
    return dict(complete=True,inferences=n,source_boundary_population_exact=True,causal_context_future_reset_encoding_labels_metrics_verified=True,
                all_probabilities_finite=True,trace_sha256=sha(OUT/trace),canonical_single_case_count=len(cases),max_probability_delta=delta)

def scorekey(r):
    return min((m['eot_f1'] or 0) for m in r['by_language'].values()),sum((m['eot_f1'] or 0) for m in r['by_language'].values())

def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False;u.is_librosa_available=im.is_librosa_available=lambda:False
    from transformers import AlbertConfig,AutoTokenizer
    from .malbert_turn_student_v1 import MalbertTurnStudent
    from .uploaded_vi_en_data_v5 import initialize
    from . import uploaded_vi_en_sentence_mix_evaluation_v7 as ev
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    guard();assert not (OUT/'report.json').exists(),'Completed audit must never be rerun'
    cp=json.loads((BASE/'checkpoint_receipt.json').read_text());baseline=json.loads((BASE/'frozen_dev.json').read_text())
    assert cp['step']==3731173 and cp['threshold']==baseline['threshold']==.1
    assert sha(BASE/'frozen/weights.pt')==cp['weights_sha256']
    assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
    assert sha(BASE/'frozen_dev.json')==cp['matching_frozen_dev_sha256']
    manifest=json.loads((MAIN/'dataset_manifest.json').read_text())
    devfiles={n:sha(MAIN/n) for n in ['dev_calibration.json.gz','dev_monitor.json.gz']}
    assert all(h==manifest['files'][n] for n,h in devfiles.items())
    oldcurves=baseline['calibration'];grid=sorted(set(LOW+[r['threshold'] for r in oldcurves]))
    protocol=dict(checkpoint_step=3731173,parameters=11941122,weights_sha256=cp['weights_sha256'],full_checkpoint_sha256=cp['full_checkpoint_sha256'],
        baseline_dev_sha256=sha(BASE/'frozen_dev.json'),new_calibration_thresholds=LOW,all_candidate_thresholds=grid,
        selection='Maximum worst-language EOT F1, then sum F1, then first threshold in ascending fixed grid; calibration only',
        monitoring='Exactly one selected new threshold on untouched fixed monitor; baseline monitor reused',
        old_above_point1_calibration_reused=True,point1_calibration_recomputed_as_anchor=True,
        per_threshold_actual_causal_reset_rollout=True,dev_files=devfiles,code_sha256={n:sha(ROOT/'turn_v2'/n) for n in MODULES},
        no_training=True,no_test_scoring=True,no_model_default_change=True,not_independent_human_audio_gold=True,
        scope='Single paused v7 checkpoint; not ASR-adapted model, not fresh pretrained comparison, no training attribution')
    if (OUT/'locked_protocol.json').exists():assert json.loads((OUT/'locked_protocol.json').read_text())==protocol
    else:save('locked_protocol.json',protocol)
    for src,name in [(BASE/'frozen/weights.pt','frozen/weights.pt'),(BASE/'frozen/student_resume_pin.pt','frozen/student_resume_pin.pt')]:
        dst=OUT/name;dst.parent.mkdir(exist_ok=True)
        if not dst.exists():os.link(src,dst)
        assert sha(dst)==sha(src)
    save('baseline_dev.json',baseline)
    weights=torch.load(OUT/'frozen/weights.pt',map_location='cpu',weights_only=True)
    model=MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    model.load_state_dict(weights,strict=True);assert model.parameter_count()==11941122;del weights
    model.cuda().eval();tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    calibration=initialize(json.load(gzip.open(MAIN/'dev_calibration.json.gz','rt',encoding='utf-8')))
    monitor=initialize(json.load(gzip.open(MAIN/'dev_monitor.json.gz','rt',encoding='utf-8')))
    assert len(calibration)==119 and len(monitor)==394
    assert all(s['split']=='dev' for s in calibration+monitor)
    assert {s['family'] for s in calibration}.isdisjoint({s['family'] for s in monitor})
    ev.OUT=OUT;predict=ev.predictor(model,tok);curves=[];proofs=[]
    for j,t in enumerate(LOW):
        guard();name='calibration_%02d'%j;trace=name+'.jsonl.gz'
        if (OUT/(name+'_verification.json')).exists():
            r=json.loads((OUT/(name+'.json')).read_text());proof=json.loads((OUT/(name+'_verification.json')).read_text());assert sha(OUT/trace)==proof['trace_sha256']
        else:
            status('CALIBRATING',threshold=t,index=j+1,total=len(LOW))
            r=ev.rollout(predict,calibration,t,trace);save(name+'.json',r)
            status('VERIFYING_CALIBRATION_TRACE',threshold=t,index=j+1,total=len(LOW))
            proof=verify_trace(trace,r,calibration,tok,predict);save(name+'_verification.json',proof)
        curves.append(r);proofs.append(proof)
    oldanchor=next(r for r in oldcurves if r['threshold']==.1)
    assert curves[-1]['by_language']==oldanchor['by_language'],'Frozen checkpoint .1 anchor must reproduce exactly'
    curves.extend(r for r in oldcurves if r['threshold']>.1);curves.sort(key=lambda r:r['threshold'])
    chosen=max(curves,key=scorekey);selected=chosen['threshold']
    save('selection_receipt.json',dict(complete=True,threshold=selected,calibration_score=scorekey(chosen),
        selected_before_monitoring=True,locked_protocol_sha256=sha(OUT/'locked_protocol.json'),calibration=curves))
    status('SCORING_SELECTED_DEV_MONITOR',threshold=selected)
    result=ev.rollout(predict,monitor,selected,'selected_monitor.jsonl.gz');save('selected_monitor.json',result)
    status('VERIFYING_SELECTED_DEV_MONITOR',threshold=selected)
    mp=verify_trace('selected_monitor.jsonl.gz',result,monitor,tok,predict);save('selected_monitor_verification.json',mp)
    if selected==.1:assert result['by_language']==baseline['by_language'] and result['by_domain']==baseline['by_domain']
    old={(r['domain'],r['language']):r for r in baseline['by_domain']}
    comparisons=[dict(domain=r['domain'],language=r['language'],before=old[r['domain'],r['language']],after=r) for r in result['by_domain']]
    guard();assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
    assert sha(OUT/'frozen/weights.pt')==cp['weights_sha256']
    assert all(sha(ROOT/'turn_v2'/n)==h for n,h in protocol['code_sha256'].items())
    report=dict(complete=True,checkpoint_step=3731173,parameters=11941122,old_threshold=.1,new_threshold=selected,
        baseline=baseline,selected=result,comparison=comparisons,calibration=[dict(threshold=r['threshold'],by_language=r['by_language']) for r in curves],
        selection=protocol['selection'],baseline_reused_not_rescored=True,point1_anchor_exact=True,unchanged_weights=True,
        original_v7_remains_paused=True,no_training=True,no_test_scored=True,no_default_replacement=True,
        assessment_scope='Operating point diagnostic only. Punctuation silver DEV, no human/audio latency or translation quality claim; no optimum outside finite grid claim.')
    save('report.json',report)
    save('completion_verification.json',dict(complete=True,report_sha256=sha(OUT/'report.json'),
        new_calibration_rollouts_audited=len(proofs),calibration_inferences_audited=sum(p['inferences'] for p in proofs),
        monitor_inferences_audited=mp['inferences'],canonical_cases=sum(p['canonical_single_case_count'] for p in proofs)+mp['canonical_single_case_count'],
        max_probability_delta=max([p['max_probability_delta'] for p in proofs]+[mp['max_probability_delta']]),
        actual_reset_each_threshold=True,all_saved_predictions_audited=True,old_above_point1_summaries_reused_not_new_trace_audit=True,
        selection_before_monitor_verified=True,point1_anchor_exact=True,parent_full_checkpoint_unchanged=True,weights_unchanged=True,
        weights_sha256=cp['weights_sha256'],no_training=True,no_test_scoring=True,no_default_replacement=True))
    save('task_state.json',dict(complete=True,reported=False,phase='COMPLETE',parent_paused=True,adopted=False,auto_resume_authorized=False))
    status('COMPLETE',threshold=selected,report_sha256=sha(OUT/'report.json'))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True);os.nice(10)
    with (OUT/'AUDIT_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as exc:
            save('error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at_utc=datetime.now(timezone.utc).isoformat()));status('ERROR',error=repr(exc));raise
