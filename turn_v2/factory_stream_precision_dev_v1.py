"""DEV-only precision/recall operating-point experiment; never changes training."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import traceback

from .factory_stream_data_v1 import sha
from .factory_stream_test_v1 import metric, kind
from .monolingual_data import encode_tail

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
CAMPAIGN = ROOT/'outputs/multilingual_stream_v2_20261102'
FROZEN = ROOT/'outputs/multilingual_stream_test_20261102_v1'
OUT = ROOT/'outputs/multilingual_stream_precision_dev_20261102_v1'


def save(name, value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage, **values):
    save('status.json',{'stage':stage,'pid':os.getpid(),'at_utc':datetime.now(timezone.utc).isoformat(),**values})


def make_folds():
    db=sqlite3.connect((CAMPAIGN/'dataset/paragraphs.sqlite').as_uri()+'?mode=ro',uri=True)
    candidates={side:{lang:[] for lang in ('vi','en')} for side in ('calibrate','verify')}
    for lang,identity,family,ordinal,record in db.execute("SELECT language,dialogue_id,family_id,ordinal,record FROM paragraphs WHERE split='dev'"):
        side='calibrate' if int(hashlib.sha256(('precision-dev-v1:'+family).encode()).hexdigest()[:8],16)%2==0 else 'verify'
        rank=hashlib.sha256(f'precision-dev-v1:{identity}:{ordinal}'.encode()).hexdigest()
        candidates[side][lang].append((rank,identity,family,json.loads(record)))
    db.close();folds={};families={};manifest={}
    for side,langs in candidates.items():
        chosen={};families[side]=set()
        for lang,rows in langs.items():
            chosen[lang]=[];seen=set()
            for _,identity,family,row in sorted(rows):
                if identity in seen:continue
                seen.add(identity);families[side].add(family);chosen[lang].append(row)
                if len(chosen[lang])==64:break
            assert len(chosen[lang])==64
        streams=[]
        for lang in ('vi','en'):
            for row in chosen[lang]:streams.append({**row,'mode':'monolingual','langs':[lang]*len(row['words'])})
        for i in range(32):
            for first,second in [('vi','en'),('en','vi')]:
                a,b=chosen[first][i],chosen[second][i];n=len(a['words'])
                streams.append({'words':a['words']+b['words'],'eots':a['eots']+[x+n for x in b['eots']],
                    'unknown':a['unknown']+[x+n for x in b['unknown']], 'mode':'mixed',
                    'langs':[first]*n+[second]*len(b['words'])})
        folds[side]=streams
        manifest[side]={'paragraphs_per_language':64,'streams':len(streams),'source_dialogues':{lang:[r['dialogue_id'] for r in rows] for lang,rows in chosen.items()},'family_ids':sorted(families[side])}
    assert not families['calibrate']&families['verify']
    save('dev_partition.json',{'folds':manifest,'source_split':'dev only','calibration_verification_family_overlap':0,
         'test_or_exposed_or_sealed_rows_read':False,'model_was_already_selected_using_other_dev_subset':True,
         'not_an_independent_test':True})
    return folds


def choose(rows):
    def cells(row):return [m for langs in row['by_mode'].values() for m in langs.values()]
    eligible=[r for r in rows if min(m['tp']+m['fp'] for m in cells(r))>=40 and min(m['eot_recall'] or 0 for m in cells(r))>=.70]
    feasible=[r for r in eligible if min(m['eot_precision'] or 0 for m in cells(r))>=.95]
    rank=lambda r:(min(m['eot_recall'] for m in cells(r)),min(m['eot_precision'] for m in cells(r)),-r['threshold'])
    if feasible:return max(feasible,key=rank),True
    return (max(eligible,key=lambda r:(min(m['eot_precision'] for m in cells(r)),min(m['eot_recall'] for m in cells(r)))) if eligible else None),False


def run():
    import numpy as np
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
    from safetensors.torch import load_file
    assert not (OUT/'report.json').exists(),'Do not rerun completed DEV experiment'
    os.nice(5);torch.set_num_threads(2)
    receipt=json.loads((FROZEN/'checkpoint_receipt.json').read_text());dev=json.loads((FROZEN/'frozen_dev.json').read_text())
    assert receipt['step']==dev['checkpoint_step'] and sha(FROZEN/'frozen/model.safetensors')==receipt['weights_sha256']
    baseline=dev['shared_threshold'];grid=sorted(set([baseline,.65,.75,.85,.90,.95,.97,.99,.995]))
    save('recipe.json',{'checkpoint':receipt,'baseline_threshold':baseline,'threshold_grid':grid,'target_precision':.95,
         'minimum_recall':.70,'minimum_predicted_cuts_per_language_mode':40,'lookahead_words':2,
         'objective':'Highest worst-language/mode recall subject to observed precision>=95%, recall>=70%, cuts>=40 on calibration DEV.',
         'read_test_samples_predictions_cases_or_scores':False,'one_common_threshold_no_language_selector':True,
         'model_weights_or_training_or_deployment_modified':False,'source_sha256':sha(__file__)})
    status('PREPARING_DEV_FOLDS');folds=make_folds()
    tok=AutoTokenizer.from_pretrained(FROZEN/'frozen',local_files_only=True,token=False)
    cfg=AutoConfig.from_pretrained(FROZEN/'frozen',local_files_only=True,token=False)
    model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32)
    model.load_state_dict(load_file(str(FROZEN/'frozen/model.safetensors')),strict=True);model.cuda().eval()
    cache={};inferences=0
    def predict(rows):
        nonlocal inferences
        missing=list(dict.fromkeys((r['context'],r['text']) for r in rows if (r['context'],r['text']) not in cache))
        for start in range(0,len(missing),64):
            part=missing[start:start+64];seqs=[encode_tail(tok,c,t,256)['input_ids'] for c,t in part]
            width=max(map(len,seqs));ids=torch.full((len(part),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.as_tensor(seq,device='cuda');mask[i,:len(seq)]=1
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):values=model(input_ids=ids,attention_mask=mask).logits.float().softmax(-1)[:,1].cpu().tolist()
            cache.update(zip(part,values));inferences+=len(part)
        return [cache[(r['context'],r['text'])] for r in rows]

    def rollout(side,thresholds):
        streams=folds[side];states=[{'t':t,'starts':[0]*len(streams),'cuts':[[] for _ in streams],
            'counts':{mode:{lang:Counter() for lang in ('vi','en')} for mode in ('monolingual','mixed')}} for t in thresholds]
        for b in range(1,max(len(s['words']) for s in streams)-1):
            rows=[];addresses=[]
            for state in states:
                for i,s in enumerate(streams):
                    if b+2>len(s['words']):continue
                    start=state['starts'][i]
                    rows.append({'context':' '.join(s['words'][start:b]),'text':' '.join(s['words'][b:b+2])});addresses.append((state,i))
            for (state,i),p in zip(addresses,predict(rows)):
                s=streams[i];pred=p>=state['t']
                if pred:state['starts'][i]=b;state['cuts'][i].append(b)
                if b not in s['unknown']:state['counts'][s['mode']][s['langs'][b-1]][kind(b in s['eots'],pred)]+=1
            if b%40==0:status('SCORING_DEV_POLICIES',fold=side,word_index=b,policy_count=len(states),unique_model_inputs=inferences)
        result=[]
        for state in states:
            delays=[];not_cut=0
            for i,s in enumerate(streams):
                for eot in s['eots']:
                    if eot+2>len(s['words']):continue
                    next_cut=next((b for b in state['cuts'][i] if b>=eot),None)
                    if next_cut is None:not_cut+=1
                    else:delays.append(next_cut-eot+2)
            result.append({'threshold':state['t'],'by_mode':{mode:{lang:metric(c) for lang,c in langs.items()} for mode,langs in state['counts'].items()},
                'words_after_reference_eot_until_model_cut':{'median':float(np.median(delays)) if delays else None,'p95':float(np.quantile(delays,.95)) if delays else None,
                'eots_without_later_model_cut_before_end':not_cut,'model_cut_may_be_wrong_boundary_counted_separately':True,'explicit_final_flush_not_counted_as_model_cut':True}})
        return result
    selection=rollout('calibrate',grid);save('calibration_results.json',selection)
    candidate,met=choose(selection)
    if candidate is None:
        save('report.json',{'complete':True,'checkpoint':receipt,'target_met_on_calibration':False,'candidate':None,'calibration':selection,'training_modified':False})
        status('COMPLETE',candidate_found=False);return
    save('locked_candidate.json',{'threshold':candidate['threshold'],'calibration_target_met':met,'locked_before_verification':True,'target':.95,'selection_only_used_calibration_fold':True})
    verification=rollout('verify',sorted(set([baseline,candidate['threshold']])))
    by_t={r['threshold']:r for r in verification};selected=by_t[candidate['threshold']]
    verified=all(m['eot_precision'] is not None and m['eot_precision']>=.95 and m['eot_recall']>=.70 and m['tp']+m['fp']>=40 for langs in selected['by_mode'].values() for m in langs.values())
    report={'complete':True,'checkpoint':receipt,'baseline_threshold':baseline,'candidate_threshold':candidate['threshold'],'target_precision':.95,
            'target_met_on_calibration':met,'target_met_on_verification':verified,'calibration':selection,'verification':verification,
            'candidate_locked_before_verification':True,'no_threshold_reselection_on_verification':True,
            'training_deployment_or_model_weights_modified':False,'test_samples_predictions_cases_or_scores_read':False,
            'unique_model_inputs':inferences,'max_future_words':2,'buffer_resets':'predicted_EOT_only',
            'scope':'Family-disjoint DEV calibration/verification on256synthetic VI/EN paragraphs; not independent test or human/audio gold. No ZH streaming assessment.',
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    save('report.json',report);status('COMPLETE',target_met_on_verification=verified,threshold=candidate['threshold'],report_sha256=sha(OUT/'report.json'))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'WORKER_LOCK').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:save('error.json',{'error':repr(e),'traceback':traceback.format_exc()});status('ERROR',error=repr(e));raise
