"""Verify the bounded ASR trial's frozen DEV traces and resume artifacts."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .asr_only_adaptation_20261107_v1 import OUT,MAIN,BASE,SOURCE,ROOT,sha,save,make_model,parent_paused,equal
from .uploaded_vi_en_data_v5 import initialize
from .tiny_compression_v1.common import encode_pair
from .malbert_eval_zip_v2 import kind,metric


def verify_trace(name,result,streams,tok):
    starts=defaultdict(int);last={};counts=defaultdict(Counter);domains=defaultdict(Counter);seen=Counter();h=hashlib.sha256();selected={}
    with gzip.open(OUT/name,'rt',encoding='utf-8') as f:
        for line in f:
            h.update(line.rstrip('\n').encode());r=json.loads(line);i=r['stream'];s=streams[i];b=r['boundary']
            seg=next(p for p in s['segments'] if p['start']<b and b+2<=p['end'])
            assert r['context_start']==starts[i] and b>last.get(i,0)
            assert r['label']==int(b in s['_eots']) and r['scored']==(b not in s['_unknown']) and r['language']==seg['language']
            inputs=dict(context=' '.join(s['words'][starts[i]:b]),text=' '.join(s['words'][b:b+2]))
            assert len(inputs['text'].split())==2
            seq=encode_pair(tok,inputs['context'],inputs['text'],256)['input_ids']
            assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==r['encoded_sha256']
            assert math.isfinite(r['p_eot']) and 0<=r['p_eot']<=1 and r['predicted_eot']==(r['p_eot']>=result['threshold'])
            if r['scored']:
                case=kind(bool(r['label']),r['predicted_eot']);counts[r['language']][case]+=1;domains[s['domain'],r['language']][case]+=1
            bucket='short' if len(seq)<=32 else 'medium' if len(seq)<=128 else 'long'
            selected.setdefault((s['domain'],seg['language'],bucket),(r,inputs))
            if r['predicted_eot']:starts[i]=b
            last[i]=b;seen[i]+=1
    assert h.hexdigest()==result['causal_input_trace_digest'] and sum(seen.values())==result['inferences']
    assert {lang:metric(c) for lang,c in counts.items()}==result['by_language']
    assert [dict(domain=d,language=l,**metric(c)) for (d,l),c in sorted(domains.items())]==result['by_domain']
    for i,s in enumerate(streams):assert seen[i]==sum(max(0,p['end']-p['start']-2) for p in s['segments'] if p.get('punctuation_present',True))
    return [(r,inputs,result['threshold']) for r,inputs in selected.values()],sum(seen.values())


def main():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False;u.is_librosa_available=im.is_librosa_available=lambda:False
    from transformers import AutoTokenizer
    from .uploaded_vi_en_sentence_mix_evaluation_v7 import predictor
    assert not (OUT/'completion_verification.json').exists(),'Read an existing verified result instead of rerunning verification'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    report=json.loads((OUT/'report.json').read_text());assert report['complete'] and report['no_test_scored']
    manifest=json.loads((OUT/'dataset_manifest.json').read_text());assert sha(OUT/'train_streams.json.gz')==manifest['train_sha256']
    assert sha(SOURCE/'train_streams.json.gz')==manifest['source_train_sha256']
    for n,h in manifest['code_sha256'].items():assert sha(ROOT/'turn_v2'/n)==h
    source=json.load(gzip.open(SOURCE/'train_streams.json.gz','rt',encoding='utf-8'))
    source_asr=[s for s in source if s['domain']=='asr/run2b'];del source
    actual_asr=json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8'))
    assert actual_asr==source_asr
    recounted=Counter()
    for s in actual_asr:
        eots=set(s['eots']);unknown=set(s['unknown']);masks={k:set(v) for k,v in s['conflict_mask'].items()}
        for seg in s['segments']:
            for k in (1,2):
                for b in range(seg['start']+1,seg['end']-k+1):
                    if b not in unknown and b not in masks[str(k)]:recounted[seg['language']+':k'+str(k)+':'+str(int(b in eots))]+=1
    assert dict(recounted)=={k:v for k,v in manifest['counts'].items() if ':k' in k}
    coverage=json.loads((OUT/'epoch_0001_coverage.json').read_text());assert dict(recounted)==coverage['by_key']
    assert sum(recounted.values())==coverage['seen']==manifest['eligible_points_per_epoch']
    tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    streams=initialize(json.load(gzip.open(OUT/'dev_monitor.json.gz','rt',encoding='utf-8')))
    assert all(s['split']=='dev' for s in streams)
    for n,h in manifest['dev_files'].items():assert sha(OUT/n)==h==sha(MAIN/n)
    dev=report['dev'];chosen=dev['selected'];baseline=json.loads((OUT/'baseline_dev.json').read_text())
    assert sha(OUT/'baseline_dev.json')==sha(BASE/'frozen_dev.json') and baseline['threshold']==.1
    selected,n=verify_trace('selected_dev_trace.jsonl.gz',chosen,streams,tok)
    if dev['fixed_threshold_trace']!='selected_dev_trace.jsonl.gz':
        extra,n2=verify_trace(dev['fixed_threshold_trace'],dev['fixed_baseline_threshold'],streams,tok);selected+=extra;n+=n2
    else:assert dev['fixed_baseline_threshold']==chosen
    m=make_model();m.load_state_dict(torch.load(OUT/'runs/finetune/weights.pt',map_location='cpu',weights_only=True),strict=True);m.eval();predict=predictor(m,tok)
    delta=0.
    for a in range(0,len(selected),16):
        part=selected[a:a+16];probs,_=predict([x[1] for x in part])
        for (r,_,threshold),p in zip(part,probs):
            delta=max(delta,abs(p-r['p_eot']));assert (p>=threshold)==r['predicted_eot']
    assert delta<1e-4
    del m;gc.collect();torch.cuda.empty_cache()
    latest=torch.load(OUT/'runs/finetune/latest_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    initial=torch.load(OUT/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    parent=torch.load(OUT/'frozen/parent_full_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    for k in ['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng']:assert equal(initial[k],parent[k])
    assert latest['epoch']==1 and latest['scored_seen']==manifest['eligible_points_per_epoch']
    assert len(latest['optimizer']['state'])==27 and {int(v['step']) for v in latest['optimizer']['state'].values()}=={latest['step']}
    for state in latest['optimizer']['state'].values():
        for v in state.values():
            if torch.is_tensor(v):assert torch.isfinite(v).all()
    assert latest['step']==3731173+latest['branch_step'] and latest['cursor']==420 and latest['stream_state']['slots']==[]
    assert sorted(map(int,latest['order']))==list(range(420)) and all(k in latest for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng'])
    weights=torch.load(OUT/'runs/finetune/weights.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,weights[k]) and torch.isfinite(v).all() for k,v in latest['model'].items())
    assert any(not torch.equal(v,parent['model'][k]) for k,v in weights.items())
    cp=json.loads((BASE/'checkpoint_receipt.json').read_text());assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256'];parent_paused()
    expected=max(dev['calibration'],key=lambda r:(min((m['eot_f1'] or 0) for m in r['by_language'].values()),sum((m['eot_f1'] or 0) for m in r['by_language'].values())))
    assert expected['threshold']==chosen['threshold']
    out=dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(OUT/'report.json'),
        all_dev_inferences_audited=n,all_causal_inputs_fullstop_masks_predicted_resets_and_confusions_verified=True,
        inference_audit_scope='All saved final monitoring predictions; calibration curve argmax recomputed, raw calibration predictions not separately saved',
        calibration_threshold_selection_recomputed=True,
        canonical_predictor_cases=len(selected),max_probability_delta=delta,all_probabilities_finite=True,
        all_canonical_decisions_identical=True,
        initial_parent_model_optimizer_all_rng_exact=True,complete_epoch_points=latest['scored_seen'],
        training_source_subset_and_all_eligible_gap_counts_verified=True,
        full_resume_optimizer_rng_order_cursor_lanes_verified=True,parent_v7_still_paused_and_unchanged=True,
        no_test_evaluation=True,no_model_default_replacement=True,labels_not_human_audio_gold=True,
        latest_full_resume_sha256=sha(OUT/'runs/finetune/latest_resume.pt'),weights_sha256=sha(OUT/'runs/finetune/weights.pt'))
    save('completion_verification.json',out);print(json.dumps(out))


if __name__=='__main__':main()
