"""User-requested paused-v7 TEST and end-only emission accounting, without training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import bisect
import copy
import fcntl
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
MAIN=ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
JOB=ROOT/'outputs/sentence_mix_end_only_test_20261107_v1'
OLDTEST=ROOT/'outputs/malbert_uploaded_test_20261105_v1'
MTOLD=ROOT/'outputs/translation_turn_test_20261105_v1'
MTDATA=ROOT/'data/translation_qwen3_06b_15640_user_20261105'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def save(name,value):
    p=JOB/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage,**kw):
    raw=Path('/proc/self/stat').read_text()
    v=dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],
           boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
           at_utc=datetime.now(timezone.utc).isoformat(),training_paused=True,**kw)
    save('status.json',v);print(json.dumps(v),flush=True)


def reserve():
    m=next(x.split() for x in Path('/proc/mounts').read_text().splitlines() if x.split()[1]=='/mnt/hdd500')
    assert m[0]=='/dev/sda1' and 'rw' in m[3].split(',')
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30


def paused():
    l=json.loads((MAIN/'launcher.json').read_text())
    assert l['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert not Path('/proc',str(l['pid'])).exists()
    assert (MAIN/'STOP_REQUESTED').exists()
    assert json.loads((MAIN/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
    return l


def make_model(weights):
    import torch
    from transformers import AlbertConfig,AutoTokenizer
    from .malbert_turn_student_v1 import MalbertTurnStudent
    m=MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    m.load_state_dict(weights,strict=True);assert m.parameter_count()==11941122<20000000
    assert all(torch.isfinite(v).all() for v in m.state_dict().values())
    return m.cuda().eval(),AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)


def freeze_and_calibrate():
    import torch
    from . import uploaded_vi_en_sentence_mix_evaluation_v7 as ev
    from .uploaded_vi_en_curriculum_data_v6 import check_order
    l=paused();reserve();frozen=JOB/'frozen';frozen.mkdir()
    pin=frozen/'student_resume_pin.pt';os.link(MAIN/'runs/finetune/latest_resume.pt',pin)
    x=torch.load(pin,map_location='cpu',weights_only=False,mmap=True)
    assert x['parameters']==11941122 and len(x['optimizer']['state'])==27 and not x['quant_scales']
    assert {int(s['step']) for s in x['optimizer']['state'].values()}=={x['step']}
    assert all(k in x for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order','cursor','stream_state'])
    assert len(x['order'])==98520 and len(set(map(int,x['order'])))==98520
    manifest=json.loads((MAIN/'dataset_manifest.json').read_text());fp=x['fingerprint']
    assert sha(MAIN/'dataset_manifest.json')==fp['dataset_manifest_sha256']
    assert sha(MAIN/'train_streams.json.gz')==fp['train_streams_sha256']
    assert sha(MAIN/'curriculum_plan.json')==fp['curriculum_plan_sha256']
    for n,h in fp['code_sha256'].items():assert sha(ROOT/'turn_v2'/n)==h
    plan=json.loads((MAIN/'curriculum_plan.json').read_text());check_order(plan['stages'][3],x['order'])
    torch.save(x['model'],frozen/'weights.pt')
    actual=torch.load(frozen/'weights.pt',map_location='cpu',weights_only=True)
    assert actual.keys()==x['model'].keys() and all(torch.equal(actual[k],x['model'][k]) for k in actual)
    cp=dict(step=x['step'],branch_step=x['branch_step'],epoch_display=x['epoch']+1,parameters=x['parameters'],
            full_checkpoint_sha256=sha(pin),weights_sha256=sha(frozen/'weights.pt'),
            full_optimizer_entries=27,optimizer_rng_order_cursor_lanes_preserved=True,
            full_filtered_epochs_complete=x['epoch'],curriculum_stage_seen=x['curriculum_stage_seen'],
            training_stop_identity=l,selection='Actual final full checkpoint after user-requested graceful pause, no TEST selection')
    save('pause_checkpoint_verification.json',dict(complete=True,**cp))
    model,tok=make_model(actual);del actual,x;gc.collect()
    calibration=json.load(gzip.open(MAIN/'dev_calibration.json.gz','rt',encoding='utf-8'))
    monitor=json.load(gzip.open(MAIN/'dev_monitor.json.gz','rt',encoding='utf-8'))
    for n in ['dev_calibration.json.gz','dev_monitor.json.gz']:assert sha(MAIN/n)==manifest['files'][n]
    ev.OUT=JOB;ev.save=save;ev.status=status
    status('CALIBRATING_PAUSED_CHECKPOINT_ON_EXISTING_DEV',checkpoint_step=cp['step'])
    result=ev.evaluate(ev.predictor(model,tok),calibration,monitor,'frozen_dev')
    cp.update(threshold=result['threshold'],threshold_source='Same paused weights; existing original DEV calibration policy before any new TEST scoring',
              matching_frozen_dev_sha256=sha(JOB/'frozen_dev.json'))
    save('checkpoint_receipt.json',cp)
    del model;gc.collect();torch.cuda.empty_cache()
    return cp


def span_count(s,p):
    ends=[b for b in s['eots'] if p['start']<b<=p['end']]
    return len(ends)+int(not ends or ends[-1]<p['end']),len(ends)


def summarize_end_only(streams,trace_path):
    cuts=defaultdict(list);decisions=Counter();rawcuts=Counter()
    with gzip.open(trace_path,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);i,k,b=r['stream'],r['lookahead_words'],r['boundary']
            s=streams[i];j=next(j for j,p in enumerate(s['segments']) if p['start']<b and b+k<=p['end'])
            decisions[i,j,k]+=1
            if r['predicted_eot']:cuts[i,j,k].append(b);rawcuts[i,k]+=1
    groups=defaultdict(Counter);cases=[];stream_groups=defaultdict(Counter)
    for i,s in enumerate(streams):
        stream_units=0
        for j,p in enumerate(s['segments']):
            if not p.get('punctuation_present',True) or p['end']-p['start']<3:continue
            n,completed=span_count(s,p);stream_units+=n
            for k in (1,2):
                cs=cuts[i,j,k]
                assert decisions[i,j,k]==max(0,p['end']-p['start']-k)
                reference={b for b in s['eots'] if p['start']<b<=p['end']-k and b not in set(s['unknown'])}
                good=len(reference.intersection(cs));no_cut=int(not cs)
                cohort='multi_sentence' if n>=2 else 'single_sentence_excluded'
                bucket=str(n) if n<=5 else '6-10' if n<=10 else '>10'
                keys=[('ALL',k,cohort),('language:'+p['language'],k,cohort),('domain:'+s['domain']+'|'+p['language'],k,cohort),('sentence_count:'+bucket,k,cohort)]
                if completed>=2:keys.extend([('ALL',k,'at_least_two_completed_fullstops'),('language:'+p['language'],k,'at_least_two_completed_fullstops')])
                if not s['known_train_source_overlap'] and n>=2:keys.append(('ALL',k,'multi_without_known_legacy_overlap'))
                for key in keys:
                    c=groups[key];c['turns']+=1;c['no_internal_model_cut']+=no_cut;c['has_internal_model_cut']+=1-no_cut
                    c['words']+=p['end']-p['start'];c['words_in_no_cut_turns']+=(p['end']-p['start'])*no_cut
                    c['model_internal_cuts']+=len(cs);c['correct_fullstop_cuts']+=good;c['other_cuts']+=len(cs)-good
                    c['eligible_internal_fullstops']+=len(reference);c['missed_internal_fullstops']+=len(reference)-good
                    c['has_cuts_but_no_correct_fullstop_cut']+=int(bool(cs) and not good)
                    c['all_eligible_fullstops_missed']+=int(bool(reference) and not good)
                    c['no_evaluable_fullstop']+=int(not reference)
                if k==2 and n>=2 and len(cases)<60:
                    cases.append(dict(stream_key=s['key'],language=p['language'],domain=s['domain'],units=n,completed_fullstops=completed,
                        no_internal_model_cut=bool(no_cut),model_cut_count=len(cs),correct_fullstop_cuts=good,
                        source=p.get('display','')[:2500],internal_cut_offsets=[b-p['start'] for b in cs]))
        for k in (1,2):
            if stream_units>=2:
                c=stream_groups[k];c['streams']+=1;c['no_internal_model_cut']+=int(not rawcuts[i,k]);c['with_cuts']+=int(bool(rawcuts[i,k]))
    rows=[]
    for (group,k,cohort),c in sorted(groups.items()):
        rows.append(dict(group=group,lookahead_words=k,cohort=cohort,**c,
                         end_only_percent=100*c['no_internal_model_cut']/c['turns'],
                         no_cut_word_share_percent=100*c['words_in_no_cut_turns']/c['words']))
    out=dict(complete=True,definition='Primary denominator: actual source turns with >=2 fullstop-delimited spans; all one-span turns excluded. A nonempty residual span counts as one; strict >=2 completed fullstop subset also reported.',
             meaning='No model EOT strictly inside the source turn. Would be whole-turn delivery IF caller flushes at turn end; evaluator itself does not reset or flush at speaker/turn boundary.',
             final_flush_is_external_not_model_eot=True,model_cut_counts_include_unknown_label_positions=True,
             trace_sha256=sha(trace_path),rows=rows,whole_stream_multi_sentence=dict(stream_groups),examples=cases)
    save('end_only_analysis.json',out)
    return out


def verify_counting_fixture():
    """Check one-span exclusion and zero-cut denominator on isolated tiny inputs."""
    streams=[]
    for i,(n,ends) in enumerate([(7,[2,5,7]),(4,[4]),(6,[3,6])]):
        streams.append(dict(key='fixture:'+str(i),words=['word']*n,eots=ends,unknown=[],
            domain='fixture',known_train_source_overlap=False,
            segments=[dict(start=0,end=n,language='vi',punctuation_present=True,display='technical fixture')]))
    folder=JOB/'technical_fixture';folder.mkdir(exist_ok=True);path=folder/'end_only_trace.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for k in (1,2):
            for i,s in enumerate(streams):
                for b in range(1,len(s['words'])-k+1):
                    f.write(json.dumps(dict(stream=i,lookahead_words=k,boundary=b,predicted_eot=(i==0 and b==2)))+'\n')
    original=globals()['save']
    try:
        globals()['save']=lambda *a:None
        result=summarize_end_only(streams,path)
    finally:globals()['save']=original
    for k in (1,2):
        a=next(r for r in result['rows'] if r['group']=='ALL' and r['lookahead_words']==k and r['cohort']=='multi_sentence')
        b=next(r for r in result['rows'] if r['group']=='ALL' and r['lookahead_words']==k and r['cohort']=='single_sentence_excluded')
        assert a['turns']==2 and a['no_internal_model_cut']==1 and a['correct_fullstop_cuts']==1
        assert a['end_only_percent']==50 and b['turns']==1
    save('technical_fixture_verification.json',dict(complete=True,isolated_not_real_test=True,
        one_sentence_excluded=True,no_cut_and_correct_cut_accounting_verified=True))


def translation_input_replay(cp):
    """Run detector on the existing 310 ASR messages, without loading Qwen or references."""
    import numpy as np
    import torch
    from .common import normalize
    from .checked_dialogues_prepare_v1 import punctuation
    from .uploaded_vi_en_sentence_mix_evaluation_v7 import predictor
    oldreport=json.loads((MTOLD/'report.json').read_text())
    assert sha(MTOLD/'segmentation_trace.json')==oldreport['segmentation_sha256']
    assert sha(MTDATA/'eval_b.json')==oldreport['eval_sha256']
    old=json.loads((MTOLD/'segmentation_trace.json').read_text())
    convs=json.loads((MTDATA/'eval_b.json').read_text())
    source={(c['id'],m['id']):dict(text=m['text'],language=m['lang']) for c in convs for m in c['messages']}
    assert len(source)==len(old)==310
    model,tok=make_model(torch.load(JOB/'frozen/weights.pt',map_location='cpu',weights_only=True));predict=predictor(model,tok)
    rows=[];all_decisions=[]
    atoms=[r['source'].split() for r in old];starts=[0]*len(old);chunks=[[] for _ in old]
    for gap in range(1,max(map(len,atoms))):
        pending=[]
        for i,ws in enumerate(atoms):
            if gap>=len(ws):continue
            future=normalize(' '.join(ws[gap:])).split()[:2]
            if len(future)<2:continue
            context=normalize(' '.join(ws[starts[i]:gap]))
            pending.append((i,dict(context=context,text=' '.join(future))))
        if not pending:continue
        probs,seqs=predict([r for _,r in pending])
        for (i,r),p,seq in zip(pending,probs,seqs):
            cut=p>=cp['threshold']
            all_decisions.append(dict(message=i,gap=gap,context_start=starts[i],context=r['context'],future=r['text'],p_eot=p,cut=cut,
                                      encoded_sha256=hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()))
            if cut:chunks[i].append(' '.join(atoms[i][starts[i]:gap]));starts[i]=gap
    counts=defaultdict(Counter)
    for i,r in enumerate(old):
        src=source[r['conv'],r['id']];ws,ends,_=punctuation(src['text']);n=len(ends)+int(not ends or ends[-1]<len(ws))
        if starts[i]<len(atoms[i]):chunks[i].append(' '.join(atoms[i][starts[i]:]))
        assert ' '.join(chunks[i])==r['source']
        no_cut=len(chunks[i])==1;cohort='multi_sentence' if n>=2 else 'single_sentence_excluded'
        keys=[('ALL',cohort),('language:'+src['language'],cohort)]
        if len(ends)>=2:keys.append(('ALL','at_least_two_completed_fullstops'))
        for key in keys:
            c=counts[key];c['messages']+=1;c['end_only']+=int(no_cut);c['with_internal_cut']+=int(not no_cut)
            c['chunks']+=len(chunks[i]);c['words']+=len(atoms[i]);c['words_in_end_only']+=len(atoms[i])*int(no_cut)
        rows.append(dict(conv=r['conv'],id=r['id'],language=src['language'],sentence_spans=n,completed_fullstops=len(ends),
                         original_written=src['text'],asr_source=r['source'],chunks=chunks[i],end_only=no_cut,
                         final_flush_forced=True))
    save('translation_input_segmentation.json',rows)
    with gzip.open(JOB/'translation_input_decisions.jsonl.gz','wt',encoding='utf-8') as f:
        for r in all_decisions:f.write(json.dumps(r,separators=(',',':'))+'\n')
    # Independently reconstruct the entire saved causal input/atom cut process.
    from .tiny_compression_v1.common import encode_pair
    starts2=[0]*len(old);cut_gaps=defaultdict(list)
    for r in all_decisions:
        i,b=r['message'],r['gap'];assert r['context_start']==starts2[i]
        context=normalize(' '.join(atoms[i][starts2[i]:b]));future=' '.join(normalize(' '.join(atoms[i][b:])).split()[:2])
        ids=encode_pair(tok,context,future,256)['input_ids']
        assert context==r['context'] and future==r['future'] and len(future.split())==2
        assert hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest()==r['encoded_sha256']
        assert math.isfinite(r['p_eot']) and r['cut']==(r['p_eot']>=cp['threshold'])
        if r['cut']:starts2[i]=b;cut_gaps[i].append(b)
    for i,r in enumerate(rows):
        expected=[];start=0
        for b in cut_gaps[i]:expected.append(' '.join(atoms[i][start:b]));start=b
        if start<len(atoms[i]):expected.append(' '.join(atoms[i][start:]))
        assert expected==r['chunks']
    result=dict(complete=True,model_inferences=len(all_decisions),message_count=310,
        inference_scope='New paused detector only on same 310 ASR-style translation-test messages. Translator and references not used; no new BLEU/chrF or measured latency.',
        single_sentence_messages_excluded_from_primary_denominator=True,
        counts=[dict(group=g,cohort=c,**v,end_only_percent=100*v['end_only']/v['messages']) for (g,c),v in sorted(counts.items())],
        source_sha256=sha(MTDATA/'eval_b.json'),old_asr_segmentation_sha256=sha(MTOLD/'segmentation_trace.json'),
        decisions_sha256=sha(JOB/'translation_input_decisions.jsonl.gz'),segmentation_sha256=sha(JOB/'translation_input_segmentation.json'),
        all_causal_inputs_and_chunk_word_conservation_verified=True,final_message_flush_forced_not_model_eot=True)
    save('translation_input_end_only_report.json',result)
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def verify_numerical(streams,cp,report):
    import torch
    from .uploaded_vi_en_sentence_mix_evaluation_v7 import predictor
    from . import malbert_eval_zip_v2 as engine
    selected={};seen=Counter()
    with gzip.open(JOB/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);assert math.isfinite(r['p_eot']) and 0<=r['p_eot']<=1
            seen[r['lookahead_words']]+=1;s=streams[r['stream']];p=engine.segment_for(s,r['boundary'],r['lookahead_words'])
            length='short' if r['input_length']<=32 else 'medium' if r['input_length']<=128 else 'long'
            selected.setdefault((s['domain'],p['language'],r['lookahead_words'],length),r)
    model,tok=make_model(torch.load(JOB/'frozen/weights.pt',map_location='cpu',weights_only=True));predict=predictor(model,tok)
    maximum=0.;values=list(selected.values())
    for a in range(0,len(values),16):
        part=values[a:a+16]
        inputs=[dict(context=' '.join(streams[r['stream']]['words'][r['context_start']:r['boundary']]),text=' '.join(streams[r['stream']]['words'][r['boundary']:r['future_end']])) for r in part]
        probs,_=predict(inputs)
        for r,p in zip(part,probs):maximum=max(maximum,abs(p-r['p_eot']));assert (p>=cp['threshold'])==r['predicted_eot']
    assert maximum<1e-4 and sum(seen.values())==report['inferences']
    del model;gc.collect();torch.cuda.empty_cache()
    return dict(complete=True,canonical_predictor_cases=len(values),max_probability_delta=maximum,
                all_canonical_decisions_identical=True,all_probabilities_finite=True,inferences_by_lookahead=dict(seen))


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False
    u.is_librosa_available=im.is_librosa_available=lambda:False
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    assert not (JOB/'report.json').exists(),'Do not rerun this completed user test'
    reserve();paused();verify_counting_fixture();status('VERIFYING_PAUSED_FULL_CHECKPOINT')
    cp=freeze_and_calibrate()
    from . import malbert_eval_zip_v2 as engine
    engine.MAIN=MAIN;engine.JOB=JOB;engine.status=status;engine.save=save
    source=OLDTEST/'prepared_streams.json.gz';source_manifest=json.loads((OLDTEST/'input_manifest.json').read_text())
    assert sha(source)==source_manifest['prepared_streams_sha256']
    os.link(source,JOB/'prepared_streams.json.gz')
    streams=json.load(gzip.open(source,'rt',encoding='utf-8'));assert all(s['split']=='test' for s in streams)
    save('input_manifest.json',dict(**source_manifest,reused_verified_source_without_raw_rescan=True,source_path=str(source)))
    save('locked_protocol.json',dict(checkpoint=cp,test_point_source_sha256=sha(source),
        threshold_locked_before_test=True,exclude_one_sentence_from_end_only_denominator=True,
        preserve_source_turn_future_and_predicted_reset_contract=True,no_test_selection=True))
    status('EVALUATING_NEW_PAUSED_V7_TEST',checkpoint_step=cp['step'],threshold=cp['threshold'])
    result=engine.evaluate(streams,cp)
    status('AUDITING_ALL_NEW_TEST_INFERENCES',inferences=result['inferences'])
    proof=engine.verify_traces(streams,cp,result)
    numerical=verify_numerical(streams,cp,result)
    status('COUNTING_MULTI_SENTENCE_END_ONLY_TEST_TURNS')
    end=summarize_end_only(streams,JOB/'predictions.jsonl.gz')
    status('REPLAYING_PAUSED_DETECTOR_ON_310_TRANSLATION_TEST_INPUTS')
    mt=translation_input_replay(cp)
    assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256'];paused()
    result.update(complete=True,checkpoint=cp,causal_audit=proof,numerical_verification=numerical,
                  end_only_analysis_sha256=sha(JOB/'end_only_analysis.json'),
                  translation_input_report_sha256=sha(JOB/'translation_input_end_only_report.json'),
                  test_used_for_selection=False,training_remains_paused=True,translator_not_rerun=True,
                  model_scope='11.94M FP32 paused v7 detector, punctuation silver; no human/audio gold or new MT quality scores',
                  evaluator_source_sha256=sha(__file__))
    save('report.json',result)
    save('completion_verification.json',{**proof,**numerical,'complete':True,'report_sha256':sha(JOB/'report.json'),
                                        'end_only_rows':len(end['rows']),'translation_messages':mt['message_count']})
    status('COMPLETE',checkpoint_step=cp['step'],report_sha256=sha(JOB/'report.json'))


if __name__=='__main__':
    JOB.mkdir(exist_ok=True);os.nice(10)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as exc:
            save('error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at_utc=datetime.now(timezone.utc).isoformat()))
            status('ERROR',error=repr(exc));raise
