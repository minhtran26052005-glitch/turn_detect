"""Repeat the requested ZIP evaluation with a newly frozen current student."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import unicodedata
import zipfile

from .common import normalize
from .checked_dialogues_prepare_v1 import punctuation

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
MAIN=ROOT/'outputs/multilingual_malbert_vi_en_checked_20261103_v3'
JOB=ROOT/'outputs/malbert_eval_zip_20261104_v2'
PREVIOUS=ROOT/'outputs/malbert_eval_zip_20261104_v1'
MINIMUM_CHECKPOINT_STEP=1334000
ZIP_SHA='27efcebac44c5e56f3bbe7793c054f5f06a36b575d41292eb8e6f7f3c0e670e9'
SINGLE={'onevoice','numeric/sentence','numeric/code'}
POLICIES=('fullstop_matches_training','all_terminal_marks_diagnostic')


def sha(path):
    with Path(path).open('rb') as f:
        h=hashlib.sha256()
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=JOB/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage,**values):
    raw=Path('/proc/self/stat').read_text()
    value={'stage':stage,'pid':os.getpid(),'start_ticks':raw[raw.rfind(')')+2:].split()[19],
        'at_utc':datetime.now(timezone.utc).isoformat(),'training_interrupted':False,**values}
    save('status.json',value);print(json.dumps(value),flush=True)


def reserve():
    assert shutil.disk_usage(JOB).free>=13*2**30 and shutil.disk_usage('/mnt/hdd500').free>=13*2**30
    rows=[l for l in Path('/proc/mounts').read_text().splitlines() if l.split()[1]=='/mnt/hdd500']
    assert rows and rows[0].split()[0]=='/dev/sda1' and 'rw' in rows[0].split()[3].split(',')


def pin_checkpoint():
    import torch
    frozen=JOB/'frozen';frozen.mkdir(exist_ok=True)
    assert not (frozen/'student_resume_pin.pt').exists(),'Preserve an existing pin; never silently repin'
    status('WAITING_FOR_MATCHING_CHECKPOINT_AND_ORIGINAL_DEV')
    while True:
        reserve();source=MAIN/'runs/finetune/latest_resume.pt'
        with source.open('rb') as f:
            x=torch.load('/proc/self/fd/'+str(f.fileno()),map_location='cpu',weights_only=False,mmap=True)
            report_path=MAIN/'runs/finetune'/('dev_step_%08d.json'%x['step'])
            if report_path.exists() and x['step']>=MINIMUM_CHECKPOINT_STEP:
                dev=json.loads(report_path.read_text())
                assert dev['step']==x['step'] and x['parameters']==11941122 and len(x['optimizer']['state'])==27
                assert x['fingerprint']['active_languages']==['vi','en']
                assert sha(MAIN/'train_vi_en.jsonl')==x['fingerprint']['train_sha256']
                for name,value in x['fingerprint']['code_sha256'].items():assert sha(ROOT/'turn_v2'/name)==value
                assert {int(v['step']) for v in x['optimizer']['state'].values()}=={x['step']}
                assert len(x['order'])==399960 and len(set(map(int,x['order'])))==399960
                assert int(min(x['order']))==0 and int(max(x['order']))==399959
                assert x['epoch']>1 and x['additional_rows_seen']>199960
                threshold=float(dev['dev']['threshold']);assert 0<threshold<1
                f.seek(0);target=frozen/'student_resume_pin.pt'
                with target.open('xb') as dst:shutil.copyfileobj(f,dst,2**20);dst.flush();os.fsync(dst.fileno())
                torch.save(x['model'],frozen/'weights.pt')
                with (frozen/'weights.pt').open('rb') as dst:os.fsync(dst.fileno())
                for name in ['dev_rows_vi_en.jsonl','dev_streams_vi_en.json','dataset_manifest.json','recipe.json']:
                    os.link(MAIN/name,frozen/name)
                save('frozen/original_dev.json',dev)
                result={'step':x['step'],'epoch':x['epoch'],'parameters':11941122,'storage_compute':'Full precision mALBERT, not INT4',
                    'cursor':x['cursor'],'additional_row_training_exposures':x['additional_rows_seen'],
                    'full_399960_row_order_permutation_verified':True,'minimum_requested_step':MINIMUM_CHECKPOINT_STEP,
                    'optimizer_entries':27,'all_rng_and_sampler_present':all(k in x for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler_rng','order','cursor']),
                    'full_checkpoint_sha256':sha(target),'weights_sha256':sha(frozen/'weights.pt'),'original_dev_sha256':sha(frozen/'original_dev.json'),
                    'threshold':threshold,'threshold_source':'Matching original rolling VI/EN DEV at the same checkpoint, before external ZIP scoring',
                    'training_fingerprint':x['fingerprint'],'main_worker_not_paused':True,'external_eval_used_for_threshold_or_checkpoint_selection':False}
                save('checkpoint_receipt.json',result)
                save('locked_protocol.json',{'checkpoint':result,'label_policies':POLICIES,'lookahead_words':[1,2],
                    'source_text_field':'source_raw only, never target/rest_of_sentence/instruction',
                    'include_keep_false_if_source_has_punctuation':True,'known_train_overlap_included_and_tagged':True,
                    'source_turns_only_limit_future_availability':True,'conversation_history_reset':'Predicted EOT only; no gold/role/language resets',
                    'single_item_sources':sorted(SINGLE),'final_flush_is_model_eot':False,'test_tuning':False})
                del x;gc.collect();return result
        del x;gc.collect();time.sleep(1)


def build_stream(domain,identity,rows,known_overlap=False):
    """Preserve source order and words, using metadata solely for future availability."""
    rows=sorted(rows,key=lambda r:(r.get('chunk_idx',0),r['_row']))
    turns=defaultdict(list)
    for r in rows:
        key=(r.get('turn_idx',r['_row']),r.get('speaker','unspecified'),r['src_lang'])
        turns[key].append(r)
    ordered=sorted(turns.values(),key=lambda part:min((r.get('chunk_idx',0),r['_row']) for r in part))
    stream={'key':domain+':'+identity,'domain':domain,'split':rows[0]['split'],'dialogue_id':identity,
        'known_train_source_overlap':known_overlap,'words':[],'segments':[],'eots':[],'unknown':[],'questions':[],
        'source_rows':len(rows),'source_keep_false_rows':sum(r.get('keep') is False for r in rows)}
    for part in ordered:
        part.sort(key=lambda r:r.get('chunk_in_turn',0))
        indices=[r.get('chunk_in_turn',0) for r in part]
        continuous=indices==list(range(len(part)))
        blocks=[part] if continuous else [[r] for r in part]
        for block in blocks:
            text=' '.join(r['source_raw'].strip() for r in block)
            words,eots,unknown=punctuation(text);questions=set()
            for m in re.finditer(r'[?!？！]',unicodedata.normalize('NFKC',text)):
                questions.add(len(normalize(unicodedata.normalize('NFKC',text)[:m.end()]).split()))
            if not words:continue
            offset=len(stream['words']);stream['words'].extend(words)
            for name,values in [('eots',eots),('unknown',unknown),('questions',questions)]:stream[name].extend(offset+i for i in values)
            stream['segments'].append({'start':offset,'end':offset+len(words),'language':block[0]['src_lang'],
                'punctuation_present':bool(re.search(r'[.。?!？！,，;；:：…]',text)),
                'any_keep_false':any(r.get('keep') is False for r in block),'source_rows':[r['_row'] for r in block],
                'chunk_continuity':continuous,'display':text})
    assert stream['split'] in ('dev','test') and all(r['split']==stream['split'] for r in rows)
    return stream


def prepare_streams():
    assert sha(JOB/'inputs/eval.zip')==ZIP_SHA
    audit=json.loads((JOB/'inputs/usage_plan.json').read_text())
    overlap={r['dialogue_id'] for r in audit['contaminated_eval_factory_dialogues']};assert len(overlap)==54
    streams=[];coverage={};members={}
    with zipfile.ZipFile(JOB/'inputs/eval.zip') as z:
        for item in z.infolist():
            if item.is_dir() or not item.filename.endswith('labels_checked.jsonl'):continue
            assert item.filename.startswith('eval/') and '..' not in Path(item.filename).parts
            domain='/'.join(item.filename.split('/')[1:-1]);groups=defaultdict(list);stats=Counter();h=hashlib.sha256()
            with z.open(item) as f:
                for index,line in enumerate(f):
                    h.update(line)
                    if not line.strip():continue
                    r=json.loads(line);stats['source_rows']+=1
                    if r.get('src_lang') not in ('vi','en') or not isinstance(r.get('source_raw'),str):stats['unsupported_language_or_missing_source']+=1;continue
                    assert r.get('split') in ('dev','test')
                    r['_row']=index;stats['split:'+r['split']]+=1;stats['keep_false_rows']+=int(r.get('keep') is False)
                    identity=r['dialogue_id'] if domain not in SINGLE else r['dialogue_id']+':item:'+str(index)
                    groups[identity].append(r)
            members[item.filename]={'sha256':h.hexdigest(),'bytes':item.file_size,'source_rows':stats['source_rows']}
            for identity,rows in groups.items():
                s=build_stream(domain,identity,rows,domain=='factory' and identity in overlap)
                if not s['words']:stats['empty_streams']+=1;continue
                stats['streams']+=1;stats['normalized_words']+=len(s['words'])
                stats['source_segments']+=len(s['segments']);stats['segments_without_punctuation']+=sum(not p['punctuation_present'] for p in s['segments'])
                streams.append(s)
            coverage[domain]=dict(stats)
    assert len({s['key'] for s in streams})==len(streams)
    assert sum(s['known_train_source_overlap'] for s in streams)==54
    with gzip.open(JOB/'prepared_streams.json.gz','wt',encoding='utf-8') as f:json.dump(streams,f,ensure_ascii=False)
    manifest={'zip_sha256':ZIP_SHA,'members':members,'source_coverage':coverage,'streams':len(streams),
        'known_train_overlap_streams_included':54,'source_only':True,'target_translation_not_scored':True,
        'prepared_streams_sha256':sha(JOB/'prepared_streams.json.gz'),'word_conservation_from_normalized_source':True,
        'scope':'Punctuation-derived EOT/NEOT over source_raw; references are not human/audio turn gold'}
    save('input_manifest.json',manifest);return streams,manifest


def segment_for(stream,b,k):
    # Calls are monotonic per stream; a cached cursor avoids scanning long ASR records.
    for segment in stream['segments']:
        if segment['start']<=b-1<segment['end']:
            return segment if b+k<=segment['end'] and segment['punctuation_present'] else None
    return None


def reference(stream,b,policy):
    if policy==POLICIES[0]:return None if b in stream['_unknown'] else b in stream['_eots']
    return None if b in stream['_unknown'] and b not in stream['_questions'] else b in stream['_eots'] or b in stream['_questions']


def metric(c):
    tp,tn,fp,fn=[c.get(x,0) for x in ['tp','tn','fp','fn']];n=tp+tn+fp+fn
    return {'samples':n,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':(tp+tn)/n if n else None,
        'eot_f1':2*tp/(2*tp+fp+fn) if tp+fn else None,'eot_precision':tp/(tp+fp) if tp+fp else None,
        'eot_recall':tp/(tp+fn) if tp+fn else None,'false_share_of_cuts':fp/(tp+fp) if tp+fp else None,
        'false_positive_rate':fp/(tn+fp) if tn+fp else None,'actual_eot':tp+fn}


def kind(y,p):return 'tp' if y and p else 'fn' if y else 'fp' if p else 'tn'


def record_count(counts,stream,segment,k,b,p):
    for policy in POLICIES:
        actual=reference(stream,b,policy)
        if actual is None:continue
        for cohort in ['all_requested','without_known_train_source_overlap']:
            if cohort!='all_requested' and stream['known_train_source_overlap']:continue
            key=(stream['domain'],stream['split'],segment['language'],k,policy,cohort)
            counts[key][kind(actual,p)]+=1


def metric_rows(counts):
    total=defaultdict(Counter)
    for key,c in counts.items():total[('ALL',)+key[1:]].update(c)
    return [dict(zip(['domain','split','language','lookahead_words','label_policy','cohort'],key),**metric(c))
        for key,c in sorted({**counts,**total}.items())]


def initialize_stream_lookups(streams):
    for s in streams:
        for a,b in [('_eots','eots'),('_unknown','unknown'),('_questions','questions')]:s[a]=set(s[b])


def evaluate(streams,checkpoint):
    import numpy as np
    import torch
    from transformers import AlbertConfig,AutoTokenizer
    from .malbert_turn_student_v1 import MalbertTurnStudent
    from .tiny_compression_v1.common import encode_pair
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    model=MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    weights=torch.load(JOB/'frozen/weights.pt',map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
    assert model.parameter_count()==11941122;model.cuda().eval()
    tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    initialize_stream_lookups(streams);counts=defaultdict(Counter);coverage=Counter();cases=[];case_counts=Counter();threshold=checkpoint['threshold'];inferences=0
    assert tok.pad_token_id==0 and tok.sep_token_id==3
    batch_size=64;batch_reductions=[]
    def predict_part(part):
        width=max(len(p[3]) for p in part)
        ids=torch.zeros((len(part),width),dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for j,(_,_,_,seq) in enumerate(part):ids[j,:len(seq)]=torch.tensor(seq,device='cuda');mask[j,:len(seq)]=1
        with torch.inference_mode():return model(ids,mask).float().softmax(-1)[:,1].cpu().tolist()
    with gzip.open(JOB/'predictions.jsonl.gz','wt',encoding='utf-8',compresslevel=5) as trace:
        for k in (1,2):
            starts=[0]*len(streams)
            for b in range(1,max(len(s['words']) for s in streams)-k+1):
                pending=[]
                for i,s in enumerate(streams):
                    if b+k>len(s['words']):continue
                    segment=segment_for(s,b,k)
                    if segment is None:coverage[str(k)+':no_same_segment_future_or_no_punctuation']+=1;continue
                    row={'context':' '.join(s['words'][starts[i]:b]),'text':' '.join(s['words'][b:b+k])}
                    assert len(row['text'].split())==k
                    ids=encode_pair(tok,row['context'],row['text'],256)['input_ids']
                    pending.append((i,segment,row,ids))
                pending.sort(key=lambda item:len(item[3]))
                start=0
                while start<len(pending):
                    part=pending[start:start+batch_size]
                    try:probs=predict_part(part)
                    except torch.cuda.OutOfMemoryError:
                        if batch_size<=8:raise
                        batch_reductions.append({'lookahead_words':k,'boundary':b,'from':batch_size,'to':batch_size//2})
                        batch_size//=2;gc.collect();torch.cuda.empty_cache();continue
                    for (i,segment,row,seq),prob in zip(part,probs):
                        s=streams[i];predicted=prob>=threshold
                        record_count(counts,s,segment,k,b,predicted);coverage[str(k)+':inferences']+=1
                        r={'stream':i,'lookahead_words':k,'boundary':b,'context_start':starts[i],'future_end':b+k,'p_eot':prob,
                            'predicted_eot':predicted,'reference_fullstop':reference(s,b,POLICIES[0]),'reference_all_marks':reference(s,b,POLICIES[1]),
                            'input_ids_sha256':hashlib.sha256(np.asarray(seq,dtype='<u2').tobytes()).hexdigest(),'input_length':len(seq),
                            'known_train_source_overlap':s['known_train_source_overlap']}
                        trace.write(json.dumps(r,separators=(',',':'))+'\n');inferences+=1
                        actual=r['reference_fullstop'];case_key=(s['domain'],s['split'],segment['language'],k,kind(actual,predicted)) if actual is not None else None
                        if actual is not None and case_counts[case_key]<2:
                            case_counts[case_key]+=1;cases.append({**r,'domain':s['domain'],'split':s['split'],'language':segment['language'],
                                'dialogue_id':s['dialogue_id'],'visible_context_tail':' '.join(s['words'][max(starts[i],b-40):b]),
                                'future':row['text'],'source_display':segment['display'][:1200],'case':kind(actual,predicted)})
                        if predicted:starts[i]=b
                    start+=len(part)
                if b%50==0:status('EVALUATING_EXTERNAL_EVAL_ZIP',lookahead_words=k,boundary=b,inferences=inferences,batch_size=batch_size)
    del model;gc.collect();torch.cuda.empty_cache()
    result={'inferences':inferences,'metric_rows':metric_rows(counts),'coverage':dict(coverage),'predictions_sha256':sha(JOB/'predictions.jsonl.gz'),
        'batching':{'length_bucketing':True,'initial_batch_size':64,'final_batch_size':batch_size,'oom_reductions':batch_reductions,
                    'chronological_order_within_every_stream_preserved':True,'compute_precision':'FP32'}}
    save('review_cases.json',cases);save('partial_report.json',result);return result


def verify_traces(streams,checkpoint,report):
    import numpy as np
    from transformers import AutoTokenizer
    from .tiny_compression_v1.common import encode_pair
    tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    starts=defaultdict(int);last={};counts=defaultdict(Counter);seen=0;cut_words=Counter();seen_per_stream=Counter()
    with gzip.open(JOB/'predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);i,k,b=r['stream'],r['lookahead_words'],r['boundary'];s=streams[i];key=(i,k);segment=segment_for(s,b,k)
            assert segment is not None and r['context_start']==starts[key] and b>last.get(key,0) and r['future_end']==b+k
            assert r['reference_fullstop']==reference(s,b,POLICIES[0]) and r['reference_all_marks']==reference(s,b,POLICIES[1])
            context=' '.join(s['words'][starts[key]:b]);text=' '.join(s['words'][b:b+k]);assert len(text.split())==k
            ids=encode_pair(tok,context,text,256)['input_ids']
            assert hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()==r['input_ids_sha256'] and len(ids)==r['input_length']<=256
            predicted=r['p_eot']>=checkpoint['threshold'];assert predicted==r['predicted_eot']
            record_count(counts,s,segment,k,b,predicted)
            if predicted:cut_words[key]+=b-starts[key];starts[key]=b
            last[key]=b;seen+=1;seen_per_stream[key]+=1
    for i,s in enumerate(streams):
        for k in (1,2):
            assert cut_words[(i,k)]+len(s['words'])-starts[(i,k)]==len(s['words'])
            expected=sum(max(0,p['end']-p['start']-k) for p in s['segments'] if p['punctuation_present'])
            assert seen_per_stream[(i,k)]==expected,(i,k,seen_per_stream[(i,k)],expected)
    assert metric_rows(counts)==report['metric_rows'] and seen==report['inferences']
    assert sha(JOB/'frozen/student_resume_pin.pt')==checkpoint['full_checkpoint_sha256']
    assert sha(JOB/'frozen/weights.pt')==checkpoint['weights_sha256']
    assert sha(JOB/'predictions.jsonl.gz')==report['predictions_sha256']
    return {'all_inferences_audited':seen,'every_encoded_input_and_causal_future_verified':True,'predicted_only_resets_verified':True,
        'all_eligible_points_accounted_for':True,'all_confusion_counts_verified':True,'source_words_conserved_and_final_flush_not_eot':True,'checkpoint_weights_and_trace_sha_verified':True}


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False
    u.is_librosa_available=im.is_librosa_available=lambda:False
    assert not (JOB/'report.json').exists(),'Do not rerun a completed user test'
    reserve();train_sha_before=sha(MAIN/'train_vi_en.jsonl');main_before=json.loads((MAIN/'status.json').read_text())
    checkpoint=pin_checkpoint();status('PREPARING_PUNCTUATED_EVAL_SOURCES',checkpoint_step=checkpoint['step'])
    streams,manifest=prepare_streams();report=evaluate(streams,checkpoint)
    status('AUDITING_ALL_CAUSAL_INFERENCES',inferences=report['inferences']);proof=verify_traces(streams,checkpoint,report)
    assert sha(MAIN/'train_vi_en.jsonl')==train_sha_before
    previous=json.loads((PREVIOUS/'report.json').read_text())
    old_streams=json.load(gzip.open(PREVIOUS/'prepared_streams.json.gz','rt',encoding='utf-8'))
    clean_streams=[{k:v for k,v in s.items() if not k.startswith('_')} for s in streams]
    assert clean_streams==old_streams,'Source point population must match the previous requested test'
    assert manifest['members']==previous['input_manifest']['members']
    old_rows={tuple(r[k] for k in ['domain','split','language','lookahead_words','label_policy','cohort']):r for r in previous['metric_rows']}
    comparison=[]
    for r in report['metric_rows']:
        key=tuple(r[k] for k in ['domain','split','language','lookahead_words','label_policy','cohort']);old=old_rows[key]
        assert r['samples']==old['samples'] and r['actual_eot']==old['actual_eot']
        if r['cohort']=='all_requested' and r['lookahead_words']==2 and r['label_policy']==POLICIES[0] and (r['domain']=='ALL' or (r['split']=='test' and r['domain'] in ('asr','daily','factory'))):
            comparison.append({'domain':r['domain'],'split':r['split'],'language':r['language'],
                               'previous':old,'current':r})
    save('comparison_with_827000.json',{'same_source_streams_and_eligible_point_population':True,
         'old_report_sha256':sha(PREVIOUS/'report.json'),'old_checkpoint':previous['checkpoint']['step'],
         'new_checkpoint':checkpoint['step'],'old_threshold':previous['checkpoint']['threshold'],'new_threshold':checkpoint['threshold'],
         'thresholds_independently_locked_from_each_checkpoint_original_dev':True,'previous_predictions_reused_not_rescored':True,
         'comparison_combines_checkpoint_and_dev_selected_threshold_changes':True,'rows':comparison})
    report.update(complete=True,checkpoint=checkpoint,input_manifest=manifest,causal_audit=proof,
        label_policies=list(POLICIES),primary_label_policy=POLICIES[0],all_requested_known_overlaps_scored=True,
        validation_test_scored_only_not_used_for_selection=True,main_training_dataset_unchanged=True,
        model_scope='11.94M full precision mALBERT; source punctuation evaluation, not translation quality or human/audio turn gold',
        main_status_before=main_before,main_status_after=json.loads((MAIN/'status.json').read_text()),
        main_dataset_sha256=train_sha_before,evaluator_source_sha256=sha(__file__),backed_up_to_drive=False)
    report['comparison_with_827000_sha256']=sha(JOB/'comparison_with_827000.json')
    save('report.json',report);save('completion_verification.json',{'complete':True,'report_sha256':sha(JOB/'report.json'),**proof})
    status('COMPLETE',checkpoint_step=checkpoint['step'],inferences=report['inferences'],report_sha256=sha(JOB/'report.json'))


if __name__=='__main__':
    import fcntl
    JOB.mkdir(parents=True,exist_ok=True);os.nice(10)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:
            save('error.json',{'error':repr(e),'traceback':traceback.format_exc(),'at_utc':datetime.now(timezone.utc).isoformat()});status('ERROR',error=repr(e));raise
