"""One frozen shared student on the exposed C75 diagnostic, never for selection."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import traceback

PROJECT=Path('/home/edabk/hoangbpm/turn_detect')
CAMPAIGN=Path('/mnt/hdd500/turn_detect_training/multilingual_distill_v1')
JOB=Path('/mnt/hdd500/turn_detect_training/multilingual_distill_test_20261031_v1')
TEST=PROJECT/'staging/test'
SOURCE=PROJECT/'outputs/translation_segmenter_v15d_20261026'
DATA=SOURCE/'full_epoch_v1/datasets/cycle000084_full_epoch_contiguous_v1/dataset.sqlite'
LANGS=('vi','en','zh')


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):value.update(block)
    return value.hexdigest()


def save(name,value):
    path=JOB/name;temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
           'at_utc':now(),'training_interrupted':False,**extra}
    save('status.json',value);print(json.dumps(value),flush=True)


def metrics(counts):
    tp,tn,fp,fn=(counts.get(k,0) for k in ('tp','tn','fp','fn'))
    n=tp+tn+fp+fn
    return {'samples':n,'accuracy':(tp+tn)/n if n else None,
        'eot_precision':tp/(tp+fp) if tp+fp else 0.,'eot_recall':tp/(tp+fn) if tp+fn else 0.,
        'eot_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def run():
    import numpy as np
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    from safetensors.torch import save_file
    from .monolingual_data import encode_tail
    from .multilingual_three_teacher_distill_v1 import rows_after,validate_row,MODEL,REV,HF_CACHE
    torch.set_num_threads(2)
    if (JOB/'report.json').exists():raise RuntimeError('This completed evaluation must not be repeated')
    assert torch.cuda.is_available() and shutil.disk_usage(JOB).free>=15*2**30
    assert Path('/mnt/hdd500').is_mount() and DATA.exists()
    free,_total=torch.cuda.mem_get_info();assert free>=4*2**30,'Insufficient free GPU memory alongside training'
    recipe=json.loads((CAMPAIGN/'recipe.json').read_text())
    dataset=json.loads((SOURCE/'full_epoch_v1/current_dataset.json').read_text())
    assert recipe['dataset_sha256']==dataset['database_sha256'] and recipe['student_revision']==REV
    status('PINNING_CURRENT_STUDENT')
    frozen=JOB/'frozen';frozen.mkdir(exist_ok=False)
    pin=frozen/'student_resume_pin.pt';os.link(CAMPAIGN/'latest_student.pt',pin)
    checkpoint_sha=sha(pin);state=torch.load(pin,map_location='cpu',weights_only=False)
    assert state['dataset_sha256']==recipe['dataset_sha256'] and state['generation']==recipe['generation']
    assert state['teacher_hashes']=={l:v['sha256'] for l,v in recipe['teachers'].items()}
    assert state['optimizer']['state'] and all(k in state for k in ['torch_rng','cuda_rng','numpy_rng','python_rng','cursors'])
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,cache_dir=str(HF_CACHE),local_files_only=True,token=False)
    cfg.num_labels=2;cfg.id2label={0:'NEOT',1:'EOT'};cfg.label2id={'NEOT':0,'EOT':1}
    cfg.save_pretrained(frozen)
    save_file({k:v.contiguous() for k,v in state['model'].items()},str(frozen/'model.safetensors'),metadata={'format':'pt'})
    tokenizer=AutoTokenizer.from_pretrained(CAMPAIGN/'student_tokenizer',local_files_only=True)
    tokenizer.save_pretrained(frozen)
    teacher_tokenizer=AutoTokenizer.from_pretrained(SOURCE/'teachers/vi/best',local_files_only=True)
    model=AutoModelForSequenceClassification.from_config(cfg,dtype=torch.float32)
    model.load_state_dict(state['model'],strict=True);model.cuda().eval()
    snapshot={'step':state['step'],'counts':state['counts'],'cursors':state['cursors'],'rounds':state['rounds'],
        'source_checkpoint_sha256':checkpoint_sha,'weights_sha256':sha(frozen/'model.safetensors'),
        'optimizer_rng_sampler_preserved_in_pin':True,'dataset_sha256':recipe['dataset_sha256'],
        'student_base':MODEL,'student_revision':REV,'same_model_for_all_languages':True,
        'training_continues_independently':True,'test_used_for_selection':False}
    del state
    save('checkpoint_receipt.json',snapshot)
    old_report=json.loads((TEST/'report.json').read_text())
    expected_counts={l:old_report['languages'][l]['metrics']['samples'] for l in LANGS}
    test_files={}
    for lang in LANGS:
        source=TEST/f'clean_{lang}.jsonl.gz';dest=frozen/source.name
        shutil.copyfile(source,dest);digest=sha(dest);assert digest==sha(source)
        test_files[lang]={'path':str(dest),'sha256':digest,'expected_rows':expected_counts[lang]}
    save('input_manifest.json',{'checkpoint':snapshot,'test_files':test_files,'source_c75_report_sha256':sha(TEST/'report.json'),
        'scope':'Previously exposed C75 prose diagnostic, not independent human/audio conversational gold',
        'heldout_test_v2_untouched':True,'unscored_reserve_untouched':True,
        'source_sha256':sha(Path(__file__)),'frozen_files_sha256':{p.name:sha(p) for p in frozen.iterdir() if p.is_file() and p.suffix!='.pt'}})
    db=sqlite3.connect(DATA.resolve().as_uri()+'?mode=ro',uri=True,timeout=30);db.execute('PRAGMA cache_size=-65536')

    def predict(rows):
        # Only these two strings enter the encoder, never old predictions/labels/punctuation metadata.
        seqs=[encode_tail(tokenizer,row['context'],row['text'],256)['input_ids'] for row in rows]
        width=max(map(len,seqs));ids=torch.full((len(rows),width),tokenizer.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.as_tensor(seq,device='cuda');mask[i,:len(seq)]=1
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model(input_ids=ids,attention_mask=mask).logits.float().softmax(-1)[:,1].cpu().tolist()

    status('CALIBRATING_FROZEN_CHECKPOINT_ON_DEV',checkpoint_step=snapshot['step'])
    dev={}
    for lang in LANGS:
        rows=[validate_row(r) for r in rows_after(db,lang,'dev','',1000)];probs=[]
        for begin in range(0,len(rows),64):probs.extend(predict(rows[begin:begin+64]))
        dev[lang]=(np.asarray(probs),np.asarray([r['label']=='EOT' for r in rows]))
    def dev_metric(p,y,t):
        pred=p>=t
        return metrics({'tp':int((pred&y).sum()),'tn':int((~pred&~y).sum()),'fp':int((pred&~y).sum()),'fn':int((~pred&y).sum())})
    threshold=float(max(np.linspace(.1,.9,81),key=lambda t:min(dev_metric(*dev[l],t)['eot_f1'] for l in LANGS)))
    save('frozen_dev.json',{'checkpoint_step':snapshot['step'],'shared_threshold':threshold,
        'by_language':{l:dev_metric(*dev[l],threshold) for l in LANGS},'sample_policy':'Same first 1000 hashed IDs per language as campaign DEV',
        'threshold_locked_before_test':True,'test_used_for_selection':False})
    report={'complete':False,'scope':'Previously exposed punctuation-derived C75 diagnostic; not independent conversational test',
        'checkpoint':snapshot,'threshold_from_same_checkpoint_dev':threshold,'threshold_selected_on_test':False,
        'checkpoint_selected_on_test':False,'languages':{},'quality_target_certified':False}
    for lang in LANGS:
        status('AUDITING_AND_SCORING_C75',language=lang,rows_done=0,checkpoint_step=snapshot['step'])
        source_meta={pid:(h,split) for pid,h,split in db.execute('SELECT page_id,text_sha256,split FROM processed_sources WHERE language=?',(lang,))}
        non_test_hashes={h for h,split in source_meta.values() if split in ('train','dev')}
        source_cache={};audit=Counter();total=0;raw_counts=Counter();clean_counts=Counter();by_k={'1':Counter(),'2':Counter()};review=[]
        trace=JOB/f'predictions_{lang}.jsonl.gz'
        with gzip.open(test_files[lang]['path'],'rt',encoding='utf-8') as source,gzip.open(trace,'wt',encoding='utf-8') as target:
            def score(rows):
                nonlocal total
                probabilities=predict(rows)
                for row,prob in zip(rows,probabilities):
                    assert row['language']==lang and row['included_in_clean_test'] and row['label'] in ('NEOT','EOT') and row['lookahead_words'] in (1,2)
                    ids=encode_tail(teacher_tokenizer,row['context'],row['text'],256)['input_ids']
                    key=hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest()
                    exact=db.execute("SELECT split FROM records INDEXED BY accepted_encoded_input_unique WHERE language=? AND input_hash=? AND disposition='accepted' AND split IN ('train','dev')",(lang,key)).fetchone()
                    sid=row['source_id']
                    if sid not in source_cache:
                        pid=sid.split(':',1)[1];meta=source_meta.get(pid)
                        source_cache[sid]=bool(meta and (meta[1] in ('train','dev') or meta[0] in non_test_hashes))
                    reasons=[]
                    if exact:reasons.append('exact_cycle84_accepted_'+exact[0]+'_teacher_input')
                    if source_cache[sid]:reasons.append('cycle84_non_test_source_or_source_copy')
                    keep=not reasons
                    for reason in reasons:audit[reason]+=1
                    audit['excluded_rows' if reasons else 'retained_rows']+=1
                    actual=row['label']=='EOT';pred=prob>=threshold
                    kind='tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn'
                    raw_counts[kind]+=1
                    if keep:clean_counts[kind]+=1;by_k[str(row['lookahead_words'])][kind]+=1
                    record={'example_id':row['example_id'],'language':lang,'context':row['context'],'text':row['text'],
                        'label':row['label'],'lookahead_words':row['lookahead_words'],'source_id':sid,
                        'p_eot':prob,'predicted':'EOT' if pred else 'NEOT','correct':pred==actual,
                        'included_after_cycle84_audit':keep,'exclusion_reasons':reasons}
                    target.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if keep and (len(review)<8 or (not record['correct'] and sum(not r['correct'] for r in review)<8)):review.append(record)
                total+=len(rows)
                if total%3200==0:status('AUDITING_AND_SCORING_C75',language=lang,rows_done=total,excluded=audit['excluded_rows'])
            batch=[]
            for line in source:
                batch.append(json.loads(line))
                if len(batch)==64:score(batch);batch=[]
            if batch:score(batch)
        assert total==test_files[lang]['expected_rows'] and sha(test_files[lang]['path'])==test_files[lang]['sha256']
        result={'input_rows':total,'metrics_after_cycle84_audit':metrics(clean_counts),'existing_C75_subset_metrics_before_cycle84_audit':metrics(raw_counts),
            'by_lookahead_words':{k:metrics(v) for k,v in by_k.items()},'overlap_audit':dict(audit),
            'audit_scope':'Exclude exact 256-token teacher inputs in cycle84 accepted train/dev and source IDs/exact source copies in cycle84 non-test',
            'audit_limits':'Student WordPiece-only collisions and near-duplicate paraphrases are not exhaustively audited; C75 has already been viewed',
            'dataset_sha256':test_files[lang]['sha256'],'predictions_sha256':sha(trace),'predictions_bytes':trace.stat().st_size}
        report['languages'][lang]=result;save(f'review_{lang}.json',review);save('partial_report.json',report)
        status('LANGUAGE_COMPLETE',language=lang,metrics=result['metrics_after_cycle84_audit'],excluded=audit['excluded_rows'])
    db.close();report.update(complete=True,completed_utc=now(),gpu_peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,2))
    save('report.json',report);status('COMPLETE',checkpoint_step=snapshot['step'],report_sha256=sha(JOB/'report.json'))


if __name__=='__main__':
    import fcntl
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as exc:
            save('error.json',{'at_utc':now(),'error':repr(exc),'traceback':traceback.format_exc()})
            status('ERROR',error=repr(exc));raise
