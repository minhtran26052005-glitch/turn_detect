"""Technical continuation: sequential audit and reuse every frozen v1 prediction."""
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sqlite3
import traceback

from . import multilingual_student_diagnostic_v1 as base

ORIGINAL=base.JOB
JOB=ORIGINAL.with_name('multilingual_distill_test_20261031_v2')
base.JOB=JOB


def run():
    import numpy as np
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
    utils.is_librosa_available=imports.is_librosa_available=lambda:False
    from transformers import AutoConfig,AutoModelForSequenceClassification,AutoTokenizer
    from safetensors.torch import load_file
    from .monolingual_data import encode_tail
    torch.set_num_threads(2)
    assert not (JOB/'report.json').exists(),'Never repeat this completed test'
    manifest=json.loads((ORIGINAL/'input_manifest.json').read_text())
    assert base.sha(Path(base.__file__))==manifest['source_sha256']
    snapshot=json.loads((ORIGINAL/'checkpoint_receipt.json').read_text())
    dev=json.loads((ORIGINAL/'frozen_dev.json').read_text())
    assert dev['checkpoint_step']==snapshot['step'] and dev['threshold_locked_before_test']
    frozen=ORIGINAL/'frozen';threshold=dev['shared_threshold']
    assert base.sha(frozen/'model.safetensors')==snapshot['weights_sha256']
    for lang,meta in manifest['test_files'].items():assert base.sha(meta['path'])==meta['sha256']
    base.save('checkpoint_receipt.json',snapshot);base.save('frozen_dev.json',dev)
    manifest['technical_continuation']={'version':2,'original':str(ORIGINAL),'source_sha256':base.sha(Path(__file__)),
        'same_weights_inputs_and_dev_threshold':True,'existing_probabilities_reused_without_rescoring':True,
        'fix':'sequential covering-index audit instead of per-row random HDD reads; source hashes from original corpus'}
    base.save('input_manifest.json',manifest)
    base.status('LOADING_SAME_FROZEN_STUDENT',checkpoint_step=snapshot['step'])
    tok=AutoTokenizer.from_pretrained(frozen,local_files_only=True)
    teacher_tok=AutoTokenizer.from_pretrained(base.SOURCE/'teachers/vi/best',local_files_only=True)
    model=AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(frozen,local_files_only=True),dtype=torch.float32)
    model.load_state_dict(load_file(str(frozen/'model.safetensors')),strict=True);model.cuda().eval()
    db=sqlite3.connect(base.DATA.resolve().as_uri()+'?mode=ro',uri=True,timeout=30);db.execute('PRAGMA cache_size=-65536')
    corpus=sqlite3.connect((base.SOURCE/'corpus.sqlite').resolve().as_uri()+'?mode=ro',uri=True,timeout=30)

    def predict(rows):
        seqs=[encode_tail(tok,r['context'],r['text'],256)['input_ids'] for r in rows]
        width=max(map(len,seqs));ids=torch.full((len(rows),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.as_tensor(seq,device='cuda');mask[i,:len(seq)]=1
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return model(input_ids=ids,attention_mask=mask).logits.float().softmax(-1)[:,1].cpu().tolist()

    report={'complete':False,'scope':'Previously exposed punctuation-derived C75 diagnostic; not independent conversational test',
        'checkpoint':snapshot,'threshold_from_same_checkpoint_dev':threshold,'threshold_selected_on_test':False,
        'checkpoint_selected_on_test':False,'languages':{},'quality_target_certified':False,
        'technical_continuation':manifest['technical_continuation']}
    for lang in base.LANGS:
        base.status('ENCODING_AUDIT_INPUTS',language=lang,rows_done=0)
        with gzip.open(manifest['test_files'][lang]['path'],'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
        assert len(rows)==manifest['test_files'][lang]['expected_rows']
        hashes=[]
        for i,row in enumerate(rows):
            assert row['language']==lang and row['included_in_clean_test']
            ids=encode_tail(teacher_tok,row['context'],row['text'],256)['input_ids']
            hashes.append(base.hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest())
            if (i+1)%5000==0:base.status('ENCODING_AUDIT_INPUTS',language=lang,rows_done=i+1)
        wanted=set(hashes);blocked={};scanned=0
        for split in ('train','dev'):
            sql="SELECT input_hash FROM records INDEXED BY accepted_fingerprint_cover WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id"
            for (key,) in db.execute(sql,(lang,split)):
                if key in wanted:blocked[key]=split
                scanned+=1
                if scanned%250000==0:base.status('AUDITING_EXACT_INPUTS_SEQUENTIAL',language=lang,scanned=scanned,matching_keys=len(blocked))
        # The preparation database omits protected test sources. Fetch C75
        # source hashes from the corpus rather than assuming missing means safe.
        non_test_hashes={h for (h,) in db.execute("SELECT text_sha256 FROM processed_sources WHERE language=? AND split IN ('train','dev')",(lang,)) if h}
        sources={};missing=0
        for sid in {r['source_id'] for r in rows}:
            meta=corpus.execute('SELECT text_sha256,split FROM sources WHERE language=? AND page_id=?',(lang,sid.split(':',1)[1])).fetchone()
            if meta is None:missing+=1;sources[sid]=False
            else:sources[sid]=meta[1] in ('train','dev') or meta[0] in non_test_hashes
        cached={};old_trace=ORIGINAL/f'predictions_{lang}.jsonl.gz'
        if old_trace.exists():
            with gzip.open(old_trace,'rt',encoding='utf-8') as f:
                for line in f:
                    record=json.loads(line);assert record['example_id'] not in cached
                    cached[record['example_id']]=record
        cached_sha=base.sha(old_trace) if old_trace.exists() else None
        all_ids={r['example_id'] for r in rows};assert set(cached)<=all_ids
        raw_counts=Counter();clean_counts=Counter();by_k={'1':Counter(),'2':Counter()};audit=Counter();review=[];total=0;reused=0
        trace=JOB/f'predictions_{lang}.jsonl.gz'
        base.status('SCORING_C75',language=lang,rows_done=0,reusable_predictions=len(cached),overlap_keys=len(blocked))
        with gzip.open(trace,'wt',encoding='utf-8') as target:
            for begin in range(0,len(rows),64):
                subset=rows[begin:begin+64];need=[r for r in subset if r['example_id'] not in cached]
                fresh=iter(predict(need) if need else [])
                for offset,row in enumerate(subset):
                    old=cached.get(row['example_id'])
                    if old is not None:
                        assert all(old[k]==row[k] for k in ('context','text','label','lookahead_words','source_id'))
                        prob=old['p_eot'];reused+=1
                    else:prob=next(fresh)
                    assert 0<=prob<=1 and row['label'] in ('NEOT','EOT') and row['lookahead_words'] in (1,2)
                    key=hashes[begin+offset];reasons=[]
                    if key in blocked:reasons.append('exact_cycle84_accepted_'+blocked[key]+'_teacher_input')
                    if sources[row['source_id']]:reasons.append('cycle84_non_test_source_or_source_copy')
                    keep=not reasons
                    for reason in reasons:audit[reason]+=1
                    audit['retained_rows' if keep else 'excluded_rows']+=1
                    actual=row['label']=='EOT';pred=prob>=threshold;kind='tp' if actual and pred else 'fn' if actual else 'fp' if pred else 'tn'
                    raw_counts[kind]+=1
                    if keep:clean_counts[kind]+=1;by_k[str(row['lookahead_words'])][kind]+=1
                    record={'example_id':row['example_id'],'language':lang,'context':row['context'],'text':row['text'],'label':row['label'],
                        'lookahead_words':row['lookahead_words'],'source_id':row['source_id'],'p_eot':prob,'predicted':'EOT' if pred else 'NEOT',
                        'correct':pred==actual,'included_after_cycle84_audit':keep,'exclusion_reasons':reasons,'reused_from_v1':old is not None}
                    target.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if keep and (len(review)<8 or (not record['correct'] and sum(not r['correct'] for r in review)<8)):review.append(record)
                total+=len(subset)
                if total%3200==0:base.status('SCORING_C75',language=lang,rows_done=total,reused=reused,excluded=audit['excluded_rows'])
        assert total==len(rows) and reused==len(cached) and base.sha(manifest['test_files'][lang]['path'])==manifest['test_files'][lang]['sha256']
        result={'input_rows':total,'metrics_after_cycle84_audit':base.metrics(clean_counts),
            'existing_C75_subset_metrics_before_cycle84_audit':base.metrics(raw_counts),'by_lookahead_words':{k:base.metrics(v) for k,v in by_k.items()},
            'overlap_audit':{**dict(audit),'accepted_train_dev_rows_scanned':scanned,'missing_source_groups_in_corpus':missing},
            'audit_scope':'Exclude exact 256-token teacher inputs in cycle84 accepted train/dev and source IDs/exact source copies in cycle84 non-test; test source hashes read from corpus',
            'audit_limits':'Student WordPiece-only collisions and near-duplicate paraphrases are not exhaustively audited; C75 has already been viewed',
            'dataset_sha256':manifest['test_files'][lang]['sha256'],'predictions_sha256':base.sha(trace),'predictions_bytes':trace.stat().st_size,
            'v1_predictions_reused':reused,'v1_trace_sha256':cached_sha}
        report['languages'][lang]=result;base.save(f'review_{lang}.json',review);base.save('partial_report.json',report)
        base.status('LANGUAGE_COMPLETE',language=lang,metrics=result['metrics_after_cycle84_audit'],excluded=audit['excluded_rows'])
        del rows,hashes,cached
    db.close();corpus.close();report.update(complete=True,completed_utc=base.now(),gpu_peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30,2))
    base.save('report.json',report);base.status('COMPLETE',checkpoint_step=snapshot['step'],report_sha256=base.sha(JOB/'report.json'))


if __name__=='__main__':
    import fcntl
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'EVALUATOR_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as exc:
            base.save('error.json',{'at_utc':base.now(),'error':repr(exc),'traceback':traceback.format_exc()});base.status('ERROR',error=repr(exc));raise
