"""One fixed legacy-dev transfer evaluation on Colab; no train or calibration."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import unicodedata
import re

JOB = Path('/content/turn_detect_old_dev_c75_vi_v1')


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)
def state(stage, **values): atomic(JOB/'status.json',{'at_utc':now(),'stage':stage,**values})
def normalize(text):
    s=unicodedata.normalize('NFKC',text).lower()
    return re.sub(r'\s+',' ',''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)).strip()
def metrics(labels, probs, threshold):
    import numpy as np
    pred=probs>=threshold;positive=labels==1
    tp=int(np.sum(pred&positive));tn=int(np.sum(~pred&~positive));fp=int(np.sum(pred&~positive));fn=int(np.sum(~pred&positive))
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return {'samples':len(labels),'threshold':threshold,'accuracy':(tp+tn)/len(labels),
            'eot_precision':precision,'eot_recall':recall,'eot_f1':2*tp/max(1,2*tp+fp+fn),
            'false_eot':fp,'missed_eot':fn,'confusion_matrix_NEOT_EOT':[[tn,fp],[fn,tp]]}


def copy_model_object(folder, entry, target):
    source=folder/entry['object'];assert source.resolve().is_relative_to(folder.resolve())
    assert source.stat().st_size==entry['bytes'] and sha(source)==entry['sha256']
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
    assert sha(target)==entry['sha256']


def publish(config):
    root=Path(config['drive_path']);assert os.path.ismount('/content/drive') and read(root/'destination_identity.json')==config['identity']
    dest=root/'old_dev_c75_vi_v1';objects=dest/'objects';objects.mkdir(parents=True,exist_ok=True);entries={}
    paths=[JOB/x for x in ('dev.jsonl','old_qat_predictions.npz','config.json','worker.py','protocol.json','report.json','predictions.npz','predictions.jsonl.gz','review_samples.jsonl')]
    for path in paths:
        checksum=sha(path);target=objects/(checksum+'.bin');state('SAVING_EVALUATION_DIRECT_TO_DRIVE',file=path.name)
        if not target.exists():
            temp=target.with_suffix('.uploading')
            with path.open('rb') as src,temp.open('wb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2);dst.flush();os.fsync(dst.fileno())
            assert sha(temp)==checksum;temp.replace(target)
        assert sha(target)==checksum and target.stat().st_size==path.stat().st_size
        entries[path.name]={'sha256':checksum,'bytes':path.stat().st_size,'object':str(target.relative_to(dest)),'readback_verified':True}
    m={'at_utc':now(),'complete':True,'destination_identity':config['identity'],'files':entries,
       'frozen_model_reference':config['frozen_model_reference'],'scope':'OLD_DEV_TRANSFER_DIAGNOSTIC_NO_MODEL_OR_THRESHOLD_SELECTION'}
    raw=(json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode();temp=dest/'latest_manifest.tmp'
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    assert temp.read_bytes()==raw;temp.replace(dest/'latest_manifest.json');atomic(JOB/'drive_manifest.json',m)


def run():
    import fcntl
    with (JOB/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (JOB/'report.json').exists(),'Completed evaluation must not be rescored'
        c=read(JOB/'config.json');assert sha(JOB/'dev.jsonl')==c['dev_sha256'];assert sha(JOB/'old_qat_predictions.npz')==c['old_predictions_sha256']
        assert os.path.ismount('/content/drive');drive=Path(c['drive_path']);assert read(drive/'destination_identity.json')==c['identity']
        assert shutil.disk_usage('/content').free>27*1024**3
        parent=drive/'diagnostic_test_c75_v1';manifest=read(parent/'latest_manifest.json');assert manifest['complete'] and manifest['destination_identity']==c['identity']
        assert manifest['files']['report.json']['sha256']==c['parent_report_sha256']
        assert manifest['files']['frozen/vi/model.safetensors']['sha256']==c['weights_sha256']
        state('COPYING_FROZEN_C75_VI_FROM_DRIVE')
        for name in ('config.json','tokenizer.json','tokenizer_config.json','model.safetensors'):
            copy_model_object(parent,manifest['files']['frozen/vi/'+name],JOB/'model'/name)
        rows=[json.loads(x) for x in (JOB/'dev.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()];assert len(rows)==17833
        protocol={'material_passport':{'mode':'run','scope':'LEGACY_DEV_DIAGNOSTIC_NOT_NEW_BLIND_TEST','model':'frozen VI cycle75/100'},
            'data_sha256':c['dev_sha256'],'weights_sha256':c['weights_sha256'],'threshold_fixed_from_new_campaign_dev':c['threshold'],
            'old_qat_threshold_fixed':0.47,'input_mapping':'Preserve original context and text; new model native encode_tail(context,text,256). No future words invented or moved.',
            'input_mismatch':'Legacy rows have current-prefix text and no future words; this differs from training the new model on left text plus1/2 following words.',
            'context_nonempty':sum(bool(x.get('context','')) for x in rows),'samples':len(rows),
            'model_or_threshold_selection_on_this_evaluation':False,'holdout_reopened':False,'quality_target_certified':False,
            'direct_old_vs_new_score_is_not_isolated_compression_effect':True,'legacy_label_source_counts':dict(__import__('collections').Counter(x.get('label_source') for x in rows))}
        atomic(JOB/'protocol.json',protocol)
        import numpy as np
        import torch
        import transformers.utils as utils
        import transformers.utils.import_utils as imports
        utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
        utils.is_librosa_available=imports.is_librosa_available=lambda:False
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();torch.set_num_threads(2)
        old=np.load(JOB/'old_qat_predictions.npz');labels=np.asarray([int(x['label']=='EOT') for x in rows],dtype=np.int64)
        assert np.array_equal(old['labels'],labels) and np.array_equal(old['indices'],np.arange(len(rows)))
        tok=AutoTokenizer.from_pretrained(JOB/'model',local_files_only=True);encoded=[];input_hash=hashlib.sha256();truncated=0
        state('ENCODING_OLD_DEV_WITH_NEW_MODEL_CODEC',total=len(rows))
        for i,row in enumerate(rows):
            current=tok.encode(normalize(row['text']),add_special_tokens=False);truncated+=len(current)>253;current=current[-253:]
            budget=253-len(current);history=tok.encode(normalize(row.get('context','')),add_special_tokens=False)[-budget:] if budget else []
            ids=[tok.cls_token_id]+history+[tok.sep_token_id]+current+[tok.sep_token_id];encoded.append(ids)
            input_hash.update(np.asarray([len(ids),*ids],dtype='<i4').tobytes())
            if i%2000==0:state('ENCODING_OLD_DEV_WITH_NEW_MODEL_CODEC',encoded=i,total=len(rows))
        model=AutoModelForSequenceClassification.from_pretrained(JOB/'model',local_files_only=True,dtype=torch.float32).cuda().eval()
        probs=np.lib.format.open_memmap(JOB/'partial_probabilities.npy',mode='w+',dtype=np.float32,shape=(len(rows),));probs[:]=np.nan;probs.flush();started=time.monotonic()
        for start in range(0,len(rows),48):
            if time.monotonic()-started>1800:raise RuntimeError('Evaluation exceeded30minute safety budget; partial evidence preserved')
            subset=encoded[start:start+48];width=max(map(len,subset));x=torch.full((len(subset),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x)
            for k,ids in enumerate(subset):x[k,:len(ids)]=torch.as_tensor(ids,device='cuda');mask[k,:len(ids)]=1
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model(input_ids=x,attention_mask=mask).logits.float().softmax(-1)[:,1].cpu().numpy()
            probs[start:start+len(subset)]=p;probs.flush();state('SCORING_OLD_DEV',completed=start+len(subset),total=len(rows),elapsed_seconds=time.monotonic()-started)
            del x,mask,p
        assert np.isfinite(probs).all();del model;torch.cuda.empty_cache()
        new_metrics=metrics(labels,probs,c['threshold']);old_metrics=metrics(labels,old['prob_EOT'],0.47)
        assert abs(old_metrics['accuracy']-0.9354567375091123)<1e-12
        new_correct=(probs>=c['threshold'])==labels;old_correct=(old['prob_EOT']>=0.47)==labels
        np.savez_compressed(JOB/'predictions.npz',indices=np.arange(len(rows)),labels=labels,prob_EOT=probs)
        selected=[]
        with gzip.open(JOB/'predictions.jsonl.gz','wt',encoding='utf-8') as f:
            for i,row in enumerate(rows):
                out={**row,'index':i,'model':'VI_C75','prediction':'EOT' if probs[i]>=c['threshold'] else 'NEOT','p_eot':float(probs[i]),'correct':bool(new_correct[i]),'threshold':c['threshold']}
                f.write(json.dumps(out,ensure_ascii=False)+'\n')
                if len(selected)<24:selected.append(out)
        (JOB/'review_samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected),encoding='utf-8')
        report={'complete':True,'completed_at_utc':now(),'scope':'LEGACY_DEV_TRANSFER_DIAGNOSTIC_NOT_NEW_BLIND_TEST','protocol':protocol,
            'new_model_metrics':new_metrics,'old_small_model_metrics':old_metrics,'model_checkpoint_cycle':75,
            'new_encoded_input_sha256':input_hash.hexdigest(),'new_current_text_truncated_rows':truncated,
            'paired_correctness':{'both_correct':int(np.sum(new_correct&old_correct)),'both_wrong':int(np.sum(~new_correct&~old_correct)),
                'new_only_correct':int(np.sum(new_correct&~old_correct)),'old_only_correct':int(np.sum(~new_correct&old_correct))},
            'full_predictions_sha256':sha(JOB/'predictions.jsonl.gz'),'prediction_arrays_sha256':sha(JOB/'predictions.npz'),
            'runtime_versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','numpy')},
            'inference_device':'Colab NVIDIA L4','threshold_not_retuned':True,'model_not_updated':True,'human_conversational_gold_certified':False}
        atomic(JOB/'report.json',report);publish(c);state('COMPLETE',report_sha256=sha(JOB/'report.json'),drive_readback_verified=True,samples=len(rows))


if __name__=='__main__':
    try:run()
    except Exception as exc:
        atomic(JOB/'error.json',{'at_utc':now(),'error':repr(exc),'traceback':traceback.format_exc()});state('FAILED_PRESERVED',error=repr(exc));raise
