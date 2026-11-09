"""Frozen final diagnostic on public observed-floor cases, no test retuning."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .common import LABEL2ID,file_hash,read_jsonl,write_json,write_jsonl
from .metrics import classification_metrics
from .natural_v6_finetune import encode_rows
from .stream_v4_train import infer


def metrics(rows,probs,thresholds):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);langs=np.array([r['language'] for r in rows])
    return {lang:classification_metrics(labels[langs==lang],probs[langs==lang],thresholds[lang]) for lang in ['vi','en','zh']}


def qwen_predictions(candidate,rows):
    from .natural_v8_lora_r2 import disable_unused_torchao
    disable_unused_torchao()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from .natural_v6_features import visible_pair,SYSTEM
    from .natural_v8_lora import predict
    receipt=json.loads(Path('outputs/natural_v5_20261025/qwen_download.json').read_text())
    small=json.loads(Path('outputs/natural_v6_20261025/encoder_download.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(receipt['local_path'],local_files_only=True);tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    compact=AutoTokenizer.from_pretrained(small['local_path'],local_files_only=True)
    baseline_tokenizer=AutoTokenizer.from_pretrained('runs/stream_v4_campaign_20261025/round_02/best',local_files_only=True)
    ids=[tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},
        {'role':'user','content':json.dumps(visible_pair(r,baseline_tokenizer,compact),ensure_ascii=False)}],tokenize=True,add_generation_prompt=True,return_dict=False) for r in rows]
    assert all(isinstance(tokens,list) and all(isinstance(value,int) for value in tokens) for tokens in ids)
    base=AutoModelForCausalLM.from_pretrained(receipt['local_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,candidate['path'],local_files_only=True).eval()
    a=tokenizer.encode('A',add_special_tokens=False);b=tokenizer.encode('B',add_special_tokens=False);assert len(a)==len(b)==1
    weights=model.get_base_model().lm_head.weight[[b[0],a[0]]].detach().float()
    probabilities=predict(model,ids,tokenizer,weights)
    del model,base;gc.collect();torch.cuda.empty_cache()
    return probabilities


def main():
    root=Path('outputs/natural_v7_20261025');data=Path('data_v6/pho_text_20261025');output=root/'final_eval'
    assert not (data/'sealed_consumed.json').exists(),'Held cohort already seen: use existing predictions as regression'
    selected= json.loads((root/'selected_before_held.json').read_text());candidate=selected['selected']
    assert selected['status']=='FROZEN_AFTER_DEV_BEFORE_HELD' and not selected['held_classifier_scored']
    model_file=Path(candidate['path'])/('adapter_model.safetensors' if candidate['kind']=='qwen_lora' else 'model.safetensors')
    expected=candidate['adapter_sha256'] if candidate['kind']=='qwen_lora' else candidate['weights_sha256']
    assert file_hash(model_file)==expected
    manifest=json.loads((data/'manifest.json').read_text());assert file_hash(data/'sealed.jsonl')==manifest['files']['sealed']['sha256']
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    initial=json.loads(Path('runs/natural_v7_text_20261025/initial_dev.json').read_text())
    controls=[next(c for c in selected['candidates'] if c['kind']=='mdeberta' and c['path']==r['baseline_path']) for r in initial['candidates']]
    freeze={'selected':candidate,'selection_sha256':file_hash(root/'selected_before_held.json'),'sealed_sha256':file_hash(data/'sealed.jsonl'),
        'data_manifest_sha256':file_hash(data/'manifest.json'),'controls':controls,'test_purpose':'Final public-data diagnostic after frozen dev choice; do not tune on these scores',
        'labels':'Automatic stereo observed-floor references; not human gold','source_sha256':file_hash(Path(__file__))}
    write_json(output/'frozen_before_test.json',freeze)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=list(read_jsonl(data/'sealed.jsonl'));crawl=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')))
    outcomes={};cache={};selected_probs=None;selected_crawl=None
    for name,chosen in [('selected',candidate),('V4_control',controls[0]),('V5_control',controls[1])]:
        fingerprint=(chosen['path'],json.dumps(chosen['dev']['thresholds'],sort_keys=True))
        if fingerprint in cache:probs=cache[fingerprint]
        elif chosen['kind']=='mdeberta':
            model=AutoModelForSequenceClassification.from_pretrained(chosen['path'],local_files_only=True,dtype=torch.float32).cuda().eval()
            tokenizer=AutoTokenizer.from_pretrained(chosen['path'],local_files_only=True)
            ids,lengths=encode_rows(rows,tokenizer);probs=infer(model,ids,lengths,batch_size=32)
            if name=='selected':
                ids,lengths=encode_rows(crawl,tokenizer);selected_crawl=infer(model,ids,lengths,batch_size=32)
            del model;gc.collect();torch.cuda.empty_cache()
            cache[fingerprint]=probs
        else:
            combined=qwen_predictions(chosen,rows+crawl);probs=combined[:len(rows)];selected_crawl=combined[len(rows):];cache[fingerprint]=probs
        if name=='selected':selected_probs=probs
        result=metrics(rows,probs,chosen['dev']['thresholds']);outcomes[name]=result
        np.savez_compressed(output/(name+'_predictions.npz'),labels=np.array([LABEL2ID[r['label']] for r in rows]),
            prob_EOT=probs,language=np.array([r['language'] for r in rows]),case_id=np.array([r['case_id'] for r in rows]))
        print(json.dumps({'scope':'NEW_HELD_NATURAL_PUBLIC','model':name,'metrics':result}),flush=True)
    thresholds=candidate['dev']['thresholds']
    write_jsonl(output/'held_predictions.jsonl',[{**r,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[r['language']] else 'NEOT',
        'threshold':thresholds[r['language']]} for r,p in zip(rows,selected_probs,strict=True)])
    write_jsonl(output/'crawl_predictions.jsonl',[{**r,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[r['language']] else 'NEOT',
        'threshold':thresholds[r['language']],'label':None,'is_gold':False} for r,p in zip(crawl,selected_crawl,strict=True)])
    call_counts={lang:len({r['session_group'] for r in rows if r['language']==lang}) for lang in ['vi','en','zh']}
    passed=all(m['accuracy']>.9 and m['eot_f1']>.9 for m in outcomes['selected'].values())
    report={'status':'FINAL_PUBLIC_HELD_DIAGNOSTIC_COMPLETE','metrics':outcomes,'selected_kind':candidate['kind'],'selected_path':candidate['path'],
        'selected_identity':expected,'thresholds':thresholds,'held_cases':len(rows),'held_calls':call_counts,
        'automatic_held_numeric_target_met':passed,'human_gold_real_quality_certified':False,'source_reference':'Automatic stereo/VAD observed floor',
        'confidence_scope':'Point estimates only; VI has one held call; not a representative quality certificate',
        'crawl_cases':len(crawl),'crawl_gold_labels':0,'crawl_accuracy':None,'crawl_eot_f1':None,
        'old_legacy_holdout_evaluated':False,'serving_default_changed':False,'elapsed_seconds':time.monotonic()-started}
    write_json(output/'report.json',report)
    write_json(data/'sealed_consumed.json',{'status':'SEEN_NOW_REGRESSION_ONLY','sealed_sha256':file_hash(data/'sealed.jsonl'),
        'frozen_selection_sha256':file_hash(root/'selected_before_held.json'),'report_sha256':file_hash(output/'report.json'),'no_threshold_or_weight_retuning':True})
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
