"""Evaluate selected local delta on frozen held material and fresh VI pilot."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

from turn_v2.stream_v4_local import AutoModelForSequenceClassification,AutoTokenizer
from turn_v2.stream_v4_train import encode_rows,infer
from turn_v2.common import LABEL2ID,file_hash,read_jsonl,write_json,write_jsonl
from turn_v2.metrics import classification_metrics


def metrics(rows,probs,thresholds):
    labels=np.asarray([LABEL2ID[r['label']] for r in rows]);langs=np.asarray([r['language'] for r in rows])
    return {lang:classification_metrics(labels[langs==lang],probs[langs==lang],thresholds[lang]) for lang in sorted(set(langs))}


def main():
    root=Path('outputs/stream_v4_20261025');run=root/'local_recovery_r2'
    output=root/'local_final';output.mkdir(exist_ok=False)
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    identity=json.loads((run/'identity.json').read_text(encoding='utf-8'))
    checkpoint=Path(report['best_path']);selected=json.loads((checkpoint.parent/'best_dev.json').read_text(encoding='utf-8'))
    thresholds=selected['thresholds']
    assert file_hash(checkpoint)==report['checkpoint_sha256']
    base=Path(identity['base_path']);assert file_hash(base/'model.safetensors')==identity['base_sha256']
    write_json(output/'frozen_before_test.json',{'base_sha256':identity['base_sha256'],'delta_sha256':report['checkpoint_sha256'],
        'thresholds':thresholds,'selected_by':'minimum development slice accuracy/F1; NO held/fresh test scores used',
        'fresh_vi_label_status':'AUTOMATIC_VAD_FLOOR_PROXY_NOT_HUMAN_GOLD','colab_state':'UNKNOWN_AFTER_404'})
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForSequenceClassification.from_pretrained(base,local_files_only=True,dtype=torch.float32).cuda().eval()
    model.load_state_dict(load_file(str(checkpoint)),strict=False)
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    results={}
    for name,path,language in [('dev',Path('data_v4/causal_20261025/dev.jsonl'),None),
        ('sealed_read_speech',Path('data_v4/causal_20261025/sealed.jsonl'),None),
        ('vi_full_legacy_dev',Path('data_v2/frozen/dev.jsonl'),'vi'),
        ('fresh_vi_natural_auto_labels',Path('data_v4/vi_natural_pause_20261025/cases.jsonl'),None)]:
        assert path.exists(),str(path)
        rows=list(read_jsonl(path))
        if language:rows=[{**r,'language':language} for r in rows]
        ids,lengths=encode_rows(rows,tokenizer);probs=infer(model,ids,lengths,batch_size=8)
        results[name]=metrics(rows,probs,thresholds)
        np.savez_compressed(output/(name+'_predictions.npz'),labels=np.array([LABEL2ID[r['label']] for r in rows]),prob_EOT=probs,language=np.array([r['language'] for r in rows]))
        print(json.dumps({'scope':name,'metrics':results[name]}),flush=True)
    fresh=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')))
    ids,lengths=encode_rows(fresh,tokenizer);probs=infer(model,ids,lengths,batch_size=8)
    write_jsonl(output/'fresh_crawl_predictions.jsonl',[{**r,'prob_EOT':float(p),'prediction':'EOT' if p>=thresholds[r['language']] else 'NEOT',
        'threshold':thresholds[r['language']],'label':None,'label_status':'PENDING_INDEPENDENT_REVIEW'} for r,p in zip(fresh,probs,strict=True)])
    # A fixed VI-control comparison on exactly the same fresh natural pilot.
    path=Path('models/turn_v2_distilled_qat4_20261024/onnx/run_model.py')
    spec=importlib.util.spec_from_file_location('vi_control',path);module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    control=module.TinyOnnxDetector(path.parent)
    vi=list(read_jsonl(Path('data_v4/vi_natural_pause_20261025/cases.jsonl')))
    control_probs=np.array([control.predict(r['text'],r.get('context',''))['prob_EOT'] for r in vi])
    results['vi_INT4_control_on_same_fresh_natural_auto_labels']=metrics(vi,control_probs,{'vi':control.contract['threshold']})
    np.savez_compressed(output/'vi_control_fresh_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in vi]),prob_EOT=control_probs)
    result={'status':'ROUNDS_AND_TESTS_COMPLETED_NOT_90_REAL_QUALIFIED','metrics':results,'thresholds':thresholds,
        'base_sha256':identity['base_sha256'],'delta_sha256':report['checkpoint_sha256'],
        'fresh_crawl_cases':len(fresh),'fresh_crawl_gold_labels':0,'fresh_crawl_accuracy':None,
        'human_verified_vi_real_test_available':False,'all_real_languages_over_90':False,
        'scope':'Development/held read speech plus SMALL automatic Vietnamese stereo-floor pilot; no human-gold real3language certification',
        'old_holdout_evaluated':False,'colab_job_state_inferred':False,'accepted_serving_model_changed':False}
    write_json(output/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
