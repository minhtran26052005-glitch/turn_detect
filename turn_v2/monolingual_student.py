"""Reuse the tested GRU distillation/QAT kernels for a single-language job."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import gc

import numpy as np
import torch
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available=hf_imports.is_torchvision_available=lambda:False
hf_utils.is_librosa_available=hf_imports.is_librosa_available=lambda:False
from .common import LABEL2ID,file_hash,read_jsonl,write_json
from .metrics import classification_metrics,probabilities
from .monolingual_data import encode_tail


def run(args):
    manifest=json.loads((args.data/'manifest.json').read_text());assert manifest['language']==args.language
    teacher=json.loads((args.teacher/'training_report.json').read_text());assert teacher['language']==args.language
    rows=list(read_jsonl(args.data/'dev.jsonl'));assert {r['language'] for r in rows}=={args.language}
    from . import trilingual_student as kernels
    # Dependency injection is confined to this worker process. Prior scripts,
    # source files and existing models are unchanged. Both training and export
    # use the same current-tail codec, unlike the older prefix codec.
    kernels.encode_pair=encode_tail
    kernels.run(SimpleNamespace(data=args.data,teacher=args.teacher,output=args.output))
    del kernels;gc.collect();torch.cuda.empty_cache()
    folder=args.output/'qat_best';contract=json.loads((folder/'turn_detector_config.json').read_text())
    contract.update(language=args.language,threshold=contract['language_thresholds'][args.language],current_truncation='tail',
        training_method='MONOLINGUAL_DISTILL_QAT',quality_scope='Same-language dev only; no unseen real quality certificate')
    contract.pop('language_thresholds');write_json(folder/'turn_detector_config.json',contract)
    from . import export_qat_student as exporter
    exporter.encode_pair=encode_tail;exporter.export(SimpleNamespace(student=folder,output=folder/'onnx_int4',dev=args.data/'dev.jsonl'))
    from transformers import AutoTokenizer
    import onnxruntime as ort
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True);options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.disable_quant_qdq','1');session=ort.InferenceSession(str(folder/'onnx_int4/model.onnx'),options,providers=['CPUExecutionProvider'])
    encoded=[encode_tail(tok,r.get('context',''),r['text'])['input_ids'] for r in rows];probs=np.zeros(len(rows))
    for start in range(0,len(rows),64):
        batch=encoded[start:start+64];width=max(map(len,batch));ids=np.zeros((len(batch),width),np.int64);mask=np.zeros_like(ids)
        for i,value in enumerate(batch):ids[i,:len(value)]=value;mask[i,:len(value)]=1
        probs[start:start+len(batch)]=probabilities(session.run(['logits'],{'input_ids':ids,'attention_mask':mask})[0])
    labels=np.array([LABEL2ID[r['label']] for r in rows]);threshold=contract['threshold'];overall=classification_metrics(labels,probs,threshold)
    slices={}
    for name in sorted({r['evaluation_slice'] for r in rows}):
        index=np.array([r['evaluation_slice']==name for r in rows]);slices[name]=classification_metrics(labels[index],probs[index],threshold)
    fake=np.load(folder/'dev_predictions.npz');assert np.array_equal(fake['labels'],labels)
    delta=float(np.max(np.abs(fake['prob_EOT']-probs)));assert delta<1e-4
    np.savez_compressed(args.output/'packed_dev_predictions.npz',labels=labels,prob_EOT=probs)
    result={'status':'MONOLINGUAL_DISTILL_QAT_EXPORTED','language':args.language,'overall_dev':overall,'dev_slices':slices,
        'natural_dev_target_met':slices['natural']['joint_score']>=.9,'unseen_real_target_certified':False,'test_evaluated':False,
        'onnx_path':str(folder/'onnx_int4'),'model_bytes':(folder/'onnx_int4/model.onnx').stat().st_size,'model_sha256':file_hash(folder/'onnx_int4/model.onnx'),
        'packed_vs_fake_quant_max_probability_delta':delta,'teacher_weights_sha256':teacher['checkpoint_sha256'],
        'data_manifest_sha256':file_hash(args.data/'manifest.json'),'source_sha256':file_hash(Path(__file__))}
    write_json(args.output/'monolingual_report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['en','zh'],required=True)
    for name in ['data','teacher','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
