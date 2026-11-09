"""Restore the frozen compact teacher exactly, verify it, cache train targets only."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .common import LABEL2ID, encode_pair, file_hash, read_jsonl, write_json
from .metrics import probabilities


def restore_teacher(path: Path):
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(str(path/'model.onnx'))
    arrays = {t.name:numpy_helper.to_array(t) for t in graph.graph.initializer}
    producers = {name:n for n in graph.graph.node for name in n.output}
    def resolve(name):
        if name in arrays:
            value = arrays[name]
            if name=='classifier.deberta.embeddings.word_embeddings.weight' and value.dtype==np.int8:
                return value.astype(np.float32)*arrays['compact_word_embedding_scale'][:,None]
            return value
        node = producers[name]
        if node.op_type=='DequantizeLinear':
            axis=next((a.i for a in node.attribute if a.name=='axis'),1)
            value,scale=resolve(node.input[0]),resolve(node.input[1])
            shape=[1]*value.ndim;shape[axis]=len(scale)
            zero=resolve(node.input[2]) if len(node.input)>2 else np.zeros_like(scale)
            return (value.astype(np.float32)-zero.reshape(shape))*scale.reshape(shape)
        if node.op_type=='Identity':return resolve(node.input[0])
        if node.op_type=='Transpose':
            perm=next((list(a.ints) for a in node.attribute if a.name=='perm'),None)
            return resolve(node.input[0]).transpose(perm)
        raise ValueError(f'Cannot resolve weight {name}: {node.op_type}')
    model=AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(path)).float()
    expected=model.state_dict();state={}
    for name in expected:
        original='classifier.'+name
        if original in arrays or original in producers:
            state[name]=torch.from_numpy(np.array(resolve(original),copy=True))
    for node in graph.graph.node:
        prefix=node.name.rsplit('/',1)[0].strip('/').removeprefix('classifier/').replace('/','.')
        if node.op_type=='MatMul' and prefix+'.weight' in expected:
            state[prefix+'.weight']=torch.from_numpy(np.array(resolve(node.input[1]).T,copy=True))
        if node.op_type=='LayerNormalization':
            for suffix, operand in [('weight',1),('bias',2)]:
                key=prefix+'.'+suffix
                if key in expected:
                    state[key]=torch.from_numpy(np.array(resolve(node.input[operand]),copy=True))
    missing=sorted(set(expected)-set(state))
    if missing:raise ValueError(f'Teacher reconstruction incomplete: {missing}')
    for name,value in state.items():
        if value.shape!=expected[name].shape:raise ValueError(f'Shape mismatch {name}')
    model.load_state_dict(state,strict=True)
    del graph,arrays,state,expected
    gc.collect()
    return model.eval()


def padded(encoded,pad=0,multiple=8):
    length=min(256,((max(len(e['input_ids']) for e in encoded)+multiple-1)//multiple)*multiple)
    return {key:np.asarray([e[key]+[pad if key=='input_ids' else 0]*(length-len(e[key])) for e in encoded],np.int64)
            for key in ('input_ids','attention_mask')}


def verify(model,path,rows,device):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.disable_quant_qdq','1')
    session=ort.InferenceSession(str(path/'model.onnx'),options,providers=['CPUExecutionProvider'])
    tok=AutoTokenizer.from_pretrained(path)
    errors=[];same=True
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode():
        for row in rows:
            arr=padded([encode_pair(tok,row.get('context',''),row['text'],256)],tok.pad_token_id)
            left=session.run(['logits'],arr)[0]
            right=model(**{k:torch.from_numpy(v).to(device) for k,v in arr.items()}).logits.float().cpu().numpy()
            errors.append(float(np.abs(probabilities(left)-probabilities(right)).max()))
            same &= bool(np.array_equal(left.argmax(-1),right.argmax(-1)))
    result={'samples':len(rows),'max_probability_delta':max(errors),'argmax_same':same,
            'passed':max(errors)<1e-4 and same,'teacher_sha256':file_hash(path/'model.onnx')}
    if not result['passed']:raise ValueError('Reconstructed teacher parity failed: '+json.dumps(result))
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.data/'manifest.json').read_text())
    assert file_hash(args.data/'train.jsonl')==manifest['split_sha256']['train']
    assert file_hash(args.data/'dev.jsonl')==manifest['split_sha256']['dev']
    marker=file_hash(args.data/'holdout_consumed.json')
    torch.set_num_threads(2);torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    teacher=restore_teacher(args.teacher).to(device)
    dev_rows=list(read_jsonl(args.data/'dev.jsonl'))
    parity=verify(teacher,args.teacher,dev_rows[:16],device)
    write_json(args.output/'teacher_parity.json',parity)
    print(json.dumps({'stage':'teacher_parity',**parity}),flush=True)
    tok=AutoTokenizer.from_pretrained(args.teacher)
    # Teacher is fixed; these targets never include dev or consumed holdout rows.
    rows=list(read_jsonl(args.data/'train.jsonl'));n=len(rows)
    ids=np.lib.format.open_memmap(args.output/'train_ids.npy',mode='w+',dtype=np.int32,shape=(n,256))
    ids[:]=tok.pad_token_id
    lengths=np.zeros(n,np.int16);labels=np.zeros(n,np.int64)
    started=time.monotonic()
    for i,row in enumerate(rows):
        enc=encode_pair(tok,row.get('context',''),row['text'],256)['input_ids']
        ids[i,:len(enc)]=enc;lengths[i]=len(enc);labels[i]=LABEL2ID[row['label']]
    ids.flush();np.save(args.output/'train_lengths.npy',lengths);np.save(args.output/'train_labels.npy',labels)
    logits=np.lib.format.open_memmap(args.output/'teacher_logits.npy',mode='w+',dtype=np.float32,shape=(n,2))
    hidden=np.lib.format.open_memmap(args.output/'teacher_hidden.npy',mode='w+',dtype=np.float16,shape=(n,teacher.config.hidden_size))
    captured=[]
    hook=teacher.pooler.register_forward_hook(lambda module, inputs, output:captured.append(output.detach()))
    order=np.argsort(lengths,kind='stable')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    with torch.inference_mode():
        for start in range(0,n,args.batch_size):
            index=order[start:start+args.batch_size];length=int(lengths[index].max());length=min(256,((length+7)//8)*8)
            batch=torch.from_numpy(np.asarray(ids[index,:length],dtype=np.int64)).to(device)
            mask=torch.arange(length,device=device)[None,:]<torch.from_numpy(lengths[index].astype(np.int64)).to(device)[:,None]
            captured.clear()
            result=teacher(input_ids=batch,attention_mask=mask.long()).logits.float()
            if not torch.isfinite(result).all():raise ValueError('Teacher cache has non-finite logits')
            logits[index]=result.cpu().numpy();hidden[index]=captured[-1].float().cpu().numpy().astype(np.float16)
            if start//args.batch_size%50==0 or start+args.batch_size>=n:
                progress={'stage':'teacher_targets','samples':min(start+args.batch_size,n),'total':n,'elapsed_seconds':time.monotonic()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    hook.remove();logits.flush();hidden.flush()
    torch.save(teacher.deberta.embeddings.word_embeddings.weight.detach().cpu(),args.output/'teacher_embeddings.pt')
    report={'status':'TRAIN_TARGETS_COMPLETE','train_samples':n,'teacher_sha256':file_hash(args.teacher/'model.onnx'),
        'train_sha256':file_hash(args.data/'train.jsonl'),'dev_sha256':file_hash(args.data/'dev.jsonl'),
        'holdout_marker_sha256':marker,'holdout_marker_unchanged':file_hash(args.data/'holdout_consumed.json')==marker,
        'holdout_evaluated':False,'teacher_tf32_for_train_targets':True,'parity':parity,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in args.output.glob('*.npy')},
        'elapsed_seconds':time.monotonic()-started}
    write_json(args.output/'cache_report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data_v2/frozen'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=128)
    run(parser.parse_args())
