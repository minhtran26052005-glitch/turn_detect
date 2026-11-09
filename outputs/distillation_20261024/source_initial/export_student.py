"""Export tiny students and verify every storage precision with real ONNX inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from .common import encode_pair, file_hash, read_jsonl, write_json
from .distill_teacher import padded
from .metrics import probabilities
from .tiny_student import load_student


def export(args):
    import onnx
    import onnxruntime as ort
    from onnx import helper,numpy_helper,TensorProto
    from transformers import AutoTokenizer
    from .compress import sort_graph_with_captures
    model=load_student(args.student).float().eval()
    tok=AutoTokenizer.from_pretrained(args.student)
    rows=list(read_jsonl(args.dev));rows=rows[:8]
    arrays=padded([encode_pair(tok,r.get('context',''),r['text'],256) for r in rows[:2]],tok.pad_token_id)
    fp32=args.student/'onnx_fp32';fp32.mkdir(exist_ok=False)
    with torch.no_grad():
        torch.onnx.export(model,(torch.from_numpy(arrays['input_ids']),torch.from_numpy(arrays['attention_mask'])),str(fp32/'model.onnx'),input_names=['input_ids','attention_mask'],output_names=['logits'],dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},opset_version=18,dynamo=False,external_data=False)
    for name in ['config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json']:shutil.copy2(args.student/name,fp32/name)
    contract=json.loads((fp32/'turn_detector_config.json').read_text());contract['onnx_session_options']={'session.disable_quant_qdq':'1'}
    contract['storage_precision']='FP32';write_json(fp32/'turn_detector_config.json',contract)
    for precision in ['INT8','FP16']:
        folder=args.student/('onnx_'+precision.lower());folder.mkdir(exist_ok=False)
        for path in fp32.glob('*.json'):shutil.copy2(path,folder/path.name)
        graph=onnx.load(str(fp32/'model.onnx'));new_weights=[];nodes=[];count=0
        matmul_inputs={n.input[1] for n in graph.graph.node if n.op_type=='MatMul'}
        for tensor in graph.graph.initializer:
            # Keep biases and scalar/vector shape data in their original format.
            if tensor.data_type!=TensorProto.FLOAT or len(tensor.dims)<2 or (len(tensor.dims)==2 and tensor.dims[0]==1):new_weights.append(tensor);continue
            values=numpy_helper.to_array(tensor);prefix=tensor.name+'_stored_'+precision
            if precision=='FP16':
                new_weights.append(numpy_helper.from_array(values.astype(np.float16),prefix))
                nodes.append(helper.make_node('Cast',[prefix],[tensor.name],to=TensorProto.FLOAT))
            else:
                axis=1 if len(values.shape)==3 or tensor.name in matmul_inputs else 0
                reduction=tuple(i for i in range(values.ndim) if i!=axis)
                scale=np.maximum(np.max(np.abs(values),axis=reduction)/127,np.finfo(np.float32).tiny).astype(np.float32)
                shape=[1]*values.ndim;shape[axis]=len(scale)
                quant=np.clip(np.rint(values/scale.reshape(shape)),-127,127).astype(np.int8)
                new_weights.extend([numpy_helper.from_array(quant,prefix),numpy_helper.from_array(scale,prefix+'_scale'),numpy_helper.from_array(np.zeros_like(scale,dtype=np.int8),prefix+'_zero')])
                nodes.append(helper.make_node('DequantizeLinear',[prefix,prefix+'_scale',prefix+'_zero'],[tensor.name],axis=axis))
            count+=1
        del graph.graph.initializer[:];graph.graph.initializer.extend(new_weights)
        old=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(nodes+old)
        sort_graph_with_captures(graph.graph);onnx.save_model(graph,str(folder/'model.onnx'))
        c=copy_contract=dict(contract);c['storage_precision']=precision;c['quantized']=precision=='INT8';c['stored_tensor_count']=count
        write_json(folder/'turn_detector_config.json',c)
    results=[]
    for name in ['onnx_fp32','onnx_fp16','onnx_int8']:
        folder=args.student/name;onnx.checker.check_model(str(folder/'model.onnx'))
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1;options.add_session_config_entry('session.disable_quant_qdq','1')
        session=ort.InferenceSession(str(folder/'model.onnx'),options,providers=['CPUExecutionProvider'])
        delta=[];same=True
        with torch.no_grad():
            for batch_size in [1,3,8]:
                arr=padded([encode_pair(tok,r.get('context',''),r['text'],256) for r in rows[:batch_size]],tok.pad_token_id)
                expected=model(**{k:torch.from_numpy(v) for k,v in arr.items()}).numpy()
                actual=session.run(['logits'],arr)[0]
                delta.append(float(np.abs(probabilities(expected)-probabilities(actual)).max()))
                same &= bool(np.array_equal(expected.argmax(-1),actual.argmax(-1)))
            # Extra right padding must preserve each model's output.
            arr=padded([encode_pair(tok,r.get('context',''),r['text'],256) for r in rows[:3]],tok.pad_token_id)
            before=session.run(['logits'],arr)[0]
            longer={k:np.pad(v,((0,0),(0,32))) for k,v in arr.items()}
            after=session.run(['logits'],longer)[0]
        padding_delta=float(np.max(np.abs(probabilities(before)-probabilities(after))))
        passed=(name!='onnx_fp32' or max(delta)<1e-4) and padding_delta<1e-5
        report={'path':str(folder),'model_bytes':(folder/'model.onnx').stat().st_size,'deployment_bytes':sum(p.stat().st_size for p in folder.iterdir() if p.is_file()),'model_sha256':file_hash(folder/'model.onnx'),'max_probability_delta_vs_pytorch_fp32':max(delta),'argmax_same_on_8_examples':same,'padding_delta':padding_delta,'technical_parity_passed':passed,'note':'Reduced storage requires full dev evaluation; small parity sample is not quality evidence'}
        write_json(args.student/(name+'_export.json'),report);results.append(report)
        if not passed:raise ValueError('Student ONNX technical parity failed')
    write_json(args.student/'export_report.json',results);print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--student',type=Path,required=True);p.add_argument('--dev',type=Path,default=Path('data_v2/frozen/dev.jsonl'));export(p.parse_args())
