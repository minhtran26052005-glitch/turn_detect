"""CPU exports of the trained small mALBERT, including real packed INT4."""
import gc
import json
from pathlib import Path
import shutil
import numpy as np


def export(model, directory, tokenizer_directory, rows, tokenizer, bits=32, quant=None):
    import torch
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from .tiny_compression_v1.common import encode_pair, file_hash
    from .tiny_compression_v1.compress import sort_graph_with_captures
    from .tiny_compression_v1.qat_student import pack_parameter
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    parameters = model.parameter_count(); assert parameters < 20_000_000
    examples = [next(r for r in rows if r['language'] == l) for l in ('vi', 'en', 'zh')]
    sequences = [encode_pair(tokenizer, r['context'], r['text'], 256)['input_ids'] for r in examples]
    width = max(map(len, sequences)); ids = torch.zeros((3, width), dtype=torch.long); mask = torch.zeros_like(ids)
    for i, s in enumerate(sequences): ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
    with torch.inference_mode():
        torch.onnx.export(model, (ids, mask), str(directory/'unpacked_control.onnx'),
            input_names=['input_ids','attention_mask'], output_names=['logits'],
            dynamic_axes={'input_ids':{0:'batch',1:'sequence'},'attention_mask':{0:'batch',1:'sequence'},'logits':{0:'batch'}},
            opset_version=18, dynamo=False, external_data=False, do_constant_folding=False)
    graph = onnx.load(str(directory/'unpacked_control.onnx')); converted = []
    if bits != 32:
        graph = onnx.version_converter.convert_version(graph, 21)
        tensors = []; nodes = []; group = quant['group_size'] if quant else 64
        for t in graph.graph.initializer:
            if t.data_type != TensorProto.FLOAT or len(t.dims) < 2:
                tensors.append(t); continue
            a = numpy_helper.to_array(t).copy(); parameter = model.get_parameter(t.name)
            assert tuple(parameter.shape) == tuple(a.shape)
            if bits == 4:
                assert quant is not None and t.name in quant['scales']
                codes, scale, restored = pack_parameter(a, quant['scales'][t.name].numpy(), group)
            else:
                blocks = (a.shape[-1]+group-1)//group
                z = np.pad(a,[(0,0)]*(a.ndim-1)+[(0,blocks*group-a.shape[-1])]).reshape(*a.shape[:-1],blocks,group)
                scale = np.maximum(np.maximum(z.max(-1)/127,z.min(-1)/-128),np.finfo(np.float16).tiny).astype(np.float16)
                expanded = np.repeat(scale.astype(np.float32),group,-1)[...,:a.shape[-1]]
                codes = np.clip(np.rint(a/expanded),-128,127).astype(np.int8)
                restored = (codes.astype(np.float32)*expanded).astype(np.float16).astype(np.float32)
            prefix = t.name+'_packed'
            if bits == 4:
                flat = codes.reshape(-1).astype(np.uint8)&15
                if flat.size%2: flat = np.pad(flat,(0,1))
                packed = flat[::2] | (flat[1::2]<<4)
                tensors.append(TensorProto(name=prefix,data_type=TensorProto.INT4,dims=a.shape,raw_data=packed.tobytes()))
                decoded = np.empty(flat.size,np.int8); decoded[::2]=packed&15; decoded[1::2]=packed>>4; decoded[decoded>=8]-=16
                assert np.array_equal(decoded[:codes.size],codes.reshape(-1))
            else: tensors.append(numpy_helper.from_array(codes,prefix))
            tensors.append(numpy_helper.from_array(scale,prefix+'_scale'))
            nodes.extend([helper.make_node('DequantizeLinear',[prefix,prefix+'_scale'],[prefix+'_half'],axis=a.ndim-1,block_size=group),
                          helper.make_node('Cast',[prefix+'_half'],[t.name],to=TensorProto.FLOAT)])
            with torch.no_grad(): parameter.copy_(torch.from_numpy(restored))
            converted.append(dict(name=t.name,shape=list(a.shape),bits=bits))
        if quant: assert {x['name'] for x in converted} == set(quant['scales'])
        original=list(graph.graph.node);del graph.graph.node[:];graph.graph.node.extend(nodes+original)
        del graph.graph.initializer[:];graph.graph.initializer.extend(tensors);sort_graph_with_captures(graph.graph)
    onnx.checker.check_model(graph,full_check=True);onnx.save(graph,str(directory/'model.onnx'))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.disable_quant_qdq','1')
    session=ort.InferenceSession(str(directory/'model.onnx'),options,providers=['CPUExecutionProvider'])
    with torch.inference_mode(): reference=model(ids,mask).numpy()
    actual=session.run(['logits'],{'input_ids':ids.numpy(),'attention_mask':mask.numpy()})[0]
    padded=session.run(['logits'],{'input_ids':np.pad(ids.numpy(),((0,0),(0,8))),'attention_mask':np.pad(mask.numpy(),((0,0),(0,8)))})[0]
    delta=float(np.max(np.abs(reference-actual))); padding_delta=float(np.max(np.abs(actual-padded)))
    assert delta<1e-3 and padding_delta<1e-3,(delta,padding_delta)
    variable=[]
    for row in examples:
        for context in ['',row['context']]:
            seq=encode_pair(tokenizer,context,row['text'],256)['input_ids']
            x=torch.tensor([seq],dtype=torch.long);m=torch.ones_like(x)
            with torch.inference_mode():expected=model(x,m).numpy()
            observed=session.run(['logits'],{'input_ids':x.numpy(),'attention_mask':m.numpy()})[0]
            difference=float(np.max(np.abs(expected-observed)));assert difference<1e-3
            variable.append(dict(length=len(seq),max_delta=difference))
    for name in ('tokenizer.json','tokenizer_config.json'):shutil.copyfile(Path(tokenizer_directory)/name,directory/name)
    model.config.save_pretrained(directory)
    proof=dict(parameters=parameters,parameter_limit_exclusive=20_000_000,storage_bits=bits,compute_precision='FP32',
               logits_parity_max_delta=delta,padding_max_delta=padding_delta,matrices=converted,
               packed_int4=bits==4,qat_scales_preserved=quant is not None,variable_length_checks=variable,
               model_sha256=file_hash(directory/'model.onnx'))
    (directory/'export_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
    del graph,session;gc.collect()
    return proof
