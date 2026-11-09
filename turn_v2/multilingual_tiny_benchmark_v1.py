"""CPU batch-one wall-time/RSS benchmark in a fresh subprocess per model."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['teacher','onnx'],required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in a.cases.read_text().splitlines()][:24]
    if a.kind=='teacher':
        import torch
        import transformers.utils as utils
        import transformers.utils.import_utils as imports
        utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
        utils.is_librosa_available=imports.is_librosa_available=lambda:False
        from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
        from safetensors.torch import load_file
        from .tiny_compression_v1.common import encode_pair
        torch.set_num_threads(2);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,token=False)
        model=AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(a.model,local_files_only=True,token=False))
        model.load_state_dict(load_file(str(a.model/'model.safetensors')),strict=True);model.eval()
        def predict(row):
            r=encode_pair(tok,row['context'],row['text'],256);r={k:torch.as_tensor([v],dtype=torch.long) for k,v in r.items()}
            with torch.inference_mode():return model(**r).logits.softmax(-1)[0,1].item()
        backend='PyTorch FP32 CPU teacher'
    else:
        path=a.model/'run_model.py';spec=importlib.util.spec_from_file_location('tiny_runtime',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        detector=module.TinyOnnxDetector(a.model,threads=2)
        def predict(row):return detector.predict(row['text'],row['context'])['prob_EOT']
        backend='ONNX Runtime CPU, matrix storage precision from contract'
    for row in rows[:5]:predict(row)
    times=[]
    for _ in range(5):
        for row in rows:
            start=time.perf_counter_ns();predict(row);times.append((time.perf_counter_ns()-start)/1e6)
    print(json.dumps({'backend':backend,'batch_size':1,'threads':2,'timed_calls':len(times),'cases':len(rows),
        'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'mean_ms':float(np.mean(times)),
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'measurement_includes_normalization_and_tokenization':True,'shared_host_not_isolated':True}))


if __name__=='__main__':main()
