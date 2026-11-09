"""Regression-only portability check of the downloaded exact Colab teacher."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.stream_v4_local import AutoModelForSequenceClassification, AutoTokenizer
from turn_v2.stream_v4_train import encode_rows, infer
from turn_v2.common import file_hash, read_jsonl, write_json


def main():
    root=Path('outputs/stream_v4_20261025')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    model_path=root/summary['selected_checkpoint']
    assert file_hash(model_path/'model.safetensors')==summary['weights_sha256']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True,dtype=torch.float32).cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    results={}
    for name,path in [('fresh_vi_natural_auto_labels',Path('data_v4/vi_natural_pause_20261025/cases.jsonl')),
        ('fresh_en_zh_natural_auto_labels',Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl'))]:
        rows=list(read_jsonl(path));ids,lengths=encode_rows(rows,tokenizer)
        probabilities=infer(model,ids,lengths,batch_size=8)
        expected=np.load(root/'colab_final'/(name+'_predictions.npz'))['prob_EOT']
        delta=float(np.max(np.abs(probabilities-expected)))
        thresholds=np.array([summary['thresholds'][r['language']] for r in rows])
        decisions_equal=bool(np.array_equal(probabilities>=thresholds,expected>=thresholds))
        results[name]={'cases':len(rows),'max_abs_probability_difference':delta,'decisions_equal':decisions_equal}
        assert delta<1e-4 and decisions_equal,results[name]
    receipt={'status':'PASS','purpose':'Regression portability only; no threshold/model selection or claim of new independent test',
        'weights_sha256':summary['weights_sha256'],'local_torch':torch.__version__,'local_gpu':torch.cuda.get_device_name(0),
        'tf32':False,'comparisons':results}
    write_json(root/'checkpoint_portability.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
