"""Verify portable inference, including real UTF-8 text without shell re-encoding."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-only',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];model=root/'models/turn_v2_distilled_min_20261024/onnx'
    spec=importlib.util.spec_from_file_location('portable_runtime',model/'run_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    detector=module.TinyOnnxDetector(model)
    cases=['hà nội là thủ đô và là một trong những trung tâm văn hóa lớn của việt nam','tôi muốn đặt vé đi','cảm ơn bạn']
    if a.native_only:
        result={'torch_imported':'torch' in sys.modules,'transformers_imported':'transformers' in sys.modules,
                'demos':[{'text':v,**detector.predict(v)} for v in cases]}
        result['passed']=not result['torch_imported'] and not result['transformers_imported']
        print(json.dumps(result));return
    sys.path.insert(0,str(root))
    from turn_v2.predict import TurnDetectorV2
    from turn_v2.common import read_jsonl,encode_pair,write_json
    reference=TurnDetectorV2(model)
    rows=list(read_jsonl(root/'data_v2/frozen/dev.jsonl'))
    edge=[{'text':'','context':''},{'text':'XIN CHÀO!!! ＡＢＣ１２３','context':'Xin chào? Tôi muốn hỏi:'},
          {'text':'cảm ơn bạn đã giúp đỡ ❤','context':'a '*1000},{'text':'xin chào '*300,'context':'ngữ cảnh '*500}]
    changed=sum(detector.encode(r['text'],r.get('context',''))!=encode_pair(reference.tokenizer,r.get('context',''),r['text'],256) for r in rows+edge)
    comparisons=[]
    for row in rows[::137][:128]+edge:
        left=detector.predict(row['text'],row.get('context',''));right=reference.predict(row['text'],row.get('context',''))
        comparisons.append({'delta':abs(left['prob_EOT']-right['prob_EOT']),'same':left['prediction']==right['prediction']})
    result={'full_dev_input_encoding_rows':len(rows),'utf8_edge_cases':edge,'changed_encoded_rows':changed,
            'inference_samples':len(comparisons),'max_probability_delta':max(v['delta'] for v in comparisons),
            'same_predictions':all(v['same'] for v in comparisons),
            'passed':changed==0 and all(v['same'] and v['delta']<1e-6 for v in comparisons)}
    destination=root/'outputs/distillation_20261024';write_json(destination/'portable_runtime_parity.json',result)
    child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--native-only'],capture_output=True,text=True,encoding='utf-8',timeout=30,check=True)
    native=json.loads(child.stdout);write_json(destination/'standalone_smoke.json',native)
    print(json.dumps({'portable':{k:v for k,v in result.items() if k!='utf8_edge_cases'},'native':native}))
    assert result['passed'] and native['passed']


if __name__=='__main__':main()
