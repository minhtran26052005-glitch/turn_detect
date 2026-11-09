"""Versioned CPU-cache handoff; preserves model checkpoints and cache prefix."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell

LOCAL=Path('outputs/translation_segmenter_v15d_20261026/full_epoch_v1')
SOURCES=['translation_segmenter_cache_v1.py','translation_segmenter_full_train_v5.py',
         'translation_segmenter_full_campaign_v8.py','translation_segmenter_cache_handoff_v1.py']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','status'));parser.add_argument('--session-path');a=parser.parse_args()
    r=ColabRuntime(0,a.session_path)
    try:
        connection=r.connect();assert connection['execution_state']=='idle'
        if a.action=='launch':
            expected={}
            for name in SOURCES:
                raw=(Path('turn_v2')/name).read_bytes();compile(raw,name,'exec');expected['turn_v2/'+name]=hashlib.sha256(raw).hexdigest()
                existing=cell(r,f'''import json,hashlib
from pathlib import Path
p=Path('/content/turn_detect_v2/turn_v2')/{name!r};print(json.dumps({{'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}}))''')
                if existing['exists']:assert existing['sha256']==expected['turn_v2/'+name],'Preserve differing source: '+name
                else:r.request('PUT','api/contents/content/turn_detect_v2/turn_v2/'+name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
            result=cell(r,f'''import json,subprocess,sys,os
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/'outputs/translation_segmenter_v15d_20261026/full_epoch_v1';control=job/'cache_handoff_v1'
assert not control.exists(),'Existing handoff must be inspected, never duplicated'
assert json.loads((job/'training_status.json').read_text())['stage']=='LOADING_FULL_EPOCH_DATA_CACHE'
control.mkdir();(control/'config.json').write_text(json.dumps({{'source_sha256':{expected!r}}},indent=2))
command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_cache_handoff_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'log':str(control/'worker.log')}};(control/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))''')
        else:
            result=cell(r,'''import json
from pathlib import Path
job=Path('/content/turn_detect_v2/outputs/translation_segmenter_v15d_20261026/full_epoch_v1');control=job/'cache_handoff_v1';o={}
for name in ('launcher.json','status.json','receipt.json','error.json','cache_prefix_adoption.json'):
 p=control/name
 if p.exists():o[name]=json.loads(p.read_text())
log=control/'worker.log'
if log.exists():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1800));o['log_tail']=f.read().decode(errors='replace')
o['cache_status']={p.name:json.loads(p.read_text()) for p in (job/'cache_status').glob('*.json')}
print(json.dumps(o))''')
    finally:r.close()
    out={'connection':connection,'result':result};(LOCAL/('prefetch_v1_'+a.action+'_receipt.json')).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))


if __name__=='__main__':main()
