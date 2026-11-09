"""Start/inspect the explicitly requested test-priority window, once."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime

LOCAL=Path('outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1')
SOURCE=Path('turn_v2/translation_segmenter_test_priority_v1.py')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','status'));args=parser.parse_args()
    runtime=ColabRuntime(0)
    try:
        connection=runtime.connect();assert connection['execution_state']=='idle','Kernel busy; no queued control cell'
        if args.action=='launch':
            raw=SOURCE.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
            runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+SOURCE.name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
            code=f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');control=root/{str(LOCAL).replace(chr(92),'/')!r}/'priority_window_v1'
assert not control.exists(),'Existing priority window is preserved; no duplicate'
data=(root/'colab'/{SOURCE.name!r}).read_bytes();assert hashlib.sha256(data).hexdigest()=={checksum!r};target=root/'turn_v2'/{SOURCE.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r}
else:target.write_bytes(data)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_test_priority_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command}};(control/'launcher.json').write_text(json.dumps(r,indent=2))
watch=command+['--watchdog']
with (control/'watchdog.log').open('a') as stream:w=subprocess.Popen(watch,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
wr={{'pid':w.pid,'start_ticks':(Path('/proc')/str(w.pid)/'stat').read_text().split()[21],'command':watch}};(control/'watchdog_launcher.json').write_text(json.dumps(wr,indent=2));print(json.dumps({{'supervisor':r,'watchdog':wr}}))'''
        else:
            code=f'''import json
from pathlib import Path
root=Path('/content/turn_detect_v2');control=root/{str(LOCAL).replace(chr(92),'/')!r}/'priority_window_v1';out={{}}
for name in ('status.json','pause_receipt.json','resume_receipt.json','error.json','watchdog_receipt.json','launcher.json'):
 p=control/name
 if p.exists():out[name]=json.loads(p.read_text())
print(json.dumps(out))'''
        result=runtime.execute(code,timeout=30,echo=False)
        if result.get('status')!='ok':raise RuntimeError(json.dumps([x.get('evalue') for x in result.get('outputs',[]) if x.get('output_type')=='error']))
        value=json.loads(''.join(x.get('text','') for x in result.get('outputs',[]) if x.get('output_type')=='stream'))
        (LOCAL/('priority_'+args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,'result':value},indent=2)+'\n',encoding='utf-8')
        print(json.dumps(value))
    finally:runtime.close()


if __name__=='__main__':main()
