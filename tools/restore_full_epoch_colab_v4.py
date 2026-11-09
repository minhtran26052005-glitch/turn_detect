"""Prepared empty-VM full-epoch recovery controller. No large Windows transfer."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell

LOCAL=Path('outputs/translation_segmenter_v15d_20261026')


def launch(r):
    config=json.loads((LOCAL/'drive_backup_config.json').read_text(encoding='utf-8'))
    checked=cell(r,f'''import json,hashlib,os,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';config={config!r}
assert not run.exists(),'Existing campaign state must be inspected, not overwritten'
assert os.path.ismount('/content/drive')
drive=Path(config['drive_path']);assert json.loads((drive/'destination_identity.json').read_text())==config['identity']
raw=(drive/'latest_manifest.json').read_bytes();m=json.loads(raw)
assert m['pass_complete'] and m['full_recovery_available'] and m.get('full_epoch_recipe_supported')
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:argv=(p/'cmdline').read_bytes().split(bytes([0]))
 except OSError:continue
 if b'-m' in argv and any(x.startswith((b'turn_v2.translation_segmenter_',b'turn_v2.restore_')) for x in argv):active.append(p.name)
assert not active,'Existing campaign/control workers: '+str(active)
assert shutil.disk_usage('/content').free>50*1024**3
root.mkdir(exist_ok=True);(root/'turn_v2').mkdir(exist_ok=True);(root/'restore_full_epoch_control_v4').mkdir(exist_ok=False)
print(json.dumps({{'manifest_sha256':hashlib.sha256(raw).hexdigest(),'captured_at_utc':m['captured_at_utc']}}))''')
    config['manifest_sha256']=checked['manifest_sha256']
    files=[('turn_v2/__init__.py',Path('turn_v2/__init__.py').read_bytes()),
      ('turn_v2/restore_drive_campaign_v3.py',Path('turn_v2/restore_drive_campaign_v3.py').read_bytes()),
      ('turn_v2/restore_full_epoch_campaign_v4.py',Path('turn_v2/restore_full_epoch_campaign_v4.py').read_bytes()),
      ('restore_full_epoch_control_v4/config.json',json.dumps(config).encode())]
    for name,raw in files:
        assert len(raw)<5*1024**2
        r.request('PUT','api/contents/content/turn_detect_v2/'+name,
          json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(r,'''import json,subprocess,sys,os
from pathlib import Path
root=Path('/content/turn_detect_v2');control=root/'restore_full_epoch_control_v4'
assert not (root/'outputs/translation_segmenter_v15d_20261026').exists()
cmd=[sys.executable,'-u','-m','turn_v2.restore_full_epoch_campaign_v4']
with (control/'worker.log').open('a') as log:p=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'})
a={'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':cmd,'log':str(control/'worker.log')};(control/'launcher.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))''')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch',));parser.add_argument('--session-path');a=parser.parse_args()
    r=ColabRuntime(0,a.session_path)
    try:
        connection=r.connect();assert connection['execution_state']=='idle'
        result=launch(r)
    finally:r.close()
    out={'connection':connection,'result':result};(LOCAL/'full_epoch_v1/restore_launch_receipt.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))


if __name__=='__main__':main()
