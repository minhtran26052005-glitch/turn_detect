"""Copy the completed pilot to the previously verified mounted Drive."""
import json
from pathlib import Path
from colab_runtime import ColabRuntime


def main():
    cfg=json.loads(Path('outputs/translation_segmenter_v15d_20261026/drive_backup_config.json').read_text(encoding='utf-8'))
    rt=ColabRuntime();session=rt.connect()
    if session['execution_state']!='idle':raise RuntimeError('Colab kernel busy; do not interrupt')
    code=f'''import os,json,pathlib,hashlib,shutil
expected=json.loads({json.dumps(json.dumps(cfg['identity'],ensure_ascii=False))})
drive=pathlib.Path({cfg['drive_path']!r})
source=pathlib.Path('/content/factory_dialogue_pilot_20261030_v4_paragraphs')
assert os.path.ismount('/content/drive') and drive.is_dir()
actual=json.loads((drive/'destination_identity.json').read_text())
assert all(actual.get(k)==v for k,v in expected.items()),'Drive destination identity mismatch'
assert json.loads((source/'output/status.json').read_text())['stage']=='COMPLETE'
dest=drive/'factory_paragraphs_pilot_20261030_v4'
dest.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
receipt={{}}
for path in [source/'factory_paragraphs_v4.py']+sorted((source/'output').glob('*.json*')):
 target=dest/path.name
 with path.open('rb') as inf,target.open('xb') as out:
  shutil.copyfileobj(inf,out,1024*1024);out.flush();os.fsync(out.fileno())
 source_sha=sha(path);target_sha=sha(target)
 assert source_sha==target_sha,'Drive readback mismatch'
 receipt[path.name]={{'bytes':path.stat().st_size,'sha256':source_sha,'readback_verified':True}}
data=json.dumps({{'destination':str(dest),'files':receipt,'scope':'synthetic data pilot only'}},ensure_ascii=False,indent=2).encode()
with (dest/'backup_receipt.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
assert (dest/'backup_receipt.json').read_bytes()==data
print(json.dumps({{'status':'BACKUP_VERIFIED','destination':str(dest),'files':receipt}},ensure_ascii=False))'''
    try:
        result=rt.execute(code,timeout=120,echo=True)
        if result['status']!='ok':raise RuntimeError('Drive backup did not complete; preserve partial files')
    finally:rt.close()

if __name__=='__main__':main()
