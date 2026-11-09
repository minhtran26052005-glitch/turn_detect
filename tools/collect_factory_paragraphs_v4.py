"""Collect only the bounded review, not the full generated corpus."""
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime

DEST=Path('outputs/factory_dialogue_pilot_20261030/qwen_v4_paragraphs_review')


def main():
    rt=ColabRuntime();session=rt.connect()
    if session['execution_state']!='idle':raise RuntimeError('Colab kernel busy; retry without interrupting it')
    code='''import pathlib,json,base64,hashlib
r=pathlib.Path('/content/factory_dialogue_pilot_20261030_v4_paragraphs/output')
status=json.loads((r/'status.json').read_text())
assert status['stage']=='COMPLETE'
data={}
for name in ('manifest.json','dialogues.jsonl'):
 data[name]=(r/name).read_bytes()
for language in ('vi','en','zh'):
 p=r/('examples_'+language+'.jsonl')
 with p.open(encoding='utf-8') as f:
  rows=[json.loads(line) for line in f]
 groups={}
 for row in rows:
  key=(row['label'],row['lookahead_words'])
  if len(groups.setdefault(key,[]))<30:groups[key].append(row)
 selected=[row for group in groups.values() for row in group]
 data['examples_'+language+'_sample.jsonl']=''.join(json.dumps(row,ensure_ascii=False)+'\\n' for row in selected).encode()
rejects=[json.loads(line) for line in (r/'rejected.jsonl').read_text().splitlines() if line.strip()]
data['rejected_summary.json']=json.dumps([{'scenario_id':x['scenario_id'],'error':x['error']} for x in rejects],ensure_ascii=False).encode()
assert sum(map(len,data.values()))<5*1024*1024
print(json.dumps({name:{'base64':base64.b64encode(value).decode(),'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)} for name,value in data.items()}))'''
    try:
        result=rt.execute(code,timeout=60,echo=False)
        if result['status']!='ok':raise RuntimeError('Colab review collection failed')
        value=json.loads(''.join(x.get('text','') for x in result['outputs'] if x['output_type']=='stream'))
        DEST.mkdir(parents=True,exist_ok=False)
        receipt={}
        for name,entry in value.items():
            data=base64.b64decode(entry['base64']);assert hashlib.sha256(data).hexdigest()==entry['sha256']
            (DEST/name).write_bytes(data);receipt[name]={'bytes':len(data),'sha256':entry['sha256']}
        (DEST/'collection_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps({'status':'COLLECTED','files':receipt},ensure_ascii=False))
    finally:rt.close()

if __name__=='__main__':main()
