"""Restore compact champions in RAM and verify a resumable campaign snapshot."""
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sqlite3
import sys
import zipfile
import numpy as np
import torch
from tokenizers import Tokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import normalize,file_hash
from turn_v2.metrics import classification_metrics,probabilities
from turn_v2.tiny_student import TinyTurnGRU

ROOT=Path('outputs/continuous_v13_20261026')


def main():
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    archive=ROOT/'latest_snapshot.zip';receipt=json.loads((ROOT/'snapshot_receipt.json').read_text(encoding='utf-8'))
    check('snapshot_exact_bytes',archive.stat().st_size==receipt['bytes']);check('snapshot_SHA256',file_hash(archive)==receipt['sha256'])
    torch.set_num_threads(2)
    with zipfile.ZipFile(archive) as z:
        check('ZIP_CRC',z.testzip() is None);check('safe_relative_archive_members',all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and ':' not in n for n in z.namelist()))
        # A WAL database cannot be deserialized into :memory: with its WAL header.
        # Validate an independent on-disk copy so SQLite can open its own journal.
        validation_db=ROOT/'validation_state.sqlite'
        validation_db.write_bytes(z.read('outputs/continuous_v13_20261026/recovery/state_snapshot.sqlite'))
        db=sqlite3.connect(validation_db)
        check('SQLite_integrity',db.execute('PRAGMA quick_check').fetchone()[0]=='ok')
        overlap=db.execute('SELECT source_id FROM samples GROUP BY source_id HAVING COUNT(DISTINCT split)>1').fetchall();check('whole_source_train_dev_isolation',len(overlap)==0)
        config=json.loads(z.read('outputs/continuous_v13_20261026/config.json'));check('no_cycle_limit',config['max_cycles']==0 and config['continuous_until_user_stop'] and not config['stop_on_dev_plateau'])
        check('no_silver_quality_stop',config['stop_on_silver90'] is False and config['real_quality_target_certified'] is False)
        for source in ['turn_v2/continuous_v13_r3.py','turn_v2/continuous_v13_state.py','turn_v2/continuous_v13_objective.py']:
            check('archived_executed_source:'+source,z.read(source)==Path(source).read_bytes())
        fingerprints=[]
        for lang in ['vi','en','zh']:
            champion=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('champion:'+lang,)).fetchone()[0]);base=champion['path'];data=z.read(base+'/student.pt')
            digest=hashlib.sha256(data).hexdigest();fingerprints.append(digest);check(lang+':champion_weights_exact',digest==champion.get('student_sha256',digest))
            model=TinyTurnGRU(**json.loads(z.read(base+'/config.json')));model.load_state_dict(torch.load(io.BytesIO(data),map_location='cpu',weights_only=True),strict=True);model.eval()
            check(lang+':CPU_restore_finite',all(torch.isfinite(p).all() for p in model.parameters()))
            rows=[json.loads(row[0]) for row in db.execute("SELECT record FROM samples WHERE language=? AND split='semantic_dev' ORDER BY cycle DESC,key LIMIT 60000",(lang,))]
            labels=np.array([0 if r['label']=='NEOT' else 1 for r in rows]);tok=Tokenizer.from_str(z.read(base+'/tokenizer.json').decode());settings=json.loads(z.read(base+'/tokenizer_config.json'))
            def special(key):
                value=settings[key];value=value['content'] if isinstance(value,dict) else value;return tok.token_to_id(value)
            cls,sep,pad=[special(k) for k in ['cls_token','sep_token','pad_token']];encoded=[]
            for row in rows:
                current=tok.encode(normalize(row['text']),add_special_tokens=False).ids[-253:];budget=253-len(current)
                history=tok.encode(normalize(row.get('context','')),add_special_tokens=False).ids[-budget:] if budget else []
                encoded.append([cls]+history+[sep]+current+[sep])
            width=max(map(len,encoded));ids=np.full((len(rows),width),pad,np.int64);mask=np.zeros_like(ids)
            for i,v in enumerate(encoded):ids[i,:len(v)]=v;mask[i,:len(v)]=1
            with torch.inference_mode():logits=model(torch.from_numpy(ids),torch.from_numpy(mask)).numpy()
            actual=classification_metrics(labels,probabilities(logits),champion['semantic_dev']['metrics']['threshold']);expected=champion['semantic_dev']['metrics']
            check(lang+':semantic_dev_metrics_recomputed',actual['samples']==expected['samples'] and actual['confusion_matrix_NEOT_EOT']==expected['confusion_matrix_NEOT_EOT'])
            frozen=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('semantic_dev_frozen:'+lang,)).fetchone()[0]);content=z.read('outputs/continuous_v13_20261026/semantic_dev/'+lang+'.jsonl')
            check(lang+':dev_fingerprint_preserved',hashlib.sha256(content).hexdigest()==frozen['sha256'])
            check(lang+':model_optimizer_resume_in_archive',any(n.endswith('/resume.pt') and '/'+lang+'/' in n for n in z.namelist()))
        check('three_distinct_model_weights',len(set(fingerprints))==3);db.close()
    nb=json.loads(Path('a.ipynb').read_text(encoding='utf-8'));before=json.loads((ROOT/'a_before_continuous.ipynb').read_text(encoding='utf-8'))
    check('old45_notebook_cells_preserved',nb['cells'][:len(before['cells'])]==before['cells'])
    cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'continuous_v13_status' in c.get('metadata',{}).get('tags',[]));proof=json.loads(Path(cell['metadata']['verification_receipt']).read_text(encoding='utf-8'))
    check('notebook_executed_on_Colab',proof['status']=='ok' and cell['outputs']==proof['outputs'] and hashlib.sha256(''.join(cell['source']).encode()).hexdigest()==proof['source_sha256'])
    result={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','passed':sum(c['passed'] for c in checks),'total':len(checks),'checks':checks,
        'snapshot_cycle':receipt['cycle'],'quality_target_certified':False,'scope':'Technical recovery and metric recomputation; not an independent quality certificate'}
    (ROOT/'recovery_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    if result['status']!='PASS':print(json.dumps([c for c in checks if not c['passed']]));raise SystemExit(1)


if __name__=='__main__':main()
