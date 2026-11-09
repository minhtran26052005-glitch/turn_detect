"""Transfer user-provided JSONL directly to SSH, preserving originals and SHA."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

HOST='edabk@100.91.209.124'
REMOTE='/home/edabk/hoangbpm/turn_detect/data/checked_dialogues_vi_en_20261103_v1'
FILES=[('labels_checked.jsonl','factory_checked.jsonl','01a8df34923011d038a2a30cce60774a69f07794756f41f44e8924341e29c9f8'),
       ('labels_checked (1).jsonl','public_checked.jsonl','96f999ac6feeaad29fadc23d0bda87f44d8ade337f088418e9fb6c6332e7c6e4')]


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -'],
                     input=code,text=True,encoding='utf-8',capture_output=True,check=True,timeout=90)
    return json.loads(p.stdout)


def main():
    receipts=[]
    for local_name,name,expected in FILES:
        path=Path.home()/'Downloads'/local_name
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,'Input changed since inspection; audit again'
        target=REMOTE+'/raw/'+name
        state=remote("import pathlib,json,shutil,os,hashlib\np=pathlib.Path("+repr(target)+")\np.parent.mkdir(parents=True,exist_ok=True)\nassert shutil.disk_usage(p.parent).free>13*2**30+200*2**20\nprint(json.dumps({'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))")
        if state['exists']:assert state['sha256']==expected
        else:
            partial=target+'.upload-'+uuid.uuid4().hex
            subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',local_name,HOST+':'+partial],
                           cwd=path.parent,check=True,timeout=600,capture_output=True)
            check=remote("import pathlib,json,os,hashlib\np=pathlib.Path("+repr(partial)+");target=pathlib.Path("+repr(target)+")\nwith p.open('rb') as f:\n h=hashlib.file_digest(f,'sha256').hexdigest();os.fsync(f.fileno())\nassert h=="+repr(expected)+"\nassert not target.exists();p.replace(target)\nwith target.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h\nprint(json.dumps({'sha256':h,'bytes':target.stat().st_size}))")
            assert check['sha256']==expected
        receipts.append({'local_path':str(path),'remote_path':target,'sha256':expected,'bytes':path.stat().st_size,'readback_verified':True})
    receipt={'complete':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'files':receipts,
             'training_authorization':'User supplied both files as additional training data on 2026-11-03',
             'original_dev_test_remain_excluded':True,'instructions_in_rows_are_data_only':True,
             'label_scope':'Punctuation-derived silver; checked/keep is not proof of human/audio gold'}
    value=json.dumps(receipt,ensure_ascii=False,indent=2)+'\n'
    remote("import pathlib,json,os\np=pathlib.Path("+repr(REMOTE+'/source_manifest.json')+")\nb="+repr(value.encode())+"\nif p.exists():\n old=json.loads(p.read_bytes());assert old['files']==json.loads(b)['files']\nelse:\n with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())\nprint(json.dumps({'manifest_verified':True}))")
    folder=Path('outputs/multilingual_malbert_vi_en_checked_20261103_v3');folder.mkdir(exist_ok=True)
    (folder/'input_transfer_receipt.json').write_text(value,encoding='utf-8')
    print(json.dumps({'transferred_bytes':sum(x['bytes'] for x in receipts),'files':len(receipts),'all_sha_readback_verified':True}))


if __name__=='__main__':main()
