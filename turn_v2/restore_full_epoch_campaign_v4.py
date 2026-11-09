"""Prepared empty-VM recovery for full epochs, including sampler cursors.

No base-model fallback. A committed manifest, every content object, source,
cleaned SQLite, optimizer/RNG and permutation/cursor must verify before launch.
Never invoke on a VM which already has campaign state or live workers.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import traceback

from . import restore_drive_campaign_v3 as io

PROJECT=Path('/content/turn_detect_v2')
ROOT=PROJECT/'outputs/translation_segmenter_v15d_20261026'
CONTROL=PROJECT/'restore_full_epoch_control_v4'
io.CONTROL=CONTROL


def fingerprint(db,language,split):
    h=hashlib.sha256();n=0
    for eid,ih,label,k in db.execute("SELECT example_id,input_hash,label,k FROM records WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id",(language,split)):
        h.update((eid+'|'+ih+'|'+label+'|'+str(k)+'\n').encode());n+=1
    return {'rows':n,'sha256':h.hexdigest()}


def main():
    config=json.loads((CONTROL/'config.json').read_text());drive=Path(config['drive_path'])
    assert os.path.ismount('/content/drive')
    assert json.loads((drive/'destination_identity.json').read_text())==config['identity']
    raw=(drive/'latest_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==config['manifest_sha256']
    manifest=json.loads(raw);assert manifest['pass_complete'] and manifest['full_recovery_available']
    assert manifest.get('full_epoch_recipe_supported'),'Use legacy recovery for an older recipe'
    assert not ROOT.exists(),'Preserve existing campaign state'
    ROOT.mkdir(parents=True);files=manifest['files']
    io.state('RESTORING_FULL_EPOCH_SOURCE_AND_DATA')
    source=CONTROL/'source.zip';io.copy_object(drive,files['source/source_bundle.zip'],source)
    io.extract_archive(source,PROJECT)
    datazip=ROOT/'recovery/data_snapshot.zip';io.copy_object(drive,files['data/data_snapshot.zip'],datazip)
    unpacked=CONTROL/'raw_data';io.extract_archive(datazip,unpacked)
    assert (unpacked/'state_snapshot.sqlite').exists()
    shutil.copy2(unpacked/'state_snapshot.sqlite',ROOT/'corpus.sqlite')
    for name in ('config.json','data_summary.json'):
        if (unpacked/name).exists():shutil.copy2(unpacked/name,ROOT/name)
    io.write_json(ROOT/'recovery/latest_snapshot.json',files['data/data_snapshot.zip']['source_receipt'])
    for logical,entry in files.items():
        if logical.startswith(('teachers/','full_epoch_v1/')):
            io.copy_object(drive,entry,io.target_under(ROOT,logical))
        elif logical.startswith('heldout_test_v2/'):
            # Frozen tests are restored unchanged, never scored or used for selection.
            target=ROOT/'drive_backup_v1/frozen_test_database.zip' if logical=='heldout_test_v2/frozen_test_database.zip' else io.target_under(ROOT,logical)
            io.copy_object(drive,entry,target)
    versions=json.loads((PROJECT/'runtime_versions.json').read_text())
    dependencies=[f'{k}=={versions[k]}' for k in ('transformers','datasets','numpy','sentencepiece','jieba','huggingface_hub') if k in versions]
    dependencies+=['safetensors','requests','beautifulsoup4','scikit-learn']
    io.subprocess_checked([sys.executable,'-m','pip','install','--quiet','--disable-pip-version-check',*dependencies],
                          'dependency_install.log','INSTALLING_RECORDED_DEPENDENCIES')
    import numpy as np
    import torch
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    rawdb=sqlite3.connect('file:'+str(ROOT/'corpus.sqlite')+'?mode=ro',uri=True)
    assert rawdb.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    next_cycle=int(rawdb.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0])
    # Verify each generation once, and every language fingerprint in that DB.
    generations={};checked={}
    for logical,entry in files.items():
        if logical.startswith('full_epoch_v1/datasets/') and logical.endswith('/manifest.json'):
            path=io.target_under(ROOT,logical);m=json.loads(path.read_text());folder=path.parent
            io.extract_archive(folder/'dataset.zip',folder)
            assert m['complete'] and io.sha(folder/'dataset.sqlite')==m['database_sha256']
            db=sqlite3.connect('file:'+str(folder/'dataset.sqlite')+'?mode=ro',uri=True)
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            for language in ('vi','en','zh'):
                for split in ('train','dev'):
                    proof=fingerprint(db,language,split);wanted=m['fingerprints'][language][split]
                    assert proof['rows']==wanted['rows'] and proof['sha256']==wanted['sha256']
            db.close();generations[m['generation']]=m
    if not generations:
        assert not any(files['teachers/'+lang+'/latest_resume.pt'].get('source_receipt',{}).get('full_epoch_generation') for lang in ('vi','en','zh')), 'Sampler checkpoint requires its immutable dataset'
    for language in ('vi','en','zh'):
        folder=ROOT/'teachers'/language;entry=files['teachers/'+language+'/latest_resume.pt'];receipt=entry['source_receipt']
        saved=torch.load(folder/'latest_resume.pt',map_location='cpu',weights_only=False)
        for key in ('cycle','cycle_step','total_steps','training_fingerprint'):assert saved[key]==receipt[key],(language,key)
        assert saved['optimizer']['state'] and all(key in saved for key in ('torch_rng','cuda_rng','numpy_rng'))
        sampler=saved.get('full_epoch')
        if sampler:
            m=generations[sampler['generation']];fp=m['fingerprints'][language]['train']
            assert sampler['fingerprint']==fp['sha256']==saved['training_fingerprint']
            n=fp['rows'];cursor=sampler['position'];assert 0<=cursor<=n
            order=sampler['order'];visits=sampler['visits']
            assert np.array_equal(np.sort(order),np.arange(n,dtype=np.int64))
            expected=np.zeros(n,dtype=np.uint8);expected[order[:cursor]]=1
            assert np.array_equal(visits,expected)
            checked[language]={'generation':sampler['generation'],'epoch':sampler['epoch'],'cursor':cursor,'rows':n,'optimizer_steps':saved['total_steps'],'sampler_verified':True}
        else:
            fp,n=io.historical_fingerprint(rawdb,language,saved['cycle']);assert fp==saved['training_fingerprint']
            checked[language]={'legacy_checkpoint_adopted':True,'optimizer_steps':saved['total_steps'],'historical_train_rows':n}
        io.write_json(folder/'resume_receipt.json',receipt);del saved
    rawdb.close()
    current=ROOT/'full_epoch_v1/current_dataset.json'
    if not current.exists() and generations:io.write_json(current,max(generations.values(),key=lambda x:x['source_cycle']))
    # Recover epoch completion receipts from report archive. Mutable launcher/status
    # are historical evidence; never adopt their old process identity.
    if 'reports/reports_and_logs.zip' in files:
        archive=CONTROL/'reports.zip';io.copy_object(drive,files['reports/reports_and_logs.zip'],archive)
        historical=CONTROL/'reports_unpacked';io.extract_archive(archive,historical)
        source=historical/'full_epoch_v1/runs'
        if source.exists():
            for path in source.rglob('*'):
                if not path.is_file():continue
                rel=path.relative_to(source);parts=rel.parts
                # Report ZIP can be later than a pinned optimizer object. Keep
                # later evidence historical; never skip uncommitted row visits.
                if len(parts)>=3 and path.name.startswith('epoch_') and path.suffix=='.json':
                    lang=parts[1];cp=checked.get(lang,{})
                    completed=cp.get('epoch',0)-(cp.get('cursor',0)<cp.get('rows',0))
                    number=int(path.stem.split('_')[-1])
                    if parts[0]!=cp.get('generation') or number>completed:continue
                target=ROOT/'full_epoch_v1/runs'/rel
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    proof={'at_utc':io.now(),'manifest_sha256':config['manifest_sha256'],'checkpoints':checked,
      'data_integrity_and_fingerprints_verified':True,'optimizer_rng_and_exact_epoch_cursor_restored':True,
      'raw_crawl_next_cycle':next_cycle,'base_model_reinitialization':False,'Windows_large_downloads':False,
      'restore_module':'restore_full_epoch_campaign_v4','cleaned_dataset_rebuild_required':not bool(generations)}
    io.write_json(CONTROL/'restore_receipt.json',proof)
    backup=ROOT/'drive_backup_v1';backup.mkdir(exist_ok=True)
    io.write_json(backup/'config.json',{k:config[k] for k in ('drive_path','identity','interval_seconds')})
    io.write_json(backup/'latest_verified_manifest.json',manifest)
    command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_drive_backup_v5']
    with (backup/'worker_v2.log').open('a') as log:p=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=io.child_record(p,command,backup/'worker_v2.log');record['drive_path']=str(drive)
    io.write_json(backup/'launcher.json',record)
    if not generations:
        job=ROOT/'full_epoch_v1';job.mkdir(exist_ok=True)
        io.write_json(job/'data_status.json',{'stage':'STARTING_RESTORED_RAW_SOURCE_AUDIT'})
        command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_full_data_v3']
        with (job/'data_worker_v3.log').open('a') as log:p=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        io.write_json(job/'data_launcher_v3.json',io.child_record(p,command,job/'data_worker_v3.log'))
    module=manifest.get('active_campaign_launcher',{}).get('module','turn_v2.translation_segmenter_full_campaign_v4')
    import re
    assert re.fullmatch(r'turn_v2\.translation_segmenter_full_campaign_v[0-9]+',module)
    expected=manifest.get('active_campaign_launcher',{}).get('source_sha256')
    if expected:assert io.sha(PROJECT/'turn_v2'/(module.split('.')[-1]+'.py'))==expected
    command=[sys.executable,'-u','-m',module]
    io.write_json(ROOT/'launcher.json',{'pid':os.getpid(),'start_ticks':(Path('/proc')/str(os.getpid())/'stat').read_text().split()[21],
       'command':command,'module':command[-1],'log':str(CONTROL/'worker.log'),'restore_receipt':str(CONTROL/'restore_receipt.json')})
    io.state('VERIFIED_HANDOFF_TO_EXACT_FULL_EPOCH_RESUME',**proof)
    os.execv(sys.executable,command)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        io.write_json(CONTROL/'error.json',{'at_utc':io.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        io.state('FAILED_PRESERVED_NO_BASE_FALLBACK',error=repr(exc));raise
