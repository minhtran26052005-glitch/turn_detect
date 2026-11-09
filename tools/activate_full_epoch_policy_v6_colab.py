"""Activate the revised full-data policy after a verified assets-only restore.

Small source/receipts only; no weights or corpus are transferred to Windows.
Existing differently versioned source and live workers are never overwritten.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell

LOCAL = Path('outputs/translation_segmenter_v15d_20261026/full_epoch_v1')
SOURCES = [
    'translation_segmenter_exact_encoding_v1.py',
    'translation_segmenter_causal_extraction_v2.py',
    'translation_segmenter_full_data_v2.py',
    'translation_segmenter_full_data_v3.py',
    'translation_segmenter_full_data_v4.py',
    'translation_segmenter_full_verify_v1.py',
    'translation_segmenter_epoch_sampling_v1.py',
    'translation_segmenter_full_train_v4.py',
    'translation_segmenter_full_campaign_v6.py',
    'translation_segmenter_drive_backup_v2.py',
    'translation_segmenter_drive_backup_v5.py',
    'translation_segmenter_drive_backup_v6.py',
    'restore_full_epoch_campaign_v6.py',
]


def upload(r):
    expected = {}
    for name in SOURCES:
        raw = (Path('turn_v2') / name).read_bytes()
        compile(raw, name, 'exec')
        checksum = hashlib.sha256(raw).hexdigest()
        check = cell(r, f'''import json,hashlib
from pathlib import Path
p=Path('/content/turn_detect_v2/turn_v2')/{name!r}
print(json.dumps({{'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}}))''')
        if check['exists']:
            assert check['sha256'] == checksum, 'Preserve existing different source: ' + name
        else:
            r.request('PUT', 'api/contents/content/turn_detect_v2/turn_v2/' + name,
                      json={'type':'file','format':'base64',
                            'content':base64.b64encode(raw).decode()})
        expected['turn_v2/' + name] = checksum
    return expected


def activate(r, expected):
    code = f'''import json,hashlib,os,shutil,subprocess,sys
from pathlib import Path
project=Path('/content/turn_detect_v2');run=project/'outputs/translation_segmenter_v15d_20261026';job=run/'full_epoch_v1';control=project/'restore_full_epoch_control_v5'
proof=json.loads((control/'restore_receipt.json').read_text())
assert json.loads((control/'status.json').read_text())['stage']=='ASSETS_RESTORED_VERIFIED_AWAITING_NEW_TRAINING_POLICY'
assert proof['data_integrity_and_fingerprints_verified'] and proof['optimizer_rng_and_exact_epoch_cursor_restored'] and not proof['base_model_reinitialization']
assert proof['raw_crawl_next_cycle']==85
assert all(proof['checkpoints'][l]['optimizer_steps']==1400 for l in ('vi','en','zh'))
active=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 except OSError:continue
 if '-m' in argv and any(x.startswith(('turn_v2.translation_segmenter_','turn_v2.restore_')) for x in argv):active.append({{'pid':p.name,'command':argv}})
assert not active,'Inspect existing workers: '+str(active)
assert shutil.disk_usage('/content').free>50*1024**3
cfg=json.loads((control/'config.json').read_text());drive=Path(cfg['drive_path'])
assert os.path.ismount('/content/drive') and json.loads((drive/'destination_identity.json').read_text())==cfg['identity']
raw=(drive/'latest_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==proof['manifest_sha256']
manifest=json.loads(raw);assert manifest['pass_complete'] and manifest['full_recovery_available']
for name,wanted in {expected!r}.items():assert hashlib.sha256((project/name).read_bytes()).hexdigest()==wanted,name
sys.path.insert(0,str(project));os.chdir(project)
from turn_v2.translation_segmenter_v15c import atomic_json
job.mkdir(exist_ok=True)
assert not (job/'current_dataset.json').exists(),'Inspect an existing cleaned generation before activation'
# Restore best metadata from the receipt of exactly the already verified object.
for language in ('vi','en','zh'):
 folder=run/'teachers'/language;entry=manifest['files']['teachers/'+language+'/best/model.safetensors']
 report=entry['source_receipt'];assert report['weights_sha256']==entry['sha256']
 atomic_json(folder/'best_report.json',report)
atomic_json(job/'policy_v6_source_receipt.json',{{'source':{expected!r},'restored_from':proof,'test_scored':False}})
atomic_json(job/'config.json',{{'languages':['vi','en','zh'],'full_epoch_without_replacement':True,'microbatch':32,'accumulation':2,'chunk_steps_per_language':500,'minimum_complete_epochs_before_convergence':3,'dev_only_selection':True,'loss_zero_required':False,'unlimited_campaign':True,'local_large_transfers':False,'source':{expected!r}}})
atomic_json(job/'data_status.json',{{'stage':'STARTING_RESTORED_RAW_SOURCE_AUDIT','source_cycle':84}})
def spawn(module,log):
 command=[sys.executable,'-u','-m',module]
 with log.open('a') as stream:p=subprocess.Popen(command,cwd=project,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
 return {{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'module':module,'source_sha256':hashlib.sha256((project/'turn_v2'/(module.split('.')[-1]+'.py')).read_bytes()).hexdigest(),'log':str(log)}}
audit=spawn('turn_v2.translation_segmenter_full_data_v4',job/'data_worker_v4.log');atomic_json(job/'data_launcher_v4.json',audit)
campaign=spawn('turn_v2.translation_segmenter_full_campaign_v6',job/'campaign_worker_v6.log');atomic_json(run/'launcher.json',campaign)
backup=run/'drive_backup_v1';backup.mkdir(exist_ok=True)
atomic_json(backup/'config.json',{{k:cfg[k] for k in ('drive_path','identity','interval_seconds')}})
atomic_json(backup/'latest_verified_manifest.json',manifest)
writer=spawn('turn_v2.translation_segmenter_drive_backup_v6',backup/'worker_v6.log');writer['drive_path']=str(drive);atomic_json(backup/'launcher.json',writer)
result={{'audit':audit,'campaign':campaign,'backup':writer,'restore_proof':proof,'data_audit_complete':False,'epoch_complete':False}}
atomic_json(job/'policy_v6_activation.json',result);print(json.dumps(result))'''
    compile(code, 'activate_full_epoch_policy', 'exec')
    return cell(r, code)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('upload','activate'))
    parser.add_argument('--session-path');args=parser.parse_args();r=ColabRuntime(0,args.session_path)
    try:
        connection=r.connect();assert connection['execution_state']=='idle'
        sources=upload(r)
        result=activate(r,sources) if args.action=='activate' else {'uploaded_sources':sources}
    finally:r.close()
    out={'connection':connection,'result':result}
    (LOCAL/('policy_v6_'+args.action+'_receipt.json')).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out))


if __name__=='__main__':main()
