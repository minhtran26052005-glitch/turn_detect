"""Control the user-authorized full-data/full-epoch transition; metadata only."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell

RUN=Path('outputs/translation_segmenter_v15d_20261026')
LOCAL=RUN/'full_epoch_v1'
SOURCES=[
 'turn_v2/translation_segmenter_full_data_v2.py',
 'turn_v2/translation_segmenter_epoch_sampling_v1.py',
 'turn_v2/translation_segmenter_full_train_v1.py',
 'turn_v2/translation_segmenter_full_campaign_v1.py',
 'turn_v2/translation_segmenter_full_transition_v1.py',
 'turn_v2/translation_segmenter_drive_backup_v2.py']


def launch(r):
    files={}
    for name in SOURCES:
        raw=Path(name).read_bytes();compile(raw.decode(),name,'exec');files[name]=hashlib.sha256(raw).hexdigest()
        r.request('PUT','api/contents/content/turn_detect_v2/colab/'+Path(name).name,
                  json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    code=f'''import json,hashlib,os,subprocess,sys,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/{str(RUN).replace(chr(92),'/')!r};job=run/'full_epoch_v1';control=job/'transition_v1'
assert not control.exists(),'Transition exists: inspect status, do not duplicate'
assert shutil.disk_usage('/content').free>45*1024**3
assert os.path.ismount('/content/drive')
config=json.loads((run/'drive_backup_v1/config.json').read_text())
assert json.loads((Path(config['drive_path'])/'destination_identity.json').read_text())==config['identity']
for name,expected in {files!r}.items():
 raw=(root/'colab'/Path(name).name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
 p=root/name
 if p.exists():assert p.read_bytes()==raw,'Preserve source; version collision: '+name
 else:p.write_bytes(raw)
from turn_v2.translation_segmenter_full_campaign_v1 import live
assert live(json.loads((run/'launcher.json').read_text())),'Campaign parent identity unavailable'
previous=json.loads((job/'data_launcher.json').read_text());assert not live(previous),'Old data audit still live'
folders=[p for p in (job/'datasets').iterdir() if (p/'source_snapshot.sqlite').exists()]
assert len(folders)==1,'Ambiguous partial source generation; inspect before adoption'
request={{'generation':folders[0].name,'technical_continuation':'v2 adds SQL split metadata; v1 error preserved','no_model_training_restarted':True}}
(job/'data_build_request.json').write_text(json.dumps(request,indent=2))
control.mkdir();(control/'source_receipt.json').write_text(json.dumps({files!r},indent=2))
(job/'config.json').write_text(json.dumps({{'languages':['vi','en','zh'],'full_epoch_without_replacement':True,'microbatch':32,'accumulation':2,'chunk_steps_per_language':500,'minimum_complete_epochs_before_convergence':3,'dev_only_selection':True,'loss_zero_required':False,'unlimited_campaign':True,'local_large_transfers':False,'source':{files!r}}},indent=2))
def spawn(module,extra,log):
 command=[sys.executable,'-u','-m',module]+extra
 with log.open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
 return {{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'log':str(log)}}
audit=spawn('turn_v2.translation_segmenter_full_data_v2',[],job/'data_worker_v2.log');(job/'data_launcher_v2.json').write_text(json.dumps(audit,indent=2))
supervisor=spawn('turn_v2.translation_segmenter_full_transition_v1',[],control/'worker.log');(control/'launcher.json').write_text(json.dumps(supervisor,indent=2))
watchdog=spawn('turn_v2.translation_segmenter_full_transition_v1',['--watchdog'],control/'watchdog.log');(control/'watchdog_launcher.json').write_text(json.dumps(watchdog,indent=2))
print(json.dumps({{'audit':audit,'transition':supervisor,'watchdog':watchdog,'generation':folders[0].name}}))'''
    compile(code,'full_epoch_launch','exec');return cell(r,code)


def status(r,via_terminal=False):
    code='''import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026';job=run/'full_epoch_v1';o={}
o.update(project_exists=root.exists(),campaign_state_exists=run.exists(),
         source_package_exists=(root/'turn_v2/__init__.py').exists(),
         boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         drive_mounted=os.path.ismount('/content/drive'))
def live(record):
 try:
  p=Path('/proc')/str(record['pid']);argv=record['command'];module=argv[argv.index('-m')+1]
  stat=(p/'stat').read_text().split()
  return stat[2]!='Z' and module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and stat[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError,TypeError):return False
controls=[p for p in job.glob('transition_v*') if p.is_dir() and p.name.split('_v')[-1].isdigit()]
control=max(controls,key=lambda p:int(p.name.split('_v')[-1])) if controls else job/'transition_v1'
restores=[p for p in root.glob('restore_full_epoch_control_v*') if p.is_dir() and p.name.split('_v')[-1].isdigit()]
restore=max(restores,key=lambda p:int(p.name.split('_v')[-1])) if restores else root/'restore_full_epoch_control_v7'
for prefix,folder,names in [('',job,('data_status.json','data_error_v2.json','data_error_v3.json','data_error_v4.json','data_error_v5.json','encoding_cache_parity.json','verification_status.json','script_audit_status.json','script_quarantine_receipt_v1.json','current_dataset.json','config.json','campaign_status.json','campaign_error.json','training_status.json','dataset_backup_ready.json','latest_round_receipt.json','policy_v6_activation.json','policy_v7_activation.json','technical_overlap_fix_v5.json','runtime_preflight_v1/report.json','runtime_preflight_v1/partial_report.json','runtime_preflight_v1/error.json')),('transition/',control,('status.json','error.json','pause_receipt.json','old_campaign_finished_receipt.json','new_backup_launcher.json')),('campaign/',run,('launcher.json','child.json','status.json')),('backup/',run/'drive_backup_v1',('launcher.json','status.json','latest_error.json')),('restore/',restore,('status.json','error.json','restore_receipt.json'))]:
 for n in names:
  p=folder/n
  if p.exists():o[prefix+n]=json.loads(p.read_text())
audits=[p for p in job.glob('data_launcher_v*.json') if p.stem.split('_v')[-1].isdigit()]
audit=max(audits,key=lambda p:int(p.stem.split('_v')[-1])) if audits else job/'data_launcher.json'
for n,p in [('audit',audit),('transition',control/'launcher.json'),('campaign',run/'launcher.json'),('child',run/'child.json'),('backup',run/'drive_backup_v1/launcher.json'),('restore',restore/'launcher.json')]:
 if p.exists():o[n+'_process_verified']=live(json.loads(p.read_text()))
o['teachers']={}
o['cache_status']={p.name:json.loads(p.read_text()) for p in (job/'cache_status').glob('*.json')}
o['prefetch_workers']={}
for p in (job/'prefetch_v1').glob('*.json'):
 record=json.loads(p.read_text());o['prefetch_workers'][p.name]={'launcher':record,'process_verified':live(record)}
manifest=job/'current_dataset.json'
if manifest.exists():
 m=json.loads(manifest.read_text());vp=job/'datasets'/m['generation']/'verification_receipt.json'
 o['data_audit_complete']=m.get('complete',False) and vp.exists() and json.loads(vp.read_text()).get('complete',False)
for lang in ('vi','en','zh'):
 o['teachers'][lang]={}
 for name in ('progress.json','resume_receipt.json','last_training_report.json'):
  p=run/'teachers'/lang/name
  if p.exists():o['teachers'][lang][name]=json.loads(p.read_text())
o['remote_free_bytes']=shutil.disk_usage('/content').free
o['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
for n in ('data_worker_v5.log','campaign_worker_v8.log',str((control/'worker.log').relative_to(job))):
 p=job/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1800));o[n+'_tail']=f.read().decode(errors='replace')
print(json.dumps(o,ensure_ascii=False))'''
    if via_terminal:
        from colab_terminal_control import execute
        result=execute(code)
        if result['remote_exception']:raise RuntimeError(result['output'])
        value=json.loads(result['output']);value['control_mode']='EPHEMERAL_TERMINAL_NO_KERNEL_INTERRUPT'
        value['process_identity_reverified']=True
        return value
    return cell(r,code)


def status_http(r):
    """Read bounded files without queuing a cell into a busy original kernel.

    Metadata timestamps are observations, not fresh OS/PID liveness proofs.
    """
    out={'control_mode':'BUSY_KERNEL_BOUNDED_METADATA_ONLY','process_identity_reverified':False}
    prefix='files/content/turn_detect_v2/outputs/translation_segmenter_v15d_20261026/'
    names=('full_epoch_v1/data_status.json','full_epoch_v1/transition_v2/status.json',
           'full_epoch_v1/campaign_status.json','drive_backup_v1/status.json','status.json','launcher.json')
    for name in names:
        try:
            response=r.http.get(r.base_url+prefix+name,timeout=8)
            if response.status_code==404:continue
            response.raise_for_status()
            assert len(response.content)<=5*1024**2
            out[name]=response.json()
        except Exception as exc:out[name+'_read_error']=type(exc).__name__
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('launch','status'));p.add_argument('--session-path');a=p.parse_args()
    r=ColabRuntime(0,a.session_path)
    try:
        c=r.connect()
        if c['execution_state']!='idle':
            assert a.action=='status','No launch is queued into a busy kernel'
            try:result=status(r,via_terminal=True)
            except Exception:result=status_http(r)
        else:result={'launch':launch,'status':status}[a.action](r)
    finally:r.close()
    LOCAL.mkdir(parents=True,exist_ok=True)
    out={'connection':c,'result':result};(LOCAL/(a.action+'_receipt.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
