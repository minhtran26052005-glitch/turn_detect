"""Exact-file cleanup of verified rejected checkpoints, authorized by user."""
import argparse,json,hashlib,subprocess,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/failed_checkpoint_cleanup_20261108'

def plan():
 x=json.loads((OUT/'remote_inventory.json').read_text());chosen=[];kept=[]
 for f in x['files']:
  p=f['path'];name=p.split('/')[1];m=x['metadata'].get(name,{})
  rejected=False;reason=None
  if name in ['multilingual_under20m_20261103_v1','multilingual_tiny_compression_20261102_v1']:
   assert m['final_report.json']['complete'] and m['final_report.json']['accepted'] is False
   rejected=True;reason='Completed final report accepted=false'
   if '/teacher/' in p:rejected=False
  if name in ['asr_only_adaptation_20261107_v1','asr_effective_batch16_20261107_v1']:
   assert m['task_state.json']['complete'] and m['task_state.json']['adopted'] is False
   rejected=True;reason='Completed bounded trial adopted=false; user now authorizes removing rejected weights'
   if p.endswith('/frozen/parent_full_resume.pt'):rejected=False
  if rejected:chosen.append(dict(f,reason=reason))
  else:kept.append(f)
 protected={(f['dev'],f['ino']) for f in kept}
 chosen=[f for f in chosen if (f['dev'],f['ino']) not in protected]
 local=json.loads((OUT/'windows_inventory.json').read_text())['files'];win=[]
 for f in local:
  p=Path(f['path']);rel=p.relative_to(ROOT).as_posix();reason=None;evidence=None
  if rel.startswith('outputs/distillation_20261024/runs/'):
   run=ROOT/Path(*Path(rel).parts[:4]);q=run/'training_report.json'
   if q.exists() and json.loads(q.read_text()).get('status')=='DEV_TARGET_NOT_MET':reason='Training report DEV_TARGET_NOT_MET';evidence=str(q)
  if rel.startswith('outputs/stream_v4_20261025/'):
   q=ROOT/'outputs/stream_v4_20261025/summary.json';m=json.loads(q.read_text())
   assert m['user_target_achieved'] is False and m['status']=='TRAINED_AND_EVALUATED_REAL_TARGET_NOT_ACHIEVED'
   reason='Completed stream_v4 real target explicitly not achieved';evidence=str(q)
  if rel=='outputs/compression_20261024/work/int8_quint8_no_constfold/model.onnx':
   q=ROOT/'outputs/compression_20261024/int8_quint8_full_dev/report.json';assert json.loads(q.read_text())['dev_quality_target_met'] is False
   reason='Specific exported INT8 candidate dev_quality_target_met=false';evidence=str(q)
  if reason:
   assert p.resolve().is_relative_to(ROOT.resolve()) and not p.is_symlink();s=p.stat()
   win.append(dict(f,relative=rel,mtime_ns=s.st_mtime_ns,reason=reason,evidence=evidence))
 p=dict(user_authorized=True,scope='Only exact checkpoint/export files in rejected branches; preserve sources, datasets, reports, traces, main paused v7 and teacher controls',remote=chosen,windows=win,
  protected_remote=[f['path'] for f in kept],no_recursive_directory_deletion=True)
 (OUT/'deletion_plan.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(remote_files=len(chosen),remote_logical_gib=sum(f['size'] for f in chosen)/2**30,windows_files=len(win),windows_logical_gib=sum(f['size'] for f in win)/2**30,windows_paths=[f['relative'] for f in win]),ensure_ascii=False))

def execute():
 p=json.loads((OUT/'deletion_plan.json').read_text());assert p['user_authorized'] and not (OUT/'deletion_receipt.json').exists()
 code=r'''import pathlib,json,hashlib,os,shutil,datetime
root=pathlib.Path('/home/edabk/hoangbpm/turn_detect').resolve();plan=json.loads(PLAN);out=root/'outputs/failed_checkpoint_cleanup_20261108';out.mkdir(exist_ok=True)
assert not (out/'deletion_receipt.json').exists()
for q in pathlib.Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:c=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (OSError,UnicodeError):continue
 assert not c.startswith(str(root/'.venv/bin/python')),c
before=shutil.disk_usage(root).free;verified=[]
for f in plan['remote']:
 q=root/f['path'];assert q.resolve().is_relative_to(root) and not q.is_symlink() and q.is_file()
 s=q.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(f['dev'],f['ino'],f['size'],f['mtime_ns'])
 h=hashlib.sha256()
 with q.open('rb') as stream:
  for b in iter(lambda:stream.read(4*2**20),b''):h.update(b)
 verified.append(dict(f,sha256=h.hexdigest()))
(out/'deletion_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
removed=[]
for f in verified:
 q=root/f['path'];s=q.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(f['dev'],f['ino'],f['size'],f['mtime_ns']);q.unlink();assert not q.exists();removed.append(f)
 t=out/'deletion_progress.tmp';t.write_text(json.dumps({'removed':removed},indent=2)+'\n');t.replace(out/'deletion_progress.json')
assert all((root/n).exists() for n in plan['protected_remote'])
for name in sorted({f['path'].split('/')[1] for f in removed}):
 d=root/'outputs'/name;s=d/'task_state.json'
 if s.exists():
  x=json.loads(s.read_text());x['checkpoint_cleanup']={'user_requested':True,'weights_removed':True,'receipt':str(out/'deletion_receipt.json'),'rerun_requires_restoring_or_retraining':True};t=s.with_name(s.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(s)
 (d/'CHECKPOINTS_REMOVED_BY_USER.json').write_text(json.dumps({'user_requested':True,'paths':[f['path'] for f in removed if f['path'].split('/')[1]==name],'reports_data_and_traces_retained':True},indent=2)+'\n')
receipt=dict(complete=True,at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),removed=removed,free_before=before,free_after=shutil.disk_usage(root).free,all_protected_files_present=True)
(out/'deletion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
'''.replace('PLAN',repr(json.dumps(p)))
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,encoding='utf-8',capture_output=True,timeout=120)
 assert r.returncode==0,r.stderr;remote=json.loads(r.stdout);(OUT/'remote_deletion_receipt.json').write_text(json.dumps(remote,indent=2)+'\n')
 before=shutil.disk_usage('C:/').free;verified=[]
 for f in p['windows']:
  q=Path(f['path']);assert q.resolve().is_relative_to(ROOT.resolve()) and not q.is_symlink() and q.is_file();s=q.stat();assert (s.st_size,s.st_mtime_ns)==(f['size'],f['mtime_ns'])
  h=hashlib.sha256()
  with q.open('rb') as stream:
   for b in iter(lambda:stream.read(4*2**20),b''):h.update(b)
  verified.append(dict(f,sha256=h.hexdigest()))
 for f in verified:
  q=Path(f['path']);s=q.stat();assert (s.st_size,s.st_mtime_ns)==(f['size'],f['mtime_ns']);q.unlink();assert not q.exists()
 receipt=dict(complete=True,remote=remote,windows=dict(removed=verified,free_before=before,free_after=shutil.disk_usage('C:/').free),scope=p['scope'])
 (OUT/'deletion_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(complete=True,remote_deleted=len(remote['removed']),remote_measured_freed_gib=(remote['free_after']-remote['free_before'])/2**30,windows_deleted=len(verified),windows_measured_freed_gib=(receipt['windows']['free_after']-before)/2**30,windows_free_gib=receipt['windows']['free_after']/2**30)))

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','execute']);args=a.parse_args();plan() if args.action=='plan' else execute()
