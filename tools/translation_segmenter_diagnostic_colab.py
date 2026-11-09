"""Control one new-source cycle75 diagnostic; bounded local artifacts only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil

from colab_runtime import ColabRuntime

RUN = 'outputs/translation_segmenter_v15d_20261026'
NAME = 'diagnostic_test_c75_v1'
LOCAL = Path(RUN) / NAME
SOURCE = Path('turn_v2/translation_segmenter_diagnostic_c75_v1.py')


def cell(runtime, code):
    result = runtime.execute(code, timeout=60, echo=False)
    if result.get('status') != 'ok':
        raise RuntimeError(json.dumps([x for x in result.get('outputs', []) if x.get('output_type') == 'error'], ensure_ascii=False))
    return json.loads(''.join(x.get('text', '') for x in result.get('outputs', []) if x.get('output_type') == 'stream'))


def launch(runtime):
    raw = SOURCE.read_bytes(); checksum = hashlib.sha256(raw).hexdigest()
    cell(runtime, "from pathlib import Path\nimport json\np=Path('/content/turn_detect_v2/colab');p.mkdir(exist_ok=True);print(json.dumps({'staging_ready':p.is_dir()}))")
    runtime.request('PUT', 'api/contents/content/turn_detect_v2/colab/' + SOURCE.name,
        json={'type': 'file', 'format': 'base64', 'content': base64.b64encode(raw).decode()})
    return cell(runtime, f'''import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/{RUN!r};job=run/{NAME!r}
def live(r):
 try:
  argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
assert live(json.loads((run/'launcher.json').read_text())),'Campaign parent unavailable'
if job.exists():
 record=json.loads((job/'launcher.json').read_text()) if (job/'launcher.json').exists() else {{}}
 assert live(record),'Existing diagnostic artifacts preserved; do not relaunch'
 print(json.dumps({{'reused_existing_worker':True,'launcher':record}}))
else:
 assert shutil.disk_usage(root).free>50*1024**3,'Colab headroom too low'
 assert os.path.ismount('/content/drive'),'Drive must remain mounted'
 for lang in ('vi','en','zh'):
  r=json.loads((run/'teachers'/lang/'resume_receipt.json').read_text())
  assert r.get('cycle')==75 and r.get('cycle_step')==100,'Exact cycle75 checkpoint unavailable for '+lang
 source=root/'colab'/{SOURCE.name!r};data=source.read_bytes();assert hashlib.sha256(data).hexdigest()=={checksum!r}
 target=root/'turn_v2'/{SOURCE.name!r}
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r},'Preserve code; version fixes'
 else:target.write_bytes(data)
 job.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_diagnostic_c75_v1']
 with (job/'worker.log').open('a') as stream:
  p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}})
 record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'source_sha256':{checksum!r},'log':str((job/'worker.log').relative_to(root))}}
 (job/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps({{'launched':True,'launcher':record}}))''')


def status(runtime):
    return cell(runtime, f'''import json,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};out={{}}
for name in ('launcher.json','status.json','error.json','pinned_receipts.json','manifest.json','partial_report.json','report.json','drive_manifest.json','training_coverage_v1.json','error_analysis_v1.json','parser_audit_v1.json','crawl_training_balance_v1.json','continuation_v2.json','continuation_v2_error.json','analysis_v1/status.json','analysis_v1/launcher.json','analysis_v1/error.json','parser_audit_v1/status.json','parser_audit_v1/launcher.json','parser_audit_v1/error.json','analysis_v2/status.json','analysis_v2/launcher.json','analysis_v2/error.json','finalizer_v1/status.json','finalizer_v1/launcher.json','finalizer_v1/error.json','finalization_receipt_v1.json'):
 p=job/name
 if p.exists():out[name]=json.loads(p.read_text())
r=out.get('launcher.json',{{}});ok=False
try:
 argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
 ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['process_verified']=ok;out['free_bytes']=shutil.disk_usage(root).free
r=out.get('analysis_v1/launcher.json',{{}});ok=False
try:
 argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
 ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['analysis_process_verified']=ok
r=out.get('parser_audit_v1/launcher.json',{{}});ok=False
try:
 argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
 ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['parser_audit_process_verified']=ok
r=out.get('analysis_v2/launcher.json',{{}});ok=False
try:
 argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
 ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
except (OSError,KeyError,ValueError,IndexError):pass
out['analysis_v2_process_verified']=ok
for folder in ('finalizer_v1','reserve_audit_v1','actual_seen_control_v1','priority_window_v1'):
 for filename in ('launcher.json','status.json','error.json'):
  p=job/folder/filename
  if p.exists():out[folder+'/'+filename]=json.loads(p.read_text())
 r=out.get(folder+'/launcher.json',{{}});ok=False
 try:
  argv=r['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(r['pid'])
  ok=module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==r['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):pass
 out[folder+'_process_verified']=ok
for filename in ('actual_seen_fit_control_v1.json','reserve_exposure_audit_v1.json'):
 p=job/filename
 if p.exists():out[filename]=json.loads(p.read_text())
out['cohort_partitions']={{}}
for lang in ('vi','en','zh'):
 p=job/(lang+'_partition.json')
 if p.exists():out['cohort_partitions'][lang]={{k:v for k,v in json.loads(p.read_text()).items() if k!='unscored_reserve_source_ids'}}
p=root/out.get('launcher.json',{{}}).get('log',str((job/'worker.log').relative_to(root)))
if p.exists():
 with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-2500));out['log_tail']=stream.read().decode(errors='replace')
print(json.dumps(out,ensure_ascii=False))''')


def analyze(runtime):
    source=Path('turn_v2/translation_segmenter_diagnostic_analysis_v1.py')
    raw=source.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+source.name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};control=job/'analysis_v1'
assert (job/'launcher.json').exists(),'Main diagnostic must be launched first'
assert not control.exists(),'Existing analysis state is preserved; do not create duplicates'
data=(root/'colab'/{source.name!r}).read_bytes();assert hashlib.sha256(data).hexdigest()=={checksum!r}
target=root/'turn_v2'/{source.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r},'Preserve source; version fixes'
else:target.write_bytes(data)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_diagnostic_analysis_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'source_sha256':{checksum!r}}};(control/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':True,'launcher':r}}))''')


def parser_audit(runtime):
    uploads=[]
    for filename in ('translation_segmenter_causal_extraction_v1.py','translation_segmenter_parser_audit_v1.py'):
        source=Path('turn_v2')/filename;raw=source.read_bytes();checksum=hashlib.sha256(raw).hexdigest();uploads.append((filename,checksum))
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+filename,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};control=job/'parser_audit_v1'
assert not control.exists(),'Existing parser audit is preserved; no duplicate'
for filename,checksum in {uploads!r}:
 data=(root/'colab'/filename).read_bytes();assert hashlib.sha256(data).hexdigest()==checksum
 target=root/'turn_v2'/filename
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==checksum,'Preserve existing module'
 else:target.write_bytes(data)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_parser_audit_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command}};(control/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':True,'launcher':r}}))''')


def continue_v2(runtime):
    uploads=[]
    for name in ('translation_segmenter_diagnostic_c75_v2.py','translation_segmenter_diagnostic_analysis_v2.py'):
        raw=(Path('turn_v2')/name).read_bytes();checksum=hashlib.sha256(raw).hexdigest();uploads.append((name,checksum))
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r}
def live(record):
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
for path in (job/'launcher.json',job/'analysis_v1/launcher.json',job/'parser_audit_v1/launcher.json'):
 assert not live(json.loads(path.read_text())),'An original worker is still active; preserve it'
assert not (job/'analysis_v2').exists() and not (job/'continuation_v2.json').exists(),'Continuation already exists'
assert (job/'manifest.json').exists() and (job/'error.json').exists(),'Expected preserved pre-score failure unavailable'
for name in ('launcher.json','status.json'):
 shutil.copy2(job/name,job/(name.replace('.json','_v1_preserved.json')))
for name,checksum in {uploads!r}:
 raw=(root/'colab'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==checksum
 target=root/'turn_v2'/name
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==checksum
 else:target.write_bytes(raw)
launched=[]
for module,control in [('turn_v2.translation_segmenter_diagnostic_c75_v2',job),('turn_v2.translation_segmenter_diagnostic_analysis_v2',job/'analysis_v2')]:
 control.mkdir(exist_ok=True);command=[sys.executable,'-u','-m',module]
 with (control/'worker_v2.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}})
 record={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'log':str((control/'worker_v2.log').relative_to(root))}};(control/'launcher.json').write_text(json.dumps(record,indent=2));launched.append(record)
print(json.dumps({{'launched':launched,'frozen_inputs_preserved':True}}))''')


def finalize(runtime):
    uploads=[]
    for path in (Path('turn_v2/translation_segmenter_diagnostic_finalize_v1.py'),Path('tools/report_translation_segmenter_diagnostic.py')):
        raw=path.read_bytes();checksum=hashlib.sha256(raw).hexdigest();uploads.append((path.name,checksum))
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+path.name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};control=job/'finalizer_v1'
assert not control.exists(),'Existing finalizer must be preserved'
for name,checksum in {uploads!r}:
 data=(root/'colab'/name).read_bytes();assert hashlib.sha256(data).hexdigest()==checksum
 if name.startswith('translation_segmenter_diagnostic_finalize'):
  target=root/'turn_v2'/name
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==checksum
  else:target.write_bytes(data)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_diagnostic_finalize_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command}};(control/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':True,'launcher':r}}))''')


def clean_only_v3(runtime):
    uploads=[]
    for name in ('translation_segmenter_clean_scoring_v1.py','translation_segmenter_diagnostic_c75_v3.py'):
        raw=(Path('turn_v2')/name).read_bytes();checksum=hashlib.sha256(raw).hexdigest();uploads.append((name,checksum))
        runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};run=root/{RUN!r}
def live(record):
 try:
  argv=record['command'];module=argv[argv.index('-m')+1];p=Path('/proc')/str(record['pid'])
  return module.encode() in (p/'cmdline').read_bytes().split(bytes([0])) and (p/'stat').read_text().split()[21]==record['start_ticks']
 except (OSError,KeyError,ValueError,IndexError):return False
record=json.loads((job/'launcher.json').read_text());assert live(record)
assert record['command'][-1]=='turn_v2.translation_segmenter_diagnostic_c75_v2','Unexpected evaluator version'
assert json.loads((job/'status.json').read_text()).get('stage')=='WAITING_FOR_CAMPAIGN_GPU','Do not interrupt scoring'
assert not (job/'partial_report.json').exists() and not (job/'report.json').exists() and not list(job.glob('test_predictions_*.jsonl.gz')),'Scored evidence already exists; preserve it'
active_gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).splitlines()
assert str(record['pid']) not in [x.strip() for x in active_gpu_pids],'Evaluator has initialized CUDA; preserve scored progress'
child=json.loads((run/'child.json').read_text());assert live(child) and '--phase' in child['command'] and child['command'][child['command'].index('--phase')+1]=='train','Campaign must still own GPU'
proof={{'evaluator_before':record,'campaign_child_untouched':child,'no_scored_trace_or_report':True,'evaluator_has_no_CUDA_context':True,'reason':'Never infer rows removed by the reserve/overlap audit'}}
(job/'pre_inference_revision_v3.json').write_text(json.dumps(proof,indent=2))
shutil.copy2(job/'launcher.json',job/'launcher_v2_preserved.json');shutil.copy2(job/'status.json',job/'status_v2_preserved.json');shutil.copy2(job/'continuation_v2.json',job/'continuation_v2_firstpass_preserved.json')
os.kill(record['pid'],signal.SIGTERM)
for i in range(40):
 if not live(record):break
 time.sleep(.25)
assert not live(record),'Evaluator has not exited; no duplicate created'
assert not list(job.glob('test_predictions_*.jsonl.gz')),'Unexpected scored trace; preserve it'
for name,checksum in {uploads!r}:
 raw=(root/'colab'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==checksum;target=root/'turn_v2'/name
 if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==checksum
 else:target.write_bytes(raw)
command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_diagnostic_c75_v3']
with (job/'worker_v3.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'log':str((job/'worker_v3.log').relative_to(root))}};(job/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':r,'pre_inference_guard':proof}}))''')


def reserve_audit(runtime):
    source=Path('turn_v2/translation_segmenter_reserve_exposure_audit_v1.py');raw=source.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+source.name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};control=job/'reserve_audit_v1'
assert not control.exists(),'Preserve existing reserve auditor'
raw=(root/'colab'/{source.name!r}).read_bytes();assert hashlib.sha256(raw).hexdigest()=={checksum!r}
target=root/'turn_v2'/{source.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r}
else:target.write_bytes(raw)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_reserve_exposure_audit_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command}};(control/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':True,'launcher':r}}))''')


def seen_control(runtime):
    source=Path('turn_v2/translation_segmenter_actual_seen_control_v1.py');raw=source.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    runtime.request('PUT','api/contents/content/turn_detect_v2/colab/'+source.name,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
    return cell(runtime,f'''import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{RUN!r}/{NAME!r};control=job/'actual_seen_control_v1'
assert not control.exists(),'Preserve existing actual-seen fit control'
raw=(root/'colab'/{source.name!r}).read_bytes();assert hashlib.sha256(raw).hexdigest()=={checksum!r};target=root/'turn_v2'/{source.name!r}
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()=={checksum!r}
else:target.write_bytes(raw)
control.mkdir();command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_actual_seen_control_v1']
with (control/'worker.log').open('a') as stream:p=subprocess.Popen(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}})
r={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command}};(control/'launcher.json').write_text(json.dumps(r,indent=2));print(json.dumps({{'launched':True,'launcher':r}}))''')


def render_review():
    report = json.loads((LOCAL / 'report.json').read_text(encoding='utf-8'))
    samples = [json.loads(raw) for raw in (LOCAL / 'review_samples.jsonl').read_text(encoding='utf-8').splitlines() if raw.strip()]
    table = []
    for lang, value in report['languages'].items():
        m = value['metrics']
        table.append(f'<tr><td>{lang}</td><td>{m["samples"]:,}</td><td>{m["accuracy"]:.2%}</td><td>{m["eot_f1"]:.2%}</td><td>{value["leakage_audit"]["removed_rows"]:,}</td></tr>')
    cases = []
    for row in samples:
        cases.append('<details><summary>' + html.escape(f'{row["language"]} | target {row["label"]} | predicted {row["predicted"]} | lookahead {row["lookahead_words"]} | correct {row["correct"]}') + '</summary><p>' + html.escape(row['context'] + ' ▏ ' + row['text']) + '</p><p>Target mark: ' + html.escape(row.get('target_punctuation','')) + '</p><pre>' + html.escape(row.get('original_source_paragraph_for_posthoc_label_audit','')) + '</pre></details>')
    page = '<!doctype html><meta charset="utf-8"><title>Cycle75 diagnostic test and errors</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse}td,th{padding:12px;border:1px solid #ccc}details{padding:12px;margin:12px 0;border:1px solid #ccc}pre{white-space:pre-wrap}</style><h1>Cycle75: new-source prose diagnostic test</h1><p>Frozen cycle75 weights, dev-only thresholds; test pages added after45. An unscored20% source reserve remains protected. Diagnostic errors may inform future TRAIN-only crawling. This is punctuation-derived prose, not human/audio conversational gold.</p><table><tr><th>Language</th><th>Clean rows</th><th>Accuracy</th><th>EOT F1</th><th>Overlap removed</th></tr>' + ''.join(table) + '</table><h2>Error audit samples</h2>' + ''.join(cases)
    (LOCAL / 'review.html').write_text(page, encoding='utf-8')


def fetch(runtime):
    if shutil.disk_usage(LOCAL).free < 10*1024**3 + 5*1024**2:
        raise RuntimeError('Windows reserve prevents review transfer; remote artifacts remain on Drive')
    result = status(runtime)
    if not result.get('report.json',{}).get('complete'):
        raise RuntimeError('Diagnostic evaluation is not complete; partial state preserved')
    total = 0
    names=['report.json','manifest.json','status.json','review_samples.jsonl','drive_manifest.json']
    names += [name for name in ('training_coverage_v1.json','error_analysis_v1.json','parser_audit_v1.json','crawl_training_balance_v1.json','actual_seen_fit_control_v1.json','reserve_exposure_audit_v1.json') if name in result]
    final_ready=result.get('finalizer_v1/status.json',{}).get('stage')=='COMPLETE'
    if final_ready:names+=['error_analysis.md','crawl_plan_v1.json','finalization_receipt_v1.json']
    for name in names:
        value = runtime.request('GET', 'api/contents/content/turn_detect_v2/' + RUN + '/' + NAME + '/' + name + '?format=text').json()
        if value.get('type') != 'file' or int(value.get('size',0)) > 5*1024**2:
            raise RuntimeError('Unexpected compact artifact size/type')
        raw = base64.b64decode(value['content']) if value.get('format') == 'base64' else value['content'].encode('utf-8')
        total += len(raw)
        if total > 5*1024**2:
            raise RuntimeError('Total local diagnostic transfer exceeds5MiB')
        (LOCAL/name).write_bytes(raw)
    render_review()
    if final_ready:
        # Reuse the transferred samples; avoid downloading a second copy of all
        # paragraphs embedded in remote HTML, keeping the aggregate below5MiB.
        page=(LOCAL/'review.html').read_text(encoding='utf-8')
        detail=html.escape((LOCAL/'error_analysis.md').read_text(encoding='utf-8'))
        page=page.replace('<h2>Error audit samples</h2>','<h2>Phân tích lỗi và hướng crawl</h2><pre>'+detail+'</pre><h2>Error audit samples</h2>')
        (LOCAL/'review.html').write_text(page,encoding='utf-8')
    return {'complete':True,'bytes_fetched':total,'review':str(LOCAL/'review.html'),'languages':result['report.json']['languages']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','status','fetch','analyze','parser-audit','continue-v2','finalize','clean-only-v3','reserve-audit','seen-control'));parser.add_argument('--session-path');args=parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    runtime=ColabRuntime(0,session_path=args.session_path)
    try:
        connection=runtime.connect()
        if connection.get('execution_state')!='idle':raise RuntimeError('Kernel busy; no diagnostic control cell queued')
        result={'launch':launch,'status':status,'fetch':fetch,'analyze':analyze,'parser-audit':parser_audit,'continue-v2':continue_v2,'finalize':finalize,'clean-only-v3':clean_only_v3,'reserve-audit':reserve_audit,'seen-control':seen_control}[args.action](runtime)
        (LOCAL/(args.action+'_receipt.json')).write_text(json.dumps({'connection':connection,'result':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if args.action=='status':
            path=LOCAL.parent/'task_state.json'
            task=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            task['diagnostic_test_c75_v1']={'state':result.get('status.json'),
                'process_verified':result.get('process_verified'),
                'analysis_state':result.get('analysis_v2/status.json',result.get('analysis_v1/status.json')),
                'analysis_process_verified':result.get('analysis_v2_process_verified',result.get('analysis_process_verified')),
                'manifest':result.get('manifest.json'),
                'completed_languages':list(result.get('partial_report.json',{}).get('languages',{})),
                'control_script':'tools/translation_segmenter_diagnostic_colab.py',
                'old_heldout_not_rescored':True,'unscored_reserve_protected':True}
            path.write_text(json.dumps(task,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'connection':connection,'result':result},ensure_ascii=False))
    finally:runtime.close()


if __name__=='__main__':main()
