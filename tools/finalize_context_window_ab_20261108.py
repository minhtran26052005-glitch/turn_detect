"""Build/verify review, then separately mark the completed A/B as reported."""
import argparse
import base64
import json
from pathlib import Path
from manage_context_window_ab_20261108 import remote, main as collect_main, ROOT, OUT, NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('review', 'report'))
    args = parser.parse_args()
    if args.action == 'review':
        code = '''import pathlib,json,hashlib,subprocess,math
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/context_window_ab_20261108_v1'
report=json.loads((p/'report.json').read_text());proof=json.loads((p/'completion_verification.json').read_text());assert report['complete'] and proof['complete']
assert hashlib.sha256((p/'report.json').read_bytes()).hexdigest()==proof['report_sha256']
checked=0;populations={}
def check(m):
 global checked
 tp,tn,fp,fn=[m[k] for k in ('tp','tn','fp','fn')];assert min(tp,tn,fp,fn)>=0
 assert m['samples']==tp+tn+fp+fn and m['actual_eot']==tp+fn
 expected={'eot_precision':tp/(tp+fp) if tp+fp else None,'eot_recall':tp/(tp+fn) if tp+fn else None,'eot_f1':2*tp/(2*tp+fp+fn) if tp+fn else None}
 for k,v in expected.items():assert m[k] is None if v is None else abs(m[k]-v)<1e-12
 checked+=1
def group(result):
 for lang,m in result['by_language'].items():
  check(m);denom=(m['samples'],m['actual_eot'],m['tn']+m['fp'])
  key=('calibration' if m['samples']<15000 else 'monitor',lang)
  assert populations.setdefault(key,denom)==denom
 for m in result['by_domain']:check(m)
 for lang,m in result['by_language'].items():
  ds=[d for d in result['by_domain'] if d['language']==lang]
  assert all(sum(d[k] for d in ds)==m[k] for k in ('tp','tn','fp','fn','samples','actual_eot'))
for arm in ('predicted_reset','source_window'):
 group(report['arms'][arm]['selected'])
 selection=json.loads((p/arm/'selection_receipt.json').read_text())
 assert selection['selected_before_monitor']
 for curve in selection['calibration']:group(curve)
 best=max(selection['calibration'],key=lambda x:(min((m['eot_f1'] or 0) for m in x['by_language'].values()),sum((m['eot_f1'] or 0) for m in x['by_language'].values())))
 assert best['threshold']==selection['threshold']==report['arms'][arm]['selected']['threshold']
(p/'independent_metric_verification.json').write_text(json.dumps(dict(complete=True,metric_groups_checked=checked,denominators_identical_across_arms_and_thresholds=True,domain_confusions_sum_to_language_totals=True,calibration_argmax_recomputed=True),indent=2)+'\\n')
result=subprocess.run(['python3',str(r/'tools/build_context_window_review_20261108.py')],cwd=r,capture_output=True,text=True,check=True)
print(json.dumps({'complete':True,'metric_groups_checked':checked,'review':json.loads(result.stdout)}))
'''
        print(json.dumps(remote(code, timeout=60))); return
    code = '''import pathlib,json,hashlib,base64,datetime,subprocess,shutil
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/context_window_ab_20261108_v1';m=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
state=json.loads((p/'task_state.json').read_text());report=json.loads((p/'report.json').read_text());proof=json.loads((p/'completion_verification.json').read_text());review=json.loads((p/'review_verification.json').read_text());metric=json.loads((p/'independent_metric_verification.json').read_text())
assert state['complete'] and report['complete'] and proof['complete'] and review['complete'] and metric['complete']
assert not state.get('reported'), 'Already reported; never duplicate'
digest=hashlib.sha256((p/'report.json').read_bytes()).hexdigest();assert digest==proof['report_sha256']==review['report_sha256']
launch=json.loads((p/'launcher.json').read_text());assert not pathlib.Path('/proc',str(launch['pid'])).exists(),'Wait for normal worker exit'
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==launch['boot_id']
assert (m/'STOP_REQUESTED').exists() and json.loads((m/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
h=hashlib.sha256()
with (m/'runs/finetune/latest_resume.pt').open('rb') as f:
 for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
assert h.hexdigest()==report['protocol']['parent_full_sha256']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
state.update(reported=True,reported_utc=now,report_sha256=digest,adopted=False,last_notified_dev_branch_step=20833,last_notified_dev_global_step=3752006)
def atomic(path,value):
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\\n');temp.replace(path)
atomic(p/'task_state.json',state)
main=json.loads((m/'task_state.json').read_text());assert main['desired_state']=='PAUSED_BY_USER_FOR_TEST'
main['last_reported_context_window_ab']='context_window_ab_20261108_v1@3752006'
main['authorized_context_window_ab'].update(complete=True,reported=True,adopted=False,report_sha256=digest,reported_utc=now,worker_exited=True)
atomic(m/'task_state.json',main)
note='\\n\\n## CONTEXT_WINDOW_AB_COMPLETE_20261108\\nHai nhánh mới đều hoàn tất333315points/20833updates, fullstate/source/DEVtrace/metric/audit đã kiểm chứng. Kết quả và ca DEV ghép cặp: ../context_window_ab_20261108_v1/review.html. Không tự adopt/mở thêm epoch, không TEST/translator/default change. Main vẫn PAUSED3731173/STOP_REQUESTED; fullSHA nguyên.\\n'
path=m/'README.md';text=path.read_text()
if 'CONTEXT_WINDOW_AB_COMPLETE_20261108' not in text:path.write_text(text+note)
receipt=dict(complete=True,at_utc=now,report_sha256=digest,parent_full_sha256=h.hexdigest(),worker_absent=True,main_paused=True,reported=True,adopted=False,ssd_free_gib=shutil.disk_usage(r).free/2**30,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30,mount=subprocess.run(['findmnt','-no','SOURCE,UUID,OPTIONS','/mnt/hdd500'],capture_output=True,text=True).stdout,gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout)
atomic(p/'notification_verification.json',receipt)
print(json.dumps(dict(receipt=receipt,files={name:base64.b64encode(path.read_bytes()).decode() for name,path in [('task_state.json',p/'task_state.json'),('main_task_state.json',m/'task_state.json'),('main_README.md',m/'README.md'),('notification_verification.json',p/'notification_verification.json')]})))
'''
    result = remote(code, timeout=60)
    targets = {'task_state.json':OUT/'task_state.json', 'notification_verification.json':OUT/'notification_verification.json',
               'main_task_state.json':ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/task_state.json',
               'main_README.md':ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7/README.md'}
    for name, encoded in result['files'].items():
        raw = base64.b64decode(encoded)
        assert len(raw) <= 2*2**20
        targets[name].write_bytes(raw)
    print(json.dumps(result['receipt']))


if __name__ == '__main__':
    main()
