"""Collect bounded new TEST reports and fresh process/storage evidence only."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
NAME='sentence_mix_end_only_test_20261107_v1'
FILES=['launcher.json','task_state.json','status.json','error.json','technical_fixture_verification.json',
       'pause_checkpoint_verification.json','checkpoint_receipt.json','locked_protocol.json',
       'input_manifest.json','frozen_dev.json','report.json','completion_verification.json',
       'end_only_analysis.json','translation_input_end_only_report.json','translation_input_segmentation.json',
       'independent_end_only_verification.json','translation_sentence_count_review_locked.json']


def main():
    assert shutil.disk_usage('C:/').free>10*2**30+5*2**20
    code='''import pathlib,json,base64,hashlib,shutil,subprocess,datetime
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');p=r/'outputs/NAME';out={};size=0
for n in FILES:
 f=p/n
 if not f.exists():continue
 assert f.stat().st_size<=2*2**20
 b=f.read_bytes();size+=len(b);assert size<=5*2**20
 out[n]={'b64':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()}
l=json.loads((p/'launcher.json').read_text());proc=pathlib.Path('/proc',str(l['pid']));boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip();process={'present':proc.exists()}
if proc.exists():
 raw=(proc/'stat').read_text();z=raw[raw.rfind(')')+2:].split();cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode();process.update(pid=l['pid'],start_ticks=z[19],cpu_ticks=int(z[11])+int(z[12]),cmdline=cmd,identity_match=(z[19]==str(l['start_ticks']) and boot==l['boot_id'] and l['module'] in cmd))
train=r/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7';tl=json.loads((train/'launcher.json').read_text());mount=[z.split() for z in pathlib.Path('/proc/mounts').read_text().splitlines() if z.split()[1]=='/mnt/hdd500'];trace=p/'predictions.jsonl.gz'
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
proof=dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),boot_id=boot,process=process,trainer_present=pathlib.Path('/proc',str(tl['pid'])).exists(),trainer_stop_marker=(train/'STOP_REQUESTED').exists(),trainer_status=json.loads((train/'status.json').read_text()),gpu=gpu,ssd_free_gib=shutil.disk_usage(r).free/2**30,hdd_mount=mount,hdd_free_gib=shutil.disk_usage('/mnt/hdd500').free/2**30 if mount else None,trace_bytes=trace.stat().st_size if trace.exists() else None)
print(json.dumps({'files':out,'proof':proof}))
'''.replace('NAME',NAME).replace('FILES',repr(FILES))
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','edabk@100.91.209.124','python3 -'],input=code,text=True,capture_output=True,encoding='utf-8',timeout=45)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);folder=ROOT/'outputs'/NAME;folder.mkdir(exist_ok=True)
    for n,item in value['files'].items():
        data=base64.b64decode(item['b64']);assert hashlib.sha256(data).hexdigest()==item['sha256'];(folder/n).write_bytes(data)
    proof=value['proof'];proof['windows_free_gib']=shutil.disk_usage('C:/').free/2**30
    (folder/'heartbeat_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    if 'pause_checkpoint_verification.json' in value['files']:
        main=ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7';s=json.loads((main/'task_state.json').read_text(encoding='utf-8'))
        s.update(desired_state='PAUSED_BY_USER_FOR_TEST',auto_resume_authorized=False,phase='PAUSED_WITH_FULL_CHECKPOINT')
        job_state=json.loads((folder/'task_state.json').read_text()) if (folder/'task_state.json').exists() else {}
        s['active_end_only_test']=dict(output=str(folder),status=json.loads((folder/'status.json').read_text())['stage'],reported=job_state.get('reported',False))
        (main/'task_state.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(proof=proof,status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else None,error=json.loads((folder/'error.json').read_text()) if (folder/'error.json').exists() else None),ensure_ascii=False))


if __name__=='__main__':main()
