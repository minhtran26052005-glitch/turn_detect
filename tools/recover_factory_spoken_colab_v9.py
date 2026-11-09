"""Inspect/stage spoken recovery, or hand over one explicitly confirmed owner.

Inspection does not modify Drive. Handover requires a scoped user statement that
the exact previous runtime ended; no credential/token contents are reported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

from colab_runtime import ColabRuntime
from factory_spoken_session import select_current_a_session

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'outputs/factory_spoken_campaign_v9'
FILES = ['factory_spoken_campaign_v9.py', 'factory_spoken_spec_v1.py',
         'factory_spoken_topics_v1.py', 'factory_paragraphs_v4.py']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['inspect', 'handover'])
    p.add_argument('--old-boot', required=True)
    p.add_argument('--new-boot', required=True)
    p.add_argument('--old-pid', type=int, required=True)
    p.add_argument('--old-start-ticks', required=True)
    args = p.parse_args()
    assert args.old_boot != args.new_boot, 'Cross-VM handover only; same-VM worker recovery must retain its owner guard'
    assert shutil.disk_usage(ROOT).free >= 10*2**30+5*2**20
    expected = {n: hashlib.sha256((ROOT/'turn_v2'/n).read_bytes()).hexdigest() for n in FILES}
    auth = None
    if args.action == 'handover':
        state = json.loads((LOCAL/'task_state.json').read_text(encoding='utf-8'))
        auth = state.get('pending_recovery_authorization')
        assert auth and auth.get('previous_runtime_ended_confirmed_by_user') is True, 'Need an actual user confirmation, not elapsed time'
        assert auth['old_boot_id'] == args.old_boot and auth['new_boot_id'] == args.new_boot
        assert auth['old_pid'] == args.old_pid and auth['old_start_ticks'] == args.old_start_ticks
    i, session, _selection = select_current_a_session()
    code = f'''import pathlib,json,os,hashlib,gzip,shutil,subprocess,time,uuid
drive=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
out=drive/'factory_spoken_campaign_v9';root=pathlib.Path('/content/factory_spoken_campaign_v9')
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot=={args.new_boot!r},'Actual new VM changed; authorization is scoped'
assert os.path.ismount('/content/drive'),'User must mount Drive'
identity=json.loads((drive/'destination_identity.json').read_text())
assert identity['nonce']=='9e618b0f2db442a6a9ee3ee18f1d7143' and identity['intended_path']==str(drive)
def sha(data):return hashlib.sha256(data).hexdigest()
owner_raw=(out/'owner.json').read_bytes();owner=json.loads(owner_raw)
assert owner['boot_id']=={args.old_boot!r} and owner['pid']=={args.old_pid!r} and owner['start_ticks']=={args.old_start_ticks!r} and owner['module']=='factory_spoken_campaign_v9','Unexpected previous owner; never bypass its guard'
processes=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  cmd=p.read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if '/content/' in cmd and any(x in cmd for x in ['/factory_spoken_campaign_v','/factory_dialogue_campaign_v','/factory_paragraphs_v','/factory_consult_dialogue_v','/factory_dialogue_pilot_v']):processes.append({{'pid':int(p.parent.name),'cmdline':cmd[:400]}})
 except OSError:pass
assert not processes,'An actual generator is already present on this VM'
assert not (out/'STOP_REQUESTED').exists() and not (root/'STOP_REQUESTED').exists(),'Explicitly stopped campaign needs separate resume instruction'
expected=json.loads({json.dumps(json.dumps(expected))})
sources={{}}
root.mkdir(exist_ok=True)
for name,digest in expected.items():
 data=(out/'source_v9'/name).read_bytes();assert sha(data)==digest,'Immutable Drive source differs'
 compile(data,name,'exec');sources[name]=digest
 target=root/name
 if target.exists():assert sha(target.read_bytes())==digest,'Local source belongs to different work'
 else:
  with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
state_raw=(out/'state.json').read_bytes();state=json.loads(state_raw)
paths=sorted((out/'commits').glob('*.json'));assert paths,'Do not restart from zero when committed data exists'
latest=int(paths[-1].stem);assert state['commit_index']<=latest
verified=[]
for idx in range(max(1,latest-2),latest+1):
 raw=(out/'commits'/f'{{idx:09d}}.json').read_bytes();c=json.loads(raw);assert c['index']==idx
 previous=sha((out/'commits'/f'{{idx-1:09d}}.json').read_bytes()) if idx>1 else None
 assert c['previous_commit_sha256']==previous and c['object_readback_verified']
 data=(out/c['object']).read_bytes();assert len(data)==c['bytes'] and sha(data)==c['object_sha256']
 payload=json.loads(gzip.decompress(data));assert payload['version']==9 and payload['register']=='spoken_transcript' and len(payload['dialogues'])==c['accepted']
 assert all(d['language']==c['language'] for d in payload['dialogues'])
 verified.append({{'index':idx,'language':c['language'],'accepted':c['accepted'],'object_sha256':c['object_sha256'],'commit_sha256':sha(raw),'readback_verified':True}})
latest_raw=(out/'commits'/f'{{latest:09d}}.json').read_bytes();authoritative=json.loads(latest_raw)['after'];authoritative['last_commit_sha256']=sha(latest_raw)
if state['commit_index']==latest:
 assert state['last_commit_sha256']==sha(latest_raw)
 for key in ['total_counts','attempts','rejected','example_counts','cycle_counts']:assert state[key]==authoritative[key]
assert (out/'owner.json').read_bytes()==owner_raw and (out/'state.json').read_bytes()==state_raw,'Drive owner/state changed during inspection; do not race another VM'
gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free,utilization.gpu,name','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20).stdout.strip()
value={{'action':{args.action!r},'new_boot_id':boot,'old_owner':owner,'identity_verified':True,'source_hashes_verified':sources,'processes_on_new_vm':processes,'commit_max':latest,'latest_commit_sha256':sha(latest_raw),'authoritative_state':authoritative,'state_file_is_stale':state['commit_index']<latest,'recent_verified_shards':verified,'colab_free_gib':shutil.disk_usage('/content').free/2**30,'gpu':gpu,'state_owner_stable_during_inspection':True}}
assert value['colab_free_gib']>=43
parts=gpu.splitlines()[0].split(',');assert int(parts[0].strip())>=22528 and int(parts[1].strip())>=19456
if {args.action!r}=='handover':
 auth=json.loads({json.dumps(json.dumps(auth))});assert auth['previous_runtime_ended_confirmed_by_user'] is True
 receipt={{'old_owner':owner,'old_owner_sha256':sha(owner_raw),'new_boot_id':boot,'authorization':auth,'verified_commit_index':latest,'verified_commit_sha256':sha(latest_raw),'verified_counts':authoritative['total_counts'],'source_hashes':sources,'at_unix':time.time(),'old_runtime_liveness_basis':'Explicit user statement that the specific previous runtime ended; no claim from new-VM PID absence'}}
 path=out/'handover'/('owner_'+owner['boot_id'].split('-')[0]+'_to_'+boot.split('-')[0]+'.json');path.parent.mkdir(exist_ok=True)
 raw=(json.dumps(receipt,ensure_ascii=False,sort_keys=True)+'\\n').encode()
 assert not path.exists(),'Existing handover receipt must be inspected instead of rewritten'
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert path.read_bytes()==raw
 assert (out/'owner.json').read_bytes()==owner_raw and (out/'state.json').read_bytes()==state_raw
 updated={{**owner,'stopped':True,'stopped_reason':'Previous runtime ended, explicitly confirmed by user; handover receipt verified','handover_receipt':str(path.relative_to(out)),'handover_receipt_sha256':sha(raw)}}
 temp=out/('owner.json.tmp-handover-'+uuid.uuid4().hex)
 data=(json.dumps(updated,ensure_ascii=False,sort_keys=True)+'\\n').encode()
 with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,out/'owner.json');assert (out/'owner.json').read_bytes()==data
 assert (out/'state.json').read_bytes()==state_raw and (out/'commits'/f'{{latest:09d}}.json').read_bytes()==latest_raw
 value['handover']={{'receipt':str(path),'sha256':sha(raw),'previous_owner_marked_stopped_on_user_confirmation':True,'generation_counters_unchanged':True}}
print(json.dumps(value,ensure_ascii=False))'''
    rt = ColabRuntime(server_index=i, session_path=session['path'])
    rt.kernel_id = session['kernel']['id']
    try:
        result = rt.execute(code, timeout=120, echo=False)
    finally:
        rt.close()
    assert result['status'] == 'ok', str([x.get('evalue') for x in result['outputs'] if x['output_type'] == 'error'])
    data = ''.join(x.get('text', '') for x in result['outputs'] if x['output_type'] == 'stream')
    value = json.loads(data); value['checked_utc'] = datetime.now(timezone.utc).isoformat()
    encoded = (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
    assert len(encoded) <= 5*2**20
    LOCAL.mkdir(parents=True, exist_ok=True)
    filename = f"recovery_{args.new_boot.split('-')[0]}_{args.action}.json"
    (LOCAL/filename).write_bytes(encoded)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
