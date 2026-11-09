"""Read back committed spoken shards and download only a small review."""
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys
from colab_runtime import ColabRuntime
from factory_spoken_session import select_owned_a_session

LOCAL=Path('outputs/factory_spoken_campaign_v10')


def main():
    assert shutil.disk_usage(Path.cwd()).free>=10*2**30+5*2**20
    authorization=json.loads((LOCAL/'task_state.json').read_text(encoding='utf-8'))
    i,session,selection=select_owned_a_session(authorization.get('actual_colab_boot_id',authorization['authorized_transition_boot']))
    code='''import json,pathlib,gzip,hashlib,base64,time,os,subprocess
drive=pathlib.Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
root=pathlib.Path('/content/factory_spoken_campaign_v10');out=drive/'factory_spoken_campaign_v9'
identity=json.loads((drive/'destination_identity.json').read_text());assert identity['nonce']=='9e618b0f2db442a6a9ee3ee18f1d7143' and identity['intended_path']==str(drive)
state=json.loads((out/'state.json').read_text());owner=json.loads((out/'owner.json').read_text());boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert owner['boot_id']==boot and owner['module']=='factory_spoken_campaign_v10' and not owner.get('stopped')
proc=pathlib.Path('/proc')/str(owner['pid']);raw=(proc/'stat').read_text();assert raw[raw.rfind(')')+2:].split()[19]==owner['start_ticks']
assert 'factory_spoken_campaign_v10.py' in (proc/'cmdline').read_bytes().decode()
value={'checked_unix':time.time(),'boot_id':boot,'owner':owner,'state':state,'status':json.loads((root/'status.json').read_text()),'verified_shards':[],'dialogues':[]}
policy=json.loads((out/'language_policy_v10.json').read_text());assert policy['active_languages']==['vi','en'] and policy['user_authorized']
assert state['active_languages']==['vi','en'] and state['language'] in ['vi','en']
idx=state['commit_index'];cp=out/'commits'/f'{idx:09d}.json';raw=cp.read_bytes();c=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==state['last_commit_sha256']
assert c['engine_version']==10 and idx>policy['baseline_commit']
assert c['after']['total_counts']==state['total_counts']
previous=(out/'commits'/f'{idx-1:09d}.json').read_bytes()
assert hashlib.sha256(previous).hexdigest()==c['previous_commit_sha256']
obj=(out/c['object']).read_bytes();assert hashlib.sha256(obj).hexdigest()==c['object_sha256']
latest=json.loads(gzip.decompress(obj));assert latest['version']==10 and latest['language'] in ['vi','en']
assert len(latest['dialogues'])==c['accepted']
assert sum(state['total_counts'][l] for l in ['vi','en'])>sum(policy['baseline_counts'][l] for l in ['vi','en'])
value['resumed_generation_verification']={'verified':True,'engine_version':10,'baseline_commit':policy['baseline_commit'],
 'baseline_counts':policy['baseline_counts'],'latest_commit':idx,'latest_commit_sha256':state['last_commit_sha256'],
 'previous_link_verified':True,'object_sha256':c['object_sha256'],'object_readback_verified':True,
 'accepted_new':sum(state['total_counts'][l]-policy['baseline_counts'][l] for l in ['vi','en'])}
for lang in ['vi','en']:
 if not state['total_counts'].get(lang):continue
 paths=sorted((p for p in (out/lang).glob('round_*/shards/*.json.gz') if int(p.name.split('_')[0])<=state['commit_index'] and (out/'commits'/(p.name.split('_')[0]+'.json')).exists()),key=lambda p:p.name)
 if not paths:continue
 p=paths[-1];idx=int(p.name.split('_')[0]);commit_path=out/'commits'/f'{idx:09d}.json';raw_commit=commit_path.read_bytes();commit=json.loads(raw_commit)
 b=p.read_bytes();actual=hashlib.sha256(b).hexdigest();assert actual==commit['object_sha256'] and commit['object_readback_verified']
 payload=json.loads(gzip.decompress(b));assert payload['version'] in (9,10) and payload['register']=='spoken_transcript' and len(payload['dialogues'])==commit['accepted']
 value['verified_shards'].append({'index':idx,'engine_version':payload['version'],'language':lang,'accepted':commit['accepted'],'rejected':commit['rejected'],'rejection_reasons':dict(__import__('collections').Counter(r['reason'] for r in payload['rejected'])),'object_sha256':actual,'commit_sha256':hashlib.sha256(raw_commit).hexdigest(),'readback_verified':True})
 value['dialogues'].extend(payload['dialogues'][:3])
value['gpu']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
data=json.dumps(value,ensure_ascii=False).encode();assert len(data)<3*2**20
print(json.dumps({'base64':base64.b64encode(data).decode(),'sha256':hashlib.sha256(data).hexdigest()}))'''
    rt=ColabRuntime(server_index=i,session_path=session['path']);rt.kernel_id=session['kernel']['id']
    try:r=rt.execute(code,timeout=90,echo=False)
    finally:rt.close()
    assert r['status']=='ok',str([x.get('evalue') for x in r['outputs'] if x['output_type']=='error'])
    text=''.join(x.get('text','') for x in r['outputs'] if x['output_type']=='stream');packet=json.loads(text)
    data=base64.b64decode(packet['base64']);assert hashlib.sha256(data).hexdigest()==packet['sha256'];v=json.loads(data);v['session_selection']=selection
    data=json.dumps(v,ensure_ascii=False,indent=2).encode('utf-8')
    cards=[]
    for d in v['dialogues']:
        turns=''.join('<section><b>'+escape(t['speaker'])+'</b><p>'+escape(' '.join(s['display'] for s in t['sentences']))+'</p></section>' for t in d['turns'])
        cards.append('<article data-lang="'+d['language']+'"><h2>'+escape(d['language'].upper()+' · '+d['dialogue_id'])+'</h2>'+turns+'</article>')
    counts=v['state']['total_counts'];heading=' · '.join(l.upper()+': '+str(counts[l]) for l in ['vi','en'])
    html='''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spoken text · bộ sinh lại Việt Anh</title><style>body{margin:0;background:#f3f5f8;color:#193040;font:16px/1.7 system-ui}main{max-width:1000px;margin:30px auto;padding:0 18px}h1{font-size:29px;line-height:1.3}h2{font-size:19px}article,.intro{background:white;border:1px solid #dce4ec;border-radius:12px;padding:20px;margin:18px 0}article section{padding:8px 14px;margin:10px 0;background:#f7f9fc;border-radius:8px}p{margin:6px 0}select{font:inherit;padding:7px}.toolbar{position:sticky;top:0;background:#f3f5f8ee;padding:12px}small{color:#667c8c}article[hidden]{display:none}</style></head><body><main><h1>Spoken text: sinh lại Việt · Anh</h1><div class="intro"><p><b>'''+escape(heading)+'''</b> · chỉ đếm hội thoại đã được chấp nhận và commit.</p><p>Mỗi lượt là một đoạn lời nói dài. Có nối ý, hỏi lại hoặc tự sửa hợp lý; số, đơn vị và mã viết theo cách đọc. Không yêu cầu mọi câu đều có từ đệm.</p><p>Đây là lời thoại tổng hợp theo phong cách nói, chưa được con người duyệt và không phải bản ghi âm thật. Chưa đưa vào train. Dấu câu nguồn chỉ làm nhãn: dấu chấm EOT, phẩy/chấm phẩy nội bộ NEOT; input chữ thường không dấu câu, nhìn trước một hoặc hai từ trong cùng đoạn người nói.</p><small>Bản xem gồm tối đa ba hội thoại từ shard mới nhất mỗi tiếng đã có dữ liệu. Số lượng là ảnh chụp tại lúc thu, chiến dịch tiếp tục chạy.</small></div><div class="toolbar"><label>Ngôn ngữ <select id="language"><option value="">Tất cả</option><option value="vi">Việt</option><option value="en">Anh</option></select></label></div>'''+''.join(cards)+'''</main><script>const l=document.querySelector('#language');l.onchange=()=>document.querySelectorAll('article').forEach(e=>e.hidden=!!l.value&&e.dataset.lang!==l.value);</script></body></html>'''
    LOCAL.mkdir(parents=True,exist_ok=True);files={'review_snapshot.json':data,'review.html':html.encode('utf-8')}
    assert sum(map(len,files.values()))<5*2**20
    for n,b in files.items():(LOCAL/n).write_bytes(b);assert (LOCAL/n).read_bytes()==b
    receipt={n:{'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in files.items()}
    (LOCAL/'review_transfer_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'state':v['state'],'resumed_generation_verification':v['resumed_generation_verification'],'verified_shards':v['verified_shards'],'gpu':v['gpu'],'review':str((LOCAL/'review.html').resolve())},ensure_ascii=False))


if __name__=='__main__':main()
