"""Collect only small status/results/review samples, not models or full traces."""
import json,pathlib,subprocess,shutil
from html import escape
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/translation_turn_test_20261105_v1'
def main():
 assert shutil.disk_usage(ROOT).free>10*2**30+5*2**20
 code=r'''
import pathlib,json,hashlib,subprocess
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');out=r/'outputs/translation_turn_test_20261105_v1';v={'files':{}}
for n in ['launcher.json','status.json','error.json','report.json','cpu_demo.json','gpu_cache_demo_parity.json','completion_verification.json']:
 p=out/n
 if p.exists():
  assert p.stat().st_size<2**20
  v['files'][n]=json.loads(p.read_text())
l=v['files']['launcher.json'];proc=pathlib.Path('/proc')/str(l['pid']);v['process']={'present':proc.exists()}
if proc.exists():
 raw=(proc/'stat').read_text();f=raw[raw.rfind(')')+2:].split();cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();v['process'].update(pid=l['pid'],start_ticks=f[19],cmdline=cmd,identity_match=f[19]==l['start_ticks'] and l['module'] in cmd)
v['main_status']=json.loads((r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4/status.json').read_text())
v['gpu']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
if (out/'report.json').exists():
 import sacrebleu
 records=json.loads((out/'records.json').read_text());groups={}
 for a in ['whole_written','whole_asr','detector_asr']:groups[a]={x['id']:x for x in records if x['arm']==a}
 cases=[]
 for i,a in groups['whole_asr'].items():
  b=groups['detector_asr'][i];delta=sacrebleu.sentence_chrf(b['hypothesis'],[b['reference']]).score-sacrebleu.sentence_chrf(a['hypothesis'],[a['reference']]).score
  cases.append(dict(id=i,src_lang=a['src_lang'],source=a['source'],reference=a['reference'],whole=a['hypothesis'],segmented=b['hypothesis'],chunks=b['chunks'],chunk_outputs=b['chunk_outputs'],delta_chrf=delta))
 v['files']['review_cases.json']=sorted(cases,key=lambda x:x['delta_chrf'])[:10]+sorted(cases,key=lambda x:x['delta_chrf'],reverse=True)[:5]
print(json.dumps(v,ensure_ascii=False))
'''
 command='cd /home/edabk/hoangbpm/turn_detect && PYTHONPATH=outputs/multilingual_tiny_compression_20261102_v1/runtime_deps .venv/bin/python -'
 p=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',command],input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=45);v=json.loads(p.stdout)
 OUT.mkdir(exist_ok=True)
 for n,x in v['files'].items():(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 (OUT/'heartbeat_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 state=OUT/'task_state.json';s=json.loads(state.read_text()) if state.exists() else {'reported':False}
 s.update(phase=v['files']['status.json']['stage'],process=v['process']);state.write_text(json.dumps(s,indent=2)+'\n')
 if 'report.json' in v['files']:
  report=v['files']['report.json'];rows=''
  for a,groups in report['metrics'].items():
   for group,m in groups.items():rows+='<tr><td>'+a+'</td><td>'+group+'</td><td>'+str(m['n'])+'</td><td>'+f"{m['chrF']:.2f}"+'</td><td>'+f"{m['BLEU']:.2f}"+'</td></tr>'
  cards=''.join('<article><b>'+escape(x['id'])+'</b><p>Nguồn: '+escape(x['source'])+'</p><p>Tham chiếu: '+escape(x['reference'])+'</p><p>Dịch nguyên message: '+escape(x['whole'])+'</p><p>Chia đoạn: '+escape(' | '.join(x['chunks']))+'</p><p>Dịch sau chia: '+escape(x['segmented'])+'</p></article>' for x in v['files']['review_cases.json'])
  html='<!doctype html><html lang=vi><meta charset=utf-8><title>Turn detection cho dịch</title><style>body{font:15px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 18px;background:#f5f7fb}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}article{background:white;padding:18px;margin:14px 0}</style><h1>Ảnh hưởng chia đoạn lên bản dịch</h1><p>Qwen3 checkpoint15640,template B,FP32 GPU greedy. 310 message của eval_b trong37 nhóm,70 tele/240 MT; ngữ cảnh MT được ghép phục vụ benchmark. Chỉ Việt/Anh. So sánh whole_asr với detector_asr để đo ảnh hưởng chia đoạn trên cùng nguồn và refs. whole_written là điều kiện giữ dấu câu.</p><p>Turn detector11,94M frozen1945080,t.1,lookahead2 normalized words,cuts chỉ ở ranh giới từ ASR. Translator không được nhìn các từ future chưa thuộc đoạn. Context chỉ gồm nguồn đã xử lý; không có reference trong prompt. Cuối mỗi message được forced flush theo harness: đây là kiểm tra ảnh hưởng cắt nội bộ message, chưa đo chốt lượt nói/latency audio thực tế.</p><table><tr><th>Điều kiện</th><th>Nhóm</th><th>N</th><th>chrF</th><th>BLEU</th></tr>'+rows+'</table><h2>Ví dụ giảm/tăng chrF nhiều nhất</h2><p>Ví dụ chọn để minh họa, không phải mẫu ngẫu nhiên. Tham chiếu có thể chứa lỗi, chưa audit độc lập mọi nhãn dịch.</p>'+cards+'</html>'
  (OUT/'review.html').write_text(html,encoding='utf-8')
 print(json.dumps({'status':v['files']['status.json'],'error':v['files'].get('error.json'),'process':v['process'],'gpu':v['gpu'],'metrics':v['files'].get('report.json',{}).get('metrics')},ensure_ascii=False))
if __name__=='__main__':main()
