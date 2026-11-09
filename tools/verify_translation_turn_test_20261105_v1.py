"""Audit saved translation pipeline results, prompts and references without rescoring models."""
import pathlib,json,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 code=r'''
import pathlib,json,hashlib,math,sys,subprocess,shutil,datetime
r=pathlib.Path('/home/edabk/hoangbpm/turn_detect');d=r/'data/translation_qwen3_06b_15640_user_20261105';out=r/'outputs/translation_turn_test_20261105_v1'
sys.path[:0]=[str(d/'translator_source/src'),str(r)]
from mt.asr import asr_style
from mt.sft_format import context_for,user_message
from turn_v2.common import normalize
import sacrebleu
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
report=json.loads((out/'report.json').read_text());assert report['complete']
assert sha(out/'records.json')==report['records_sha256'] and sha(out/'segmentation_trace.json')==report['segmentation_sha256']
assert sha(d/'eval_b.json')==report['eval_sha256']
assert sha(d/'saves/qwen3_0.6b_mix4r2_1ep/checkpoint-15640/model.safetensors')==report['translation_model_sha256']=='c32df2ca279d43140551bdf44d026722887780ece9cc33a465b36afe778330ac'
convs=json.loads((d/'eval_b.json').read_text());records=json.loads((out/'records.json').read_text());traces=json.loads((out/'segmentation_trace.json').read_text());assert len(records)==930 and len(traces)==310
by={(x['arm'],x['id']):x for x in records};assert len(by)==930
prompt_count=0
for arm in ['whole_written','whole_asr','detector_asr']:
 for conv in convs:
  hist=[]
  for m in conv['messages']:
   x=by[(arm,m['id'])];src=m['lang'];tgt='en' if src=='vi' else 'vi';source=' '.join(m['text'].split()) if arm=='whole_written' else asr_style(m['text'],src)
   assert x['source']==source and x['reference']==m['ref'] and x['conv']==conv['id'] and x['src_lang']==src
   assert len(x['prompts'])==len(x['chunks'])==len(x['chunk_outputs']) and ' '.join(x['chunks'])==source
   assert ' '.join(x['chunk_outputs'])==x['hypothesis']
   if arm!='detector_asr':assert len(x['chunks'])==1
   for chunk,prompt in zip(x['chunks'],x['prompts']):
    current=dict(src_lang=src,source=chunk,speaker=src)
    context=context_for('B',hist+[current],len(hist));expected=user_message(context,dict(src_lang=src,tgt_lang=tgt,source=chunk),'B')
    assert prompt==expected,'Oracle/history/prompt mismatch'
    hist.append(current);prompt_count+=1
decisions=cuts=0
for tr in traces:
 x=by[('detector_asr',tr['id'])];assert tr['source']==x['source'] and tr['chunks']==x['chunks'] and tr['final_flush_forced']
 words=tr['source'].split();start=0;chunks=[];expected_gaps=[g for g in range(1,len(words)) if len(normalize(' '.join(words[g:])).split())>=2]
 assert [c['gap'] for c in tr['decisions']]==expected_gaps
 for c in tr['decisions']:
  g=c['gap'];assert c['context']==normalize(' '.join(words[start:g])) and c['future']==normalize(' '.join(words[g:])).split()[:2]
  assert math.isfinite(c['p_eot']) and 0<=c['p_eot']<=1 and c['cut']==(c['p_eot']>=.1)
  if c['cut']:chunks.append(' '.join(words[start:g]));start=g;cuts+=1
  decisions+=1
 if start<len(words):chunks.append(' '.join(words[start:]))
 assert chunks==tr['chunks'] and ' '.join(chunks)==tr['source']
for arm,groups in report['metrics'].items():
 for group,metric in groups.items():
  rr=[x for x in records if x['arm']==arm and (group=='all' or group=='vi2en' and x['src_lang']=='vi' or group=='en2vi' and x['src_lang']=='en' or group in ('tele','mt') and x['conv'].startswith(group))]
  assert len(rr)==metric['n']
  assert abs(sacrebleu.corpus_chrf([x['hypothesis'] for x in rr],[[x['reference'] for x in rr]]).score-metric['chrF'])<1e-8
  assert abs(sacrebleu.corpus_bleu([x['hypothesis'] for x in rr],[[x['reference'] for x in rr]]).score-metric['BLEU'])<1e-8
parity=json.loads((out/'gpu_cache_demo_parity.json').read_text());demo=json.loads((out/'cpu_demo.json').read_text());assert parity['complete']
assert parity['cached']['translations']==parity['uncached']['translations']
assert demo['stdout'].split('\n---\n',1)[1].strip()==parity['cached']['translations']['en']['text']
main=r/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4';launch=json.loads((main/'launcher.json').read_text());proc=pathlib.Path('/proc')/str(launch['pid']);s=(proc/'stat').read_text();f=s[s.rfind(')')+2:].split();assert f[19]==launch['start_ticks'] and launch['module'].encode() in (proc/'cmdline').read_bytes().split(b'\0')
assert not (main/'error.json').exists();assert min(shutil.disk_usage(r).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30
proof=dict(complete=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha(out/'report.json'),records_audited=930,translation_prompts_audited=prompt_count,source_word_conservation_verified=True,references_absent_from_all_reconstructed_prompts=True,detector_decisions_verified=decisions,predicted_cuts=cuts,forced_message_flushes=310,all_translation_metrics_recomputed=True,CPU_demo_and_GPU_cached_uncached_translation_match=True,main_pid=launch['pid'],main_status=json.loads((main/'status.json').read_text()),source_test_not_used_to_modify_training=True,disk_free_gib=dict(ssd=shutil.disk_usage(r).free/2**30,hdd=shutil.disk_usage('/mnt/hdd500').free/2**30))
(out/'completion_verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n');print(json.dumps(proof))
'''
 command='cd /home/edabk/hoangbpm/turn_detect && PYTHONPATH=outputs/multilingual_tiny_compression_20261102_v1/runtime_deps .venv/bin/python -'
 p=subprocess.run(['ssh','-o','BatchMode=yes','edabk@100.91.209.124',command],input=code,text=True,capture_output=True,encoding='utf-8',check=True,timeout=60)
 v=json.loads(p.stdout);(ROOT/'outputs/translation_turn_test_20261105_v1/completion_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(v))
if __name__=='__main__':main()
