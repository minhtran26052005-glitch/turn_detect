"""Frozen MT + turn-detector evaluation, without changing the running trainer."""
import os,sys,json,time,gc,hashlib,subprocess,threading,fcntl,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/edabk/hoangbpm/turn_detect')
DATA=ROOT/'data/translation_qwen3_06b_15640_user_20261105'
OUT=ROOT/'outputs/translation_turn_test_20261105_v1'
MODEL=DATA/'saves/qwen3_0.6b_mix4r2_1ep/checkpoint-15640'
SRC=DATA/'translator_source'
sys.path.insert(0,str(SRC/'src'))

def save(n,v):
 p=OUT/n;tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(stage,**kw):save('status.json',dict(stage=stage,at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**kw))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def run():
 import torch,numpy as np,sacrebleu
 import transformers.utils as u
 import transformers.utils.import_utils as im
 u.is_torchvision_available=im.is_torchvision_available=lambda:False
 u.is_librosa_available=im.is_librosa_available=lambda:False
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.32)
 assert not (OUT/'report.json').exists()
 assert shutil.disk_usage(ROOT).free>13*2**30
 status('CPU_EXACT_CLI_DEMO')
 args=[str(SRC/'scripts/demo/translate_cli.py'),'--model',str(MODEL),'--template','B','--device','cpu','--cpu-threads','2','--show-prompt','--style','written','--context','en: hi there, how do you do','--lang','en','vi: mình ổn nhé, còn bạn?']
 wrapper="import sys,runpy;import transformers.utils as u;import transformers.utils.import_utils as i;u.is_torchvision_available=i.is_torchvision_available=lambda:False;u.is_librosa_available=i.is_librosa_available=lambda:False;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
 t0=time.time();p=subprocess.run([sys.executable,'-c',wrapper,*args],env=env,capture_output=True,text=True,timeout=600)
 (OUT/'cpu_demo_stdout.txt').write_text(p.stdout);(OUT/'cpu_demo_stderr.txt').write_text(p.stderr)
 assert p.returncode==0,p.stderr[-2000:]
 save('cpu_demo.json',dict(args=args,stdout=p.stdout,wall_s=time.time()-t0))
 status('LOADING_GPU_TRANSLATOR')
 from transformers import AutoModelForCausalLM,AutoTokenizer,AlbertConfig
 from mt.translator import Translator,interpret,piece
 from mt.sft_format import STOP
 from mt.asr import asr_style
 from turn_v2.common import normalize
 from turn_v2.malbert_turn_student_v1 import MalbertTurnStudent
 from turn_v2.uploaded_vi_en_evaluation_v4 import predictor
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,token=False);tok.padding_side='left'
 tr=Translator.__new__(Translator);tr.torch=torch;tr.device='cuda';tr.tok=tok
 tr.model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,local_files_only=True,token=False).cuda().eval()
 sid=tok.convert_tokens_to_ids(STOP);tr.eos=[sid]+([tok.eos_token_id] if tok.eos_token_id not in (None,sid) else [])
 tr.max_new_tokens=512;tr.params=sum(p.numel() for p in tr.model.parameters());tr.model_path=str(MODEL);tr.lock=threading.Lock()
 hist=[dict(speaker='en',src_lang='en',source='hi there, how do you do',translations={})]
 tr.model.generation_config.use_cache=False
 a=interpret(tr,'B',['vi','en'],hist,'vi','vi','mình ổn nhé, còn bạn?',normalize=False)
 tr.model.generation_config.use_cache=True
 b=interpret(tr,'B',['vi','en'],hist,'vi','vi','mình ổn nhé, còn bạn?',normalize=False)
 assert a['translations']['en']==b['translations']['en'],'Cache parity failed on demo'
 save('gpu_cache_demo_parity.json',dict(complete=True,uncached=a,cached=b,model_parameters=tr.params))
 mt=ROOT/'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
 pin=ROOT/'outputs/malbert_uploaded_test_20261105_v1/frozen/weights.pt'
 assert sha(pin)=='3f38a0bddb857b8941a2a0bf47628a41b4003e270a19a0055c492e7041ed3c88'
 dtok=AutoTokenizer.from_pretrained(mt/'tokenizer',local_files_only=True,token=False)
 detector=MalbertTurnStudent(AlbertConfig.from_pretrained(mt/'pretrained',local_files_only=True,token=False),sep_token_id=3)
 detector.load_state_dict(torch.load(pin,map_location='cpu',weights_only=True),strict=True);detector.cuda().eval();dp=predictor(detector,dtok)
 convs=json.loads((DATA/'eval_b.json').read_text());assert sum(len(c['messages']) for c in convs)==310
 recs=[];segment_traces=[]
 for arm in ['whole_written','whole_asr','detector_asr']:
  status('TRANSLATING_TEST',arm=arm,completed=0,total=310);n=0
  for conv in convs:
   history=[]
   for m in conv['messages']:
    src=m['lang'];assert src in ('vi','en');tgt='en' if src=='vi' else 'vi'
    source=' '.join(m['text'].split()) if arm=='whole_written' else asr_style(m['text'],src)
    chunks=[source];decisions=[]
    if arm=='detector_asr':
     words=source.split();start=0;chunks=[]
     # Preserve ASR word atoms (e.g. I'm); detector normalization can split punctuation internally.
     for gap in range(1,len(words)):
      future=normalize(' '.join(words[gap:])).split()[:2]
      if len(future)<2:continue
      context=normalize(' '.join(words[start:gap]));probs,_=dp([dict(context=context,text=' '.join(future))]);cut=probs[0]>=.1
      decisions.append(dict(gap=gap,context=context,future=future,p_eot=probs[0],cut=cut))
      if cut:chunks.append(' '.join(words[start:gap]));start=gap
     if start<len(words):chunks.append(' '.join(words[start:]))
     assert ' '.join(chunks)==source
     segment_traces.append(dict(conv=conv['id'],id=m['id'],source=source,chunks=chunks,decisions=decisions,final_flush_forced=True))
    outputs=[];ms=0;tokens=0;prompts=[]
    for text in chunks:
     result=interpret(tr,'B',[src,tgt],history,src,src,text,normalize=False)
     history.append(piece(result,src,src));outputs.append(result['translations'][tgt]['text']);ms+=result['ms'];tokens+=result['out_tokens'];prompts.append(result['translations'][tgt]['prompt'])
    recs.append(dict(arm=arm,conv=conv['id'],id=m['id'],src_lang=src,source=source,reference=m['ref'],hypothesis=' '.join(outputs),chunks=chunks,chunk_outputs=outputs,ms=ms,out_tokens=tokens,prompts=prompts))
    n+=1
    if n%10==0:
     status('TRANSLATING_TEST',arm=arm,completed=n,total=310);save('partial_records.json',recs);save('segmentation_trace.json',segment_traces)
  save('partial_records.json',recs);save('segmentation_trace.json',segment_traces)
 metrics={}
 for arm in ['whole_written','whole_asr','detector_asr']:
  metrics[arm]={}
  for group in ['all','vi2en','en2vi','tele','mt']:
   rr=[r for r in recs if r['arm']==arm and (group=='all' or group=='vi2en' and r['src_lang']=='vi' or group=='en2vi' and r['src_lang']=='en' or group in ('tele','mt') and r['conv'].startswith(group))]
   metrics[arm][group]=dict(n=len(rr),chrF=sacrebleu.corpus_chrf([r['hypothesis'] for r in rr],[[r['reference'] for r in rr]]).score,BLEU=sacrebleu.corpus_bleu([r['hypothesis'] for r in rr],[[r['reference'] for r in rr]]).score)
 assert len(recs)==930 and len(segment_traces)==310
 save('records.json',recs);save('segmentation_trace.json',segment_traces)
 report=dict(complete=True,metrics=metrics,model_parameters=tr.params,translation_model_sha256=sha(MODEL/'model.safetensors'),eval_sha256=sha(DATA/'eval_b.json'),records_sha256=sha(OUT/'records.json'),segmentation_sha256=sha(OUT/'segmentation_trace.json'),turn_detector_checkpoint=1945080,turn_detector_threshold=.1,translation_template='B',compute='FP32 GPU, greedy, KV cache enabled; CPU exact CLI demo separate',reference_never_in_prompt=True,test_used_for_training_or_threshold_selection=False,forced_flush_at_each_original_message=True,scope='Measures within-message segmentation impact, not real-time end-of-turn delay across messages',segmentation_chunks=sum(len(x['chunks']) for x in segment_traces),baseline_messages=310,source_word_conservation_all_passed=True,sacrebleu_version=sacrebleu.__version__,main_training_unchanged=True)
 save('report.json',report);status('COMPLETE',metrics=metrics)

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True);os.nice(10)
 with (OUT/'EVALUATOR_LOCK').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:run()
  except Exception as e:save('error.json',dict(error=repr(e),traceback=traceback.format_exc()));status('ERROR',error=repr(e));raise
