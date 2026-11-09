"""Durable spoken transcript generation: 10k VI, 10k EN, 10k ZH, repeat until stopped."""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import threading
import time
import traceback
import unicodedata
import uuid

from factory_spoken_spec_v1 import SYSTEM, task_for, quality_check, parse_generated, make_examples, normalize, advance

MODEL='Qwen/Qwen3-8B'
REV='b968826d9c46dd6066d109eabc6255188de91218'
ROOT=Path('/content/factory_spoken_campaign_v9')
DRIVE=Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
OUT=DRIVE/'factory_spoken_campaign_v9'
NONCE='9e618b0f2db442a6a9ee3ee18f1d7143'
LANGS=('vi','en','zh')
QUOTA=10000
STOP=False


def sha_bytes(b):return hashlib.sha256(b).hexdigest()


def json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n').encode()


def write_atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    data=json_bytes(value)
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def write_verified(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    digest=sha_bytes(data)
    if path.exists():
        assert sha_bytes(path.read_bytes())==digest,'Refuse to overwrite a different committed object'
        return digest
    tmp=path.with_name(path.name+'.part-'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    assert sha_bytes(tmp.read_bytes())==digest,'Drive readback SHA mismatch'
    os.replace(tmp,path)
    assert sha_bytes(path.read_bytes())==digest,'Drive final readback SHA mismatch'
    return digest


def assert_drive():
    if not os.path.ismount('/content/drive'):raise RuntimeError('Drive mount unavailable')
    identity=json.loads((DRIVE/'destination_identity.json').read_text())
    if identity.get('nonce')!=NONCE or identity.get('intended_path')!=str(DRIVE):
        raise RuntimeError('Drive identity mismatch; never recreate its nonce')


def stopped():return STOP or (OUT/'STOP_REQUESTED').exists() or (ROOT/'STOP_REQUESTED').exists()


def stop(*_):
    global STOP
    STOP=True


def signature(dialogue):
    text=' '.join(s['normalized'] for t in dialogue['turns'] for s in t['sentences'])
    exact=sha_bytes(text.encode())
    tokens=re.findall(r'[^\W_]+',re.sub(r'\d+(?:[.,]\d+)?','#',text),re.UNICODE)
    grams=Counter(' '.join(tokens[i:i+3]) for i in range(max(0,len(tokens)-2)))
    vector=[0]*64
    for gram,count in grams.items():
        number=int.from_bytes(hashlib.blake2b(gram.encode(),digest_size=8).digest(),'big')
        for bit in range(64):vector[bit]+=(1 if (number>>bit)&1 else -1)*count
    sim=sum(1<<bit for bit,score in enumerate(vector) if score>=0)
    return exact,sim


class DuplicateIndex:
    def __init__(self):self.exact=set();self.bands=defaultdict(list)
    def duplicate(self,lang,exact,sim):
        if (lang,exact) in self.exact:return 'exact_duplicate'
        candidates=set()
        for band in range(4):candidates.update(self.bands[(lang,band,(sim>>(16*band))&65535)])
        if any((other^sim).bit_count()<=3 for other in candidates):return 'near_duplicate_simhash'
        return None
    def add(self,lang,exact,sim):
        self.exact.add((lang,exact))
        for band in range(4):self.bands[(lang,band,(sim>>(16*band))&65535)].append(sim)


def recover():
    state={'schema':9,'cycle':1,'language':'vi','cycle_counts':{'vi':0,'en':0,'zh':0},
       'total_counts':{'vi':0,'en':0,'zh':0},'attempts':{'vi':0,'en':0,'zh':0},'rejected':{'vi':0,'en':0,'zh':0},
       'example_counts':{'vi':0,'en':0,'zh':0},'commit_index':0,'last_commit_sha256':None,'oom_retries':0,'batch_size':32}
    index=DuplicateIndex();previous=None
    for path in sorted((OUT/'commits').glob('*.json')):
        raw=path.read_bytes();commit=json.loads(raw)
        assert commit['index']==state['commit_index']+1 and commit['previous_commit_sha256']==previous,'Broken commit chain'
        assert commit['object_readback_verified'] and (OUT/commit['object']).stat().st_size==commit['bytes']
        state=commit['after'];previous=sha_bytes(raw);state['last_commit_sha256']=previous
        for item in commit['signatures']:index.add(item['language'],item['exact'],int(item['simhash'],16))
    # Read back the most recent data object before resuming; older objects retain
    # their individual SHA readback receipts from their original commit.
    if state['commit_index']:
        last=json.loads((OUT/'commits'/f"{state['commit_index']:09d}.json").read_bytes())
        assert sha_bytes((OUT/last['object']).read_bytes())==last['object_sha256']
    return state,index


def restore_pending():
    paths=sorted((ROOT/'pending').glob('*/commit.json'),key=lambda p:json.loads(p.read_bytes())['index'])
    for marker in paths:
        commit_data=marker.read_bytes();commit=json.loads(commit_data)
        target=OUT/'commits'/f"{commit['index']:09d}.json"
        if target.exists():
            assert sha_bytes(target.read_bytes())==sha_bytes(commit_data)
            continue
        data=(marker.parent/'object.json.gz').read_bytes()
        assert sha_bytes(data)==commit['object_sha256']
        write_verified(OUT/commit['object'],data);write_verified(target,commit_data)


def run():
    import shutil
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    ROOT.mkdir(parents=True,exist_ok=True);assert_drive();OUT.mkdir(exist_ok=True)
    config={'version':9,'model':MODEL,'revision':REV,'order':list(LANGS),'accepted_per_language_per_cycle':QUOTA,
       'repeat_until_user_stop':True,'turns':'four alternating substantial spoken paragraphs, 2-8 sentences each','inputs':'one language per dialogue',
       'labels':'full stop EOT; internal comma/semicolon NEOT; 1/2 word same-speaker lookahead',
       'source':'Qwen synthetic spoken-style transcripts, punctuation-derived, unreviewed, not human/audio gold','register':'spoken_transcript','numbers_units_codes':'written as spoken, no digit numerals','auto_train':False}
    if (OUT/'config.json').exists():assert json.loads((OUT/'config.json').read_text())==config,'Config mismatch'
    else:write_verified(OUT/'config.json',json_bytes(config))
    owner={'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'pid':os.getpid(),
      'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],'module':'factory_spoken_campaign_v9'}
    old=json.loads((OUT/'owner.json').read_text()) if (OUT/'owner.json').exists() else None
    if old and not old.get('stopped'):
        if old['boot_id']!=owner['boot_id']:raise RuntimeError('Previous runtime owner unknown; verify it before recovery')
        p=Path('/proc')/str(old['pid'])
        if p.exists() and (p/'stat').read_text().split()[21]==old['start_ticks']:raise RuntimeError('Existing campaign owner still running')
    write_atomic(OUT/'owner.json',owner)
    restore_pending()
    state,duplicates=recover();current={};mutex=threading.Lock();done=threading.Event()
    def update(stage,**extra):
        with mutex:
            current.clear();current.update({'stage':stage,'pid':os.getpid(),'start_ticks':owner['start_ticks'],
              'engine_version':9,'at_unix':time.time(),'state':state,'gpu_peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,2) if torch.cuda.is_initialized() else 0,**extra})
            write_atomic(ROOT/'status.json',current)
    def heartbeat():
        while not done.wait(20):
            with mutex:
                if current:current['at_unix']=time.time();write_atomic(ROOT/'status.json',current)
    threading.Thread(target=heartbeat,daemon=True).start()
    update('LOADING_QWEN')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REV,token=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,token=False,dtype=torch.bfloat16,device_map='auto').eval()
    props=torch.cuda.get_device_properties(0);active_batch=state.get('batch_size',32)
    while not stopped():
        if shutil.disk_usage('/content').free<25*2**30:
            update('WAITING_FOR_COLAB_DISK_RESERVE');time.sleep(30);continue
        try:assert_drive()
        except Exception as e:update('WAITING_FOR_DRIVE',error=repr(e));time.sleep(30);continue
        advance(state);lang=state['language'];needed=QUOTA-state['cycle_counts'][lang]
        tasks=[task_for(lang,state['cycle'],state['attempts'][lang]+i) for i in range(min(active_batch,needed))]
        prompts=[tokenizer.apply_chat_template([{'role':'system','content':SYSTEM[lang]},{'role':'user','content':t['prompt']}],
             tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in tasks]
        update('GENERATING',language=lang,batch=len(tasks),batch_started_unix=time.time())
        oom=False
        try:
            torch.manual_seed(tasks[0]['seed']%(2**63-1))
            inputs=tokenizer(prompts,padding=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=1800,do_sample=True,temperature=.9,top_p=.94,
                    repetition_penalty=1.03,pad_token_id=tokenizer.eos_token_id)
            suffix=generated[:,inputs['input_ids'].shape[1]:]
            finished=[bool((row==tokenizer.eos_token_id).any().item()) for row in suffix]
            texts=tokenizer.batch_decode(suffix,skip_special_tokens=True)
            del suffix
            del inputs,generated
        except torch.cuda.OutOfMemoryError:oom=True
        except Exception as exc:
            update('GENERATION_RETRY',error=repr(exc));time.sleep(20);continue
        if oom:
            if 'inputs' in locals():del inputs
            torch.cuda.empty_cache();state['oom_retries']+=1
            active_batch=max(1,active_batch//2);update('REDUCING_BATCH_AFTER_OOM',batch=active_batch);continue
        accepted=[];examples=[];rejects=[];signatures=[]
        for task,raw,is_finished in zip(tasks,texts,finished):
            try:
                if not is_finished:raise ValueError('token_limit_truncated')
                text=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip(),flags=re.I)
                dialogue=parse_generated(text,task);quality_check(dialogue,lang)
                exact,sim=signature(dialogue);reason=duplicates.duplicate(lang,exact,sim)
                if reason:raise ValueError(reason)
                rows=make_examples(dialogue)
                if {x['label'] for x in rows}!={'EOT','NEOT'}:raise ValueError('missing_EOT_or_NEOT')
                dialogue['generation_task']=task;dialogue['review_state']='UNREVIEWED';dialogue['training_eligible']=False
                for row in rows:row['split']='generated_unassigned';row['review_state']='UNREVIEWED';row['training_eligible']=False
                accepted.append(dialogue);examples.extend(rows);duplicates.add(lang,exact,sim)
                signatures.append({'language':lang,'exact':exact,'simhash':f'{sim:016x}'})
            except Exception as e:rejects.append({'id':task['id'],'reason':repr(e),'task':task,'raw':raw})
        after=json.loads(json.dumps(state));after['attempts'][lang]+=len(tasks);after['cycle_counts'][lang]+=len(accepted)
        after['total_counts'][lang]+=len(accepted);after['rejected'][lang]+=len(rejects);after['example_counts'][lang]+=len(examples)
        after['commit_index']+=1;after['batch_size']=active_batch
        payload={'version':9,'language':lang,'cycle':state['cycle'],'tasks':tasks,'dialogues':accepted,'examples':examples,'rejected':rejects,'register':'spoken_transcript','not_human_or_audio_gold':True}
        data=gzip.compress(json_bytes(payload),compresslevel=5,mtime=0)
        relative=f"{lang}/round_{state['cycle']:05d}/shards/{after['commit_index']:09d}_{uuid.uuid4().hex}.json.gz"
        object_sha=sha_bytes(data)
        commit={'index':after['commit_index'],'previous_commit_sha256':state['last_commit_sha256'],
          'object':relative,'bytes':len(data),'object_sha256':object_sha,'object_readback_verified':True,
          'accepted':len(accepted),'rejected':len(rejects),'examples':len(examples),'language':lang,
          'signatures':signatures,'engine_version':9,'after':after,'committed_unix':time.time()}
        commit_data=json_bytes(commit)
        # Retry precisely this commit; never regenerate or discard a batch on a
        # temporary Drive error. The commit manifest is the durable boundary.
        while True:
            try:
                assert_drive();write_verified(OUT/relative,data)
                commit_sha=write_verified(OUT/'commits'/f"{after['commit_index']:09d}.json",commit_data)
                after['last_commit_sha256']=commit_sha;write_atomic(OUT/'state.json',after)
                break
            except Exception as e:
                update('WAITING_FOR_DRIVE_COMMIT',error=repr(e),pending_accepted=len(accepted))
                if stopped():
                    # Preserve a pending transaction locally and release the GPU
                    # even if Drive is temporarily unavailable at stop time.
                    pending=ROOT/'pending'/uuid.uuid4().hex
                    write_verified(pending/'object.json.gz',data)
                    write_verified(pending/'commit.json',commit_data)
                    update('STOPPED_PENDING_DRIVE',pending_accepted=len(accepted))
                    owner['stopped']=True
                    try:write_atomic(OUT/'owner.json',owner)
                    except OSError:pass
                    done.set();return
                time.sleep(30)
        state=after
        update('BATCH_COMMITTED',last_accepted=len(accepted),last_rejected=len(rejects),last_reasons=dict(Counter(x['reason'] for x in rejects)))
        write_atomic(OUT/'status.json',current)
        # Use remaining memory for useful parallel generation. Leave headroom;
        # batch size can shrink again on OOM. Do not allocate dummy VRAM.
        used=torch.cuda.max_memory_reserved()/props.total_memory
        if used<.85 and active_batch<32 and len(accepted)>0:active_batch=min(32,active_batch+4)
        if len(accepted)==0 and after['attempts'][lang]>=5*active_batch:
            update('LOW_ACCEPTANCE_REVIEW_NEEDED',last_reasons=dict(Counter(x['reason'] for x in rejects)))
            time.sleep(10)
    update('STOPPED_BY_REQUEST');owner['stopped']=True;write_atomic(OUT/'owner.json',owner)
    write_atomic(OUT/'status.json',current);done.set()


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (ROOT/'WORKER_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:
            write_atomic(ROOT/'error.json',{'error':repr(e),'traceback':traceback.format_exc(),'at_unix':time.time()});raise

if __name__=='__main__':main()
