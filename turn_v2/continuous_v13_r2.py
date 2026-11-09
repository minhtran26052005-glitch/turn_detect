"""Resumable Colab campaign with no fixed cycle count or plateau stop."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import traceback
import zipfile

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .auto_rl_v11 import PROBE,prefixes
from .auto_rl_v12 import GeneratedJudge,MODEL,REV
from .common import LABEL2ID,digest,file_hash,input_id,normalize,read_jsonl,write_jsonl
from .continuous_v13_state import Store,atomic_json,singleton
from .reward_v10_train import encode,infer
from .reward_v10_objective import actor_loss
from .metrics import classification_metrics,calibrate_on_dev
from .tiny_student import load_student

LANGS=['vi','en','zh'];RUN=Path('outputs/continuous_v13_20261026');DATA=Path('data_v10/reward_20261025')
TOPICS={'vi':['Khoa học','Năng lượng','Nước','Đại dương','Giáo dục','Văn hóa','Sức khỏe','Kiến trúc','Nông nghiệp','Máy tính','Thể thao','Địa lý','Kinh tế','Giao thông','Y học','Lịch sử'],
    'en':['Science','Energy','Water','Ocean','Education','Culture','Health','Architecture','Agriculture','Computer','Sport','Geography','Economics','Transport','Medicine','History'],
    'zh':['科学','能源','水','海洋','教育','文化','健康','建筑','农业','计算机','体育','地理学','经济学','交通','医学','历史']}
STOP=False


def request_stop(*args):
    global STOP;STOP=True


def status(stage,**values):
    atomic_json(RUN/'status.json',{'status':'RUNNING','stage':stage,'pid':os.getpid(),'heartbeat_utc':datetime.now(timezone.utc).isoformat(),
        'quality_target_certified':False,'continuous_until_user_stop':True,**values})


def visible_key(row):return row['language']+input_id(row)


def judged(rows,judge,store,split,cycle):
    # Score source/current text only, never old reference labels, suffixes,
    # student predictions or identifiers. The judge class constructs its JSON.
    audit=judge.score(rows) if rows else [];accepted=[]
    for row in audit:
        if row['judge_accepted']:
            record={**row,'observed_floor_reference':row.get('label'),'label':row['judge_label'],'label_source':'QWEN8B_SEMANTIC_TWO_VIEW_SILVER','is_gold':False,'scope':'SEMANTIC_COMPLETENESS_NOT_OBSERVED_FLOOR'}
            if store.add(visible_key(record),record,split,cycle):accepted.append(record)
    return audit,accepted


class Crawler:
    def __init__(self,store):
        self.store=store;self.http=requests.Session();self.http.headers['User-Agent']='TurnDetectorResearch/0.13 (bounded per-request crawl; persistent provenance and backoff)'
        self.last_request=0.

    def pages(self,language,cycle,pages=3,semantic_dev=False):
        now=time.time();blocked=self.store.get('wiki_backoff:'+language,0)
        if blocked>now:return [],[{'language':language,'retry_after_seconds':round(blocked-now)}]
        title_index=self.store.get('topic_cursor:'+language,0);titles=[]
        if title_index<len(TOPICS[language]):titles=[TOPICS[language][title_index]];self.store.put('topic_cursor:'+language,title_index+1)
        # After the finite topic queue, random article discovery supplies new
        # sources. The campaign is not limited to two lists of titles.
        queries=[{'titles':'|'.join(titles)}] if titles else []
        queries.append({'generator':'random','grnnamespace':0,'grnlimit':pages})
        sources=[];errors=[]
        for query in queries:
            time.sleep(max(0.,1.2-(time.time()-self.last_request)));self.last_request=time.time()
            try:
                response=self.http.get(f'https://{language}.wikipedia.org/w/api.php',params={'action':'query','format':'json','formatversion':2,'prop':'extracts|info|revisions','explaintext':1,'exintro':1,'inprop':'url','rvprop':'ids','redirects':1,**query},timeout=35)
                if response.status_code in [429,503]:
                    retry=response.headers.get('Retry-After','120');seconds=int(retry) if retry.isdigit() else 120
                    self.store.put('wiki_backoff:'+language,time.time()+min(1800,max(90,seconds)));errors.append({'language':language,'HTTP':response.status_code,'backoff_seconds':seconds});break
                response.raise_for_status();payload=response.json()
                for page in payload.get('query',{}).get('pages',[]):
                    if 'pageid' not in page or not page.get('extract'):continue
                    sid=language+'wiki:'+str(page['pageid'])
                    if self.store.source_seen(sid):continue
                    if not semantic_dev and self.store.db.execute('SELECT 1 FROM protected_sources WHERE source_id=?',(sid,)).fetchone():continue
                    source={'language':language,'source_id':sid,'title':page['title'],'url':page.get('fullurl'),'page_revision':page.get('revisions',[{}])[0].get('revid'),
                        'text':page['extract'],'sha256':digest(page['extract']),'license':'Wikipedia CC BY-SA terms; attribution preserved'}
                    if self.store.source(source,cycle):sources.append(source)
                    if len(sources)>=pages:break
            except (requests.RequestException,ValueError) as error:errors.append({'language':language,'error':type(error).__name__+':'+str(error)[:120]})
            if len(sources)>=pages:break
        return sources,errors


def metric(rows,probs,threshold=None):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);value=calibrate_on_dev(labels,probs) if threshold is None else classification_metrics(labels,probs,threshold)
    return {'metrics':value,'selection_score':value['joint_score'],'reference':'FROZEN_QWEN_SEMANTIC_SILVER_NOT_HUMAN_GOLD'}


def evaluate(model,tok,semantic,natural,threshold=None):
    x,l=encode(semantic,tok);prob=infer(model,x,l);score=metric(semantic,prob,threshold)
    nx,nl=encode(natural,tok);npred=infer(model,nx,nl)
    floor=classification_metrics(np.array([LABEL2ID[r['label']] for r in natural]),npred,score['metrics']['threshold'])
    return score,floor,prob


def bootstrap(store,judge,crawler):
    guard=json.loads(Path('data_v10/reward_20261025/auto_rl_guard.json').read_text())
    store.protect(guard['protected_inputs'],guard['protected_sources'])
    for lang in LANGS:
        for r in read_jsonl(DATA/lang/'dev.jsonl'):store.protect([visible_key(r)],[r['source_id']] if r.get('source_id') else [])
    if not store.get('probe_done'):
        probe=judge.score(PROBE);write_jsonl(RUN/'judge_probe.jsonl',probe)
        if not all(r['judge_label']==r['expected'] for r in probe):raise RuntimeError('Judge probe failed; preserve state for diagnosis')
        store.put('probe_done',True)
    for lang in LANGS:
        if store.get('semantic_dev_frozen:'+lang):continue
        status('FREEZING_ALIGNED_SEMANTIC_DEV',language=lang)
        # Re-judge the existing causal natural-dev text once under the semantic
        # definition; original floor references are retained separately.
        natural=[r for r in read_jsonl(DATA/lang/'dev.jsonl') if r['evaluation_slice']=='natural']
        audit,accepted=judged(natural,judge,store,'semantic_dev',0);write_jsonl(RUN/'semantic_dev'/(lang+'_natural_audit.jsonl'),audit)
        sources,errors=crawler.pages(lang,0,pages=3,semantic_dev=True)
        candidates=[r for source in sources for r in prefixes(source)]
        more,extra=judged(candidates,judge,store,'semantic_dev',0);write_jsonl(RUN/'semantic_dev'/(lang+'_prose_audit.jsonl'),more)
        semantic=store.rows(lang,'semantic_dev');counts=Counter(r['label'] for r in semantic)
        if min(counts.get('EOT',0),counts.get('NEOT',0))<5:raise RuntimeError('Semantic dev lacks both classes: '+lang)
        write_jsonl(RUN/'semantic_dev'/(lang+'.jsonl'),semantic)
        store.put('semantic_dev_frozen:'+lang,{'cases':len(semantic),'counts':dict(counts),'sha256':file_hash(RUN/'semantic_dev'/(lang+'.jsonl')),'errors':errors})
    if not store.get('natural_train_relabeled'):
        for lang in LANGS:
            rows=[r for r in read_jsonl(DATA/lang/'train.jsonl') if r['pool']=='natural']
            rows=[r for r in rows if store.allowed(visible_key(r),r['source_id'])]
            status('RELABELING_NATURAL_TRAIN_TO_MATCH_REWARD',language=lang,candidates=len(rows))
            audit,accepted=judged(rows,judge,store,'train',0);write_jsonl(RUN/'seed_train_audit'/(lang+'.jsonl'),audit)
        store.put('natural_train_relabeled',True)
    for lang in LANGS:
        if store.get('champion:'+lang):continue
        path=Path('outputs/auto_rl_v12_20261025/round_01/vi') if lang=='vi' else Path('outputs/reward_v10_20261025')/lang/'data_sft'
        model=load_student(path,'cuda');tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        semantic=store.rows(lang,'semantic_dev');natural=[r for r in read_jsonl(DATA/lang/'dev.jsonl') if r['evaluation_slice']=='natural']
        score,floor,prob=evaluate(model,tok,semantic,natural)
        store.put('champion:'+lang,{'path':str(path),'semantic_dev':score,'natural_floor_dev':floor,'precision':'FP32_LATENT_RESEARCH','generation':0})
        del model;gc.collect();torch.cuda.empty_cache()


def train_language(store,lang,cycle,branch,initial,steps=200):
    semantic=store.rows(lang,'semantic_dev');train=store.rows(lang,'train');counts=Counter(r['label'] for r in train)
    if min(counts.get('EOT',0),counts.get('NEOT',0))<5:return {'status':'WAITING_FOR_BOTH_REWARD_CLASSES','language':lang,'counts':dict(counts)}
    folder=RUN/'cycles'/f'{cycle:06d}'/lang/branch;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'report.json').exists():return json.loads((folder/'report.json').read_text())
    tok=AutoTokenizer.from_pretrained(initial['path'],local_files_only=True);model=load_student(initial['path'],'cuda');anchor=load_student(initial['path'],'cuda')
    anchor.eval()
    for p in anchor.parameters():p.requires_grad_(False)
    x,lengths=encode(train,tok);labels=np.array([LABEL2ID[r['label']] for r in train],np.int64);pools={label:np.flatnonzero(labels==label) for label in [0,1]}
    natural=[r for r in read_jsonl(DATA/lang/'dev.jsonl') if r['evaluation_slice']=='natural']
    seed=20261026+cycle;rng=np.random.default_rng(seed);torch.manual_seed(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);start_step=0
    resume=folder/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start_step=saved['step']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    history=[]
    for step in range(start_step+1,steps+1):
        if STOP:break
        model.train();index=np.concatenate([rng.choice(pools[label],128,replace=True) for label in [0,1]]);rng.shuffle(index);width=int(lengths[index].max())
        ids=torch.from_numpy(x[index,:width]).cuda();mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index]).cuda()[:,None];y=torch.from_numpy(labels[index]).cuda()
        optimizer.zero_grad(set_to_none=True);logits=model(ids,mask.long()).float()
        with torch.inference_mode():reference=anchor(ids,mask.long()).float().softmax(-1)
        if branch=='rl':loss,stats=actor_loss(logits,y,reference)
        else:
            ce=F.cross_entropy(logits,y,reduction='none',label_smoothing=.02)
            if branch=='cost_ce':ce=ce*torch.where(y==0,2.,1.)
            loss=ce.mean()+.05*F.kl_div(F.log_softmax(logits,-1),reference,reduction='batchmean');stats={}
        assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
        for group in optimizer.param_groups:group['lr']=2e-4*min(1.,step/20)*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
        optimizer.step()
        if step%25==0 or step==steps:
            temporary=folder/'resume.tmp';torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},temporary);temporary.replace(resume)
            value={'language':lang,'cycle':cycle,'branch':branch,'step':step,'steps':steps,'loss':float(loss.detach()),'expected_training_reward':float(stats.get('expected_reward',torch.tensor(0.))),
                'train_samples':len(train),'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'training_precision':'FP32_LATENT_NO_ORIGINAL_QAT_SCALE_CONSTRAINT'}
            atomic_json(folder/'progress.json',value);status('TRAINING',**value);history.append(value)
    if STOP:
        del model,anchor,optimizer;gc.collect();torch.cuda.empty_cache();return {'status':'CHECKPOINTED_FOR_USER_STOP','path':str(folder)}
    score,floor,prob=evaluate(model,tok,semantic,natural)
    for name in ['config.json','tokenizer.json','tokenizer_config.json']:shutil.copy2(Path(initial['path'])/name,folder/name)
    torch.save(model.state_dict(),folder/'student.pt');contract=json.loads((Path(initial['path'])/'turn_detector_config.json').read_text())
    contract.update(threshold=score['metrics']['threshold'],training_method='CONTINUOUS_SEMANTIC_'+branch.upper(),quality_status='SILVER_DEV_ONLY_NOT_HUMAN_CERTIFIED',storage_precision='FP32_LATENT',current_truncation='tail')
    atomic_json(folder/'turn_detector_config.json',contract)
    np.savez_compressed(folder/'semantic_dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in semantic]),prob_EOT=prob)
    report={'status':'CANDIDATE_COMPLETE','path':str(folder),'language':lang,'cycle':cycle,'branch':branch,'steps_completed':steps,'semantic_dev':score,'natural_floor_dev':floor,
        'student_sha256':file_hash(folder/'student.pt'),'reference':'semantic dev auto-judge silver; natural-floor dev separate diagnostic','precision':'FP32_LATENT_RESEARCH_NOT_PACKED_INT4','training_log':history,'quality_target_certified':False}
    atomic_json(folder/'report.json',report);del model,anchor,optimizer;gc.collect();torch.cuda.empty_cache();return report


def snapshot(store,cycle):
    folder=RUN/'recovery';folder.mkdir(exist_ok=True);database=folder/'state_snapshot.sqlite';store.backup(database)
    archive=folder/'latest_snapshot.zip';temporary=folder/'snapshot.tmp';members=[database,RUN/'config.json',RUN/'status.json',RUN/'identity.json']
    members.extend((RUN/'semantic_dev').glob('*.jsonl'));members.extend(Path('turn_v2').glob('*.py'));members.append(DATA/'auto_rl_guard.json')
    for lang in LANGS:
        path=Path(store.get('champion:'+lang)['path']);members.extend(path/name for name in ['student.pt','config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json'])
        # Include latest candidate state/optimizer for interruption recovery.
        for active_cycle in {cycle,store.get('cycle',1)}:
            current=RUN/'cycles'/f'{active_cycle:06d}'/lang
            if current.exists():members.extend(p for p in current.rglob('*') if p.is_file() and p.name in ['resume.pt','report.json','progress.json'])
        members.extend((DATA/lang)/name for name in ['train.jsonl','dev.jsonl','manifest.json'])
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        seen=set()
        for path in members:
            if path.exists() and path.as_posix() not in seen:z.write(path,path.as_posix());seen.add(path.as_posix())
    temporary.replace(archive);receipt={'cycle':cycle,'path':str(archive),'sha256':file_hash(archive),'bytes':archive.stat().st_size,'created_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(folder/'latest_snapshot.json',receipt)
    # Mounting Drive requires a normal user authorization in the notebook.
    # If already mounted, the worker also writes a compact durable copy itself.
    drive=Path('/content/drive/MyDrive/turn_detect_continuous_20261026')
    if drive.parent.is_dir():
        drive.mkdir(exist_ok=True);shutil.copy2(archive,drive/'latest_snapshot.zip');atomic_json(drive/'latest_snapshot.json',receipt)
    return receipt


def run(args):
    global RUN;RUN=Path(args.output);RUN.mkdir(parents=True,exist_ok=True)
    for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,request_stop)
    with singleton(RUN/'worker.lock'):
        store=Store(RUN/'state.sqlite');torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        config={'run':str(RUN),'max_cycles':args.max_cycles,'continuous_until_user_stop':args.max_cycles==0,'cadence_seconds':args.cadence,'pages_per_language':3,
            'steps_per_candidate':args.steps,'seed':20261026,'judge':MODEL,'judge_revision':REV,'retention':'All receipts preserved; no deletion of old experiments',
            'stop_on_dev_plateau':False,'stop_on_silver90':False,'real_quality_target_certified':False}
        if not (RUN/'config.json').exists():atomic_json(RUN/'config.json',config)
        else:assert json.loads((RUN/'config.json').read_text())==config,'Resume with the original configuration'
        if not (RUN/'identity.json').exists():atomic_json(RUN/'identity.json',{'source_sha256':file_hash(Path(__file__)),'state_source_sha256':file_hash(Path('turn_v2/continuous_v13_state.py')),
            'reward_model':MODEL,'reward_revision':REV,'original_serving_models_modified':False,'meaning':'Text semantic completeness learned from automatic silver; real floor diagnostics not substituted for human gold'})
        crawler=Crawler(store);judge=None;failures=0
        while not STOP:
            try:
                if (RUN/'STOP_REQUESTED').exists():request_stop();break
                if shutil.disk_usage(RUN).free<15*2**30:
                    status('WAITING_FOR_REMOTE_DISK',free_GiB=shutil.disk_usage(RUN).free/2**30);time.sleep(30);continue
                if judge is None:
                    status('LOADING_JUDGE',resume_cycle=store.get('cycle',1));judge=GeneratedJudge()
                bootstrap(store,judge,crawler)
                if not store.get('initial_snapshot_done'):
                    snapshot(store,0);store.put('initial_snapshot_done',True)
                cycle=store.get('cycle',1);start=time.time();folder=RUN/'cycles'/f'{cycle:06d}';folder.mkdir(parents=True,exist_ok=True)
                if args.max_cycles and cycle>args.max_cycles:break
                for lang in LANGS:
                    stage='crawl_done:'+str(cycle)+':'+lang
                    if not store.get(stage):
                        status('CRAWLING_AND_JUDGING',cycle=cycle,language=lang,pool_counts=store.counts())
                        sources,errors=crawler.pages(lang,cycle,pages=3);candidates=[]
                        for source in sources:
                            candidates.extend(r for r in prefixes(source) if store.allowed(visible_key(r),r['source_id']))
                        audit,accepted=judged(candidates,judge,store,'train',cycle)
                        write_jsonl(folder/(lang+'_judge_audit.jsonl'),audit);atomic_json(folder/(lang+'_crawl.json'),{'pages':len(sources),'candidates':len(candidates),'accepted':len(accepted),'errors':errors})
                        store.put(stage,True)
                    initial=store.get('champion:'+lang);candidates=[]
                    for branch in ['sft','rl']+(['cost_ce'] if cycle%5==0 else []):
                        if STOP:break
                        candidate=train_language(store,lang,cycle,branch,initial,args.steps);candidates.append(candidate)
                    eligible=[c for c in candidates if c.get('status')=='CANDIDATE_COMPLETE' and c['semantic_dev']['selection_score']>initial['semantic_dev']['selection_score']]
                    if eligible:
                        winner=max(eligible,key=lambda c:c['semantic_dev']['selection_score']);store.put('champion:'+lang,winner);store.event('CHAMPION_IMPROVED',winner)
                    else:store.event('CYCLE_NO_DEV_IMPROVEMENT',{'cycle':cycle,'language':lang,'continue':True})
                if STOP:break
                summary={'cycle':cycle,'pool_counts':store.counts(),'champions':{lang:store.get('champion:'+lang) for lang in LANGS},'duration_seconds':time.time()-start,
                    'silver_dev_is_not_real_gold':True,'quality_target_certified':False,'next_cycle':cycle+1}
                atomic_json(folder/'summary.json',summary);store.put('cycle',cycle+1);status('CYCLE_COMPLETE_CONTINUING',**summary)
                snapshot(store,cycle);failures=0
                until=time.time()+max(0,args.cadence-(time.time()-start))
                while time.time()<until and not STOP:
                    if (RUN/'STOP_REQUESTED').exists():request_stop();break
                    time.sleep(min(10,max(0,until-time.time())))
            except Exception as error:
                failures+=1;trace=traceback.format_exc();store.event('RECOVERABLE_ERROR',{'type':type(error).__name__,'message':str(error),'traceback':trace})
                atomic_json(RUN/'latest_error.json',{'cycle':store.get('cycle',1),'at':datetime.now(timezone.utc).isoformat(),'type':type(error).__name__,'message':str(error),'traceback':trace})
                status('RETRYING_AFTER_ERROR',error_type=type(error).__name__,error=str(error),backoff_seconds=min(600,30*2**min(failures,4)))
                if isinstance(error,torch.OutOfMemoryError):
                    judge=None;gc.collect();torch.cuda.empty_cache()
                until=time.time()+min(600,30*2**min(failures,4))
                while time.time()<until and not STOP:time.sleep(min(10,max(0,until-time.time())))
        atomic_json(RUN/'status.json',{'status':'STOPPED_BY_REQUEST' if STOP else 'REQUESTED_BOUNDED_RUN_COMPLETED','pid':os.getpid(),'cycle':store.get('cycle',1),'quality_target_certified':False})
        try:snapshot(store,max(0,store.get('cycle',1)-1))
        except Exception:pass


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(RUN));p.add_argument('--max-cycles',type=int,default=0)
    p.add_argument('--steps',type=int,default=200);p.add_argument('--cadence',type=int,default=300);run(p.parse_args())
