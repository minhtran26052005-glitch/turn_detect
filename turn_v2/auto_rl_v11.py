"""Bounded automatic crawl -> independent judge -> one-step RL -> dev gate."""
from collections import Counter
import gc
import json
from pathlib import Path
import re
import time
import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import LABEL2ID,digest,file_hash,input_id,normalize,read_jsonl,write_json,write_jsonl
from .reward_v10_qwen import MODEL,REV,SYSTEM,VIEWS
from .reward_v10_train import encode,infer,describe
from .reward_v10_objective import actor_loss
from .tiny_student import load_student
from .qat_student import prepare_qat,latent_state

ROOT=Path('outputs/auto_rl_v11_20261025')
TITLES=[{'vi':['Đồng hồ','Xe đạp'],'en':['Clock','Garden'],'zh':['时钟','公园']},
    {'vi':['Thư viện','Âm nhạc'],'en':['Rail transport','Music'],'zh':['铁路运输','音乐']}]
PROBE=[{'language':'vi','text':'tôi muốn đặt vé đi đến','context':'','expected':'NEOT'},
    {'language':'vi','text':'cảm ơn bạn đã giúp tôi','context':'','expected':'EOT'},
    {'language':'en','text':'I would like to travel from','context':'','expected':'NEOT'},
    {'language':'en','text':'Thank you for your help','context':'','expected':'EOT'},
    {'language':'zh','text':'我想预订从','context':'','expected':'NEOT'},
    {'language':'zh','text':'谢谢你的帮助','context':'','expected':'EOT'}]


def prefixes(source):
    """Cut candidates only; no labels/rewards are inferred from cut position."""
    result=[];sentences=re.split(r'(?<=[.!?。！？])\s*|\n+',source['text'])
    for sentence in [s for s in sentences if 25<=len(s)<=700][:6]:
        matches=list(re.finditer(r'[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s\w]' if source['language']=='zh' else r'\S+',sentence))
        if len(matches)<6:continue
        for ratio in [.35,.65,1.]:
            end=matches[min(len(matches)-1,max(0,int(len(matches)*ratio)-1))].end();text=sentence[:end].strip()
            if len(normalize(text))<5:continue
            result.append({'candidate_id':digest(source['source_id']+'\0'+sentence+'\0'+str(end)),'language':source['language'],'text':text,'context':'',
                'source_id':source['source_id'],'source_title':source['title'],'source_url':source['url'],'source_revision':source['page_revision'],
                'source_sentence_sha256':digest(sentence),'hidden_suffix_sha256':digest(sentence[end:]),'label':None,'is_gold':False,'candidate_only':True})
    return result


def crawl(round_index,protected_sources,protected_inputs):
    http=requests.Session();http.headers['User-Agent']='TurnDetectorResearch/0.11 (bounded offline RL research; provenance recorded)';sources=[];rows=[];errors=[]
    for lang,titles in TITLES[round_index].items():
        for title in titles:
            try:
                response=http.get(f'https://{lang}.wikipedia.org/w/api.php',params={'action':'query','format':'json','formatversion':2,'prop':'extracts|info|revisions','explaintext':1,'exintro':1,'inprop':'url','rvprop':'ids','redirects':1,'titles':title},timeout=35)
                response.raise_for_status()
                for page in response.json().get('query',{}).get('pages',[]):
                    sid=lang+'wiki:'+str(page['pageid'])
                    if sid in protected_sources or not page.get('extract'):continue
                    source={'language':lang,'source_id':sid,'title':page['title'],'url':page.get('fullurl'),'page_revision':page.get('revisions',[{}])[0].get('revid'),
                        'text':page['extract'],'sha256':digest(page['extract']),'license':'Wikipedia CC BY-SA terms; attribution retained'}
                    sources.append(source)
                    for row in prefixes(source):
                        key=lang+input_id(row)
                        if key in protected_inputs:continue
                        protected_inputs.add(key);rows.append(row)
            except (requests.RequestException,ValueError) as error:errors.append({'language':lang,'title':title,'error':type(error).__name__+':'+str(error)[:150]})
            time.sleep(.5)
    return sources,rows,errors


class Judge:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True);self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REV,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        self.head=self.model.lm_head.weight[[self.tokenizer.encode(v,add_special_tokens=False)[0] for v in ['A','B','C']]].detach().float()

    def score(self,rows):
        examples=[{'role':'user','content':'Observed English words: Please tell me the way to'},{'role':'assistant','content':'B'},
            {'role':'user','content':'Observed Vietnamese words: tôi muốn mua một chiếc'},{'role':'assistant','content':'B'},
            {'role':'user','content':'Observed Chinese words: 今天的会议结束了'},{'role':'assistant','content':'A'}]
        prompts=[]
        for row in rows:
            visible=json.dumps({'language':row['language'],'previous_partner_words':normalize(row.get('context','')),'current_speaker_words':normalize(row['text'])},ensure_ascii=False)
            for view in VIEWS:
                prompts.append(self.tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},*examples,{'role':'user','content':view+'\n'+visible}],tokenize=True,add_generation_prompt=True,return_dict=False))
        result=np.zeros((len(prompts),3))
        with torch.inference_mode():
            for start in range(0,len(prompts),32):
                batch=self.tokenizer.pad({'input_ids':prompts[start:start+32]},padding=True,return_tensors='pt').to('cuda');assert batch['input_ids'].shape[-1]<1536
                mask=batch['attention_mask'];positions=(mask.cumsum(-1)-1).clamp_min(0)
                hidden=self.model.model(**batch,position_ids=positions,use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
                result[start:start+len(batch['input_ids'])]=F.linear(hidden,self.head).softmax(-1).cpu().numpy()
        audit=[]
        for i,row in enumerate(rows):
            scores=result[2*i:2*i+2];decisions=scores.argmax(1);accepted=bool(decisions[0]==decisions[1] and decisions[0]!=2 and scores.max(1).min()>=.8)
            audit.append({**row,'judge_labels':[['EOT','NEOT','AMBIGUOUS'][int(d)] for d in decisions],'judge_scores':scores.tolist(),'judge_accepted':accepted,
                'judge_label':['EOT','NEOT','AMBIGUOUS'][int(decisions[0])] if accepted else None,'judge_gold':False,'teacher_revision':REV})
        return audit


def train_round(language,folder,accepted,round_index,state):
    base=state['path'];tok=AutoTokenizer.from_pretrained(base,local_files_only=True);model=load_student(base,'cuda')
    params=torch.load(base/'quant_params.pt',map_location='cpu',weights_only=True);scales={n:v.to('cuda') for n,v in params['scales'].items()};prepare_qat(model,params['group_size'],scales)
    anchor=load_student(base,'cuda');prepare_qat(anchor,params['group_size'],scales);anchor.eval()
    for p in anchor.parameters():p.requires_grad_(False)
    dev=list(read_jsonl(Path('data_v10/reward_20261025')/language/'dev.jsonl'));dx,dl=encode(dev,tok)
    before_probs=infer(model,dx,dl);before=describe(dev,before_probs)
    replay=[r for r in read_jsonl(Path('data_v10/reward_20261025')/language/'train.jsonl') if r['pool']=='natural'];rx,rl=encode(replay,tok);ry=np.array([LABEL2ID[r['label']] for r in replay],np.int64)
    records=[{**r,'label':r['judge_label']} for r in accepted];ids,lengths=encode(records,tok);labels=np.array([LABEL2ID[r['label']] for r in records],np.int64)
    torch.manual_seed(20261025+round_index);rng=np.random.default_rng(20261025+round_index);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    counts=Counter(r['judge_label'] for r in records)
    if min(counts.get('EOT',0),counts.get('NEOT',0))<2:
        write_json(folder/'report.json',{'status':'SKIPPED_INSUFFICIENT_REWARD_CLASS_DIVERSITY','language':language,'judge_counts':dict(counts),'dev_before':before,'updated':False});return state
    actions_receipt=[]
    for step in range(1,76):
        model.train();optimizer.zero_grad(set_to_none=True);index=rng.choice(len(records),64,replace=True);width=int(lengths[index].max())
        x=torch.from_numpy(ids[index,:width]).cuda();mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index]).cuda()[:,None];y=torch.from_numpy(labels[index]).cuda()
        logits=model(x,mask.long()).float()
        with torch.inference_mode():reference=anchor(x,mask.long()).float().softmax(-1)
        loss,stats=actor_loss(logits,y,reference)
        # Observed-floor replay anchors the policy against optimizing the judge
        # alone. Candidate labels never enter the measured dev references.
        j=rng.choice(len(replay),64,replace=True);w=int(rl[j].max());a=torch.from_numpy(rx[j,:w]).cuda();m=torch.arange(w,device='cuda')[None]<torch.from_numpy(rl[j]).cuda()[:,None]
        supervised=F.cross_entropy(model(a,m.long()).float(),torch.from_numpy(ry[j]).cuda(),label_smoothing=.02)
        total=.5*loss+.5*supervised;assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%25==0:
            value={'language':language,'round':round_index+1,'step':step,'sampled_judge_reward':float(stats['sampled_reward']),'reference_KL':float(stats['kl']),'loss':float(total.detach())};actions_receipt.append(value);write_json(folder/'progress.json',value)
    probs=infer(model,dx,dl);after=describe(dev,probs);kept=after['selection_score']>before['selection_score']
    np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
    if kept:
        import shutil
        for name in ['config.json','tokenizer.json','tokenizer_config.json','quant_params.pt']:shutil.copy2(base/name,folder/name)
        torch.save(latent_state(model),folder/'student.pt');contract=json.loads((base/'turn_detector_config.json').read_text());contract.update(threshold=after['threshold'],quality_status='AUTO_RL_JUDGE_REWARD_DEV_ONLY_NOT_CERTIFIED');write_json(folder/'turn_detector_config.json',contract)
        state={'path':folder,'dev':after}
    write_json(folder/'report.json',{'status':'AUTO_CRAWL_JUDGE_RL_ROUND_COMPLETE','language':language,'steps':75,'judge_counts':dict(counts),'dev_before':before,'dev_after':after,
        'candidate_retained':kept,'reward_history':actions_receipt,'quality_target_certified':False,'old_regressions_used_for_gate':False})
    del model,anchor,optimizer;gc.collect();torch.cuda.empty_cache();return state


def run():
    ROOT.mkdir(parents=True,exist_ok=False);started=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    states={}
    for language in ['vi','en','zh']:
        selection=json.loads((Path('outputs/reward_v10_20261025')/language/'selection.json').read_text());states[language]={'path':Path('outputs/reward_v10_20261025')/language/selection['selected_phase']}
    write_json(ROOT/'identity.json',{'status':'PREDECLARED_TWO_ROUND_AUTO_CRAWL_RL','rounds_max':2,'train_steps_per_language_round':75,'reward_model':MODEL,'reward_revision':REV,
        'reward_scope':'Semantic-completeness judge; not measured real turn-yield timing','judge_prompt_sha256':digest(SYSTEM+'\n'.join(VIEWS)),'probe':PROBE,
        'source_sha256':file_hash(Path(__file__)),'start_models':{lang:{'path':str(s['path']),'sha256':file_hash(s['path']/'student.pt')} for lang,s in states.items()},
        'fixed_reward':{'correct':1,'premature':-2,'late':-.5},'protected_eval_targets_in_judge':False,'test_evaluation':False,'local_teacher_download':False})
    judge=Judge();probe=judge.score(PROBE);write_jsonl(ROOT/'judge_probe.jsonl',probe)
    probe_ok=all(r['judge_label']==r['expected'] for r in probe)
    if not probe_ok:
        write_json(ROOT/'report.json',{'status':'JUDGE_PROBE_FAILED_NO_POLICY_UPDATES','probe_passed':False,'updates':0,'quality_target_certified':False,'note':'Do not learn reward from a judge that fails obvious incomplete/complete probe'});return
    guard=json.loads(Path('data_v10/reward_20261025/auto_rl_guard.json').read_text())
    protected_sources=set(guard['protected_sources']);protected_inputs=set(guard['protected_inputs'])
    for language in ['vi','en','zh']:
        for r in read_jsonl(Path('data_v10/reward_20261025')/language/'dev.jsonl'):protected_inputs.add(language+input_id(r))
    summaries=[]
    for round_index in range(2):
        folder=ROOT/('round_%02d'%(round_index+1));folder.mkdir();write_json(ROOT/'progress.json',{'stage':'CRAWLING','round':round_index+1})
        sources,candidates,errors=crawl(round_index,protected_sources,protected_inputs);write_jsonl(folder/'sources.jsonl',sources);write_jsonl(folder/'candidates.jsonl',candidates)
        audit=judge.score(candidates) if candidates else [];write_jsonl(folder/'judge_audit.jsonl',audit)
        for language in ['vi','en','zh']:
            output=folder/language;output.mkdir();accepted=[r for r in audit if r['language']==language and r['judge_accepted']]
            write_json(ROOT/'progress.json',{'stage':'RL_POLICY_UPDATE','round':round_index+1,'language':language,'accepted':len(accepted)})
            states[language]=train_round(language,output,accepted,round_index,states[language])
        summaries.append({'round':round_index+1,'pages':len(sources),'candidates':len(candidates),'accepted':sum(r['judge_accepted'] for r in audit),'errors':errors})
    write_json(ROOT/'report.json',{'status':'BOUNDED_AUTO_CRAWL_JUDGE_RL_COMPLETE','rounds':summaries,'selected_models':{lang:{'path':str(s['path']),'dev':s.get('dev')} for lang,s in states.items()},
        'probe_passed':probe_ok,'quality_target_certified':False,'source_sha256':file_hash(Path(__file__)),'GPU_peak_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':run()
