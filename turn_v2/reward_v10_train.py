"""Matched data/CE/cost-CE/REINFORCE ablation for three independent QAT GRUs."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import transformers.utils as hf_utils
import transformers.utils.import_utils as hf_imports
hf_utils.is_torchvision_available=hf_imports.is_torchvision_available=lambda:False
hf_utils.is_librosa_available=hf_imports.is_librosa_available=lambda:False
from transformers import AutoTokenizer
from .common import LABEL2ID,encode_pair,file_hash,input_id,read_jsonl,write_json
from .monolingual_data import encode_tail
from .tiny_student import load_student
from .qat_student import prepare_qat,latent_state
from .reward_v10_objective import actor_loss
from .metrics import classification_metrics

DATA=Path('data_v10/reward_20261025');ROOT=Path('outputs/reward_v10_20261025')
INITIAL={'vi':'outputs/qat_20261024/runs/qat_gru96_g64_20261024_r2',
    'en':'outputs/monolingual_20261025/runs/mono_en_student_20261025/qat_best',
    'zh':'outputs/monolingual_20261025/runs/mono_zh_student_20261025/qat_best'}


def encode(rows,tok):
    ids=np.zeros((len(rows),256),np.int64);lengths=np.zeros(len(rows),np.int64)
    for i,r in enumerate(rows):
        current=encode_tail(tok,r.get('context',''),r['text'])['input_ids'];ids[i,:len(current)]=current;lengths[i]=len(current)
    return ids,lengths


def infer(model,ids,lengths,batch=256):
    result=np.zeros(len(ids));model.eval();order=np.argsort(lengths)
    with torch.inference_mode():
        for start in range(0,len(order),batch):
            index=order[start:start+batch];width=int(lengths[index].max());x=torch.from_numpy(ids[index,:width]).cuda();mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index]).cuda()[:,None]
            logits=model(x,mask.long()).float();assert torch.isfinite(logits).all();result[index]=logits.softmax(-1)[:,1].cpu().numpy()
    return result


def describe(rows,probs):
    labels=np.array([LABEL2ID[r['label']] for r in rows]);sources=sorted({r['evaluation_slice'] for r in rows});masks={s:np.array([r['evaluation_slice']==s for r in rows]) for s in sources}
    best=None;best_key=None
    for threshold in np.linspace(.05,.95,181):
        metrics={s:classification_metrics(labels[m],probs[m],float(threshold)) for s,m in masks.items()}
        # Prevent a natural-slice improvement from hiding a collapse on replay.
        key=(min(v['joint_score'] for v in metrics.values()),-sum(v['false_eot'] for v in metrics.values()),-abs(threshold-.5))
        if best_key is None or key>best_key:best=metrics;best_key=key
    return {'threshold':best['natural']['threshold'],'slices':best,'selection_score':best_key[0],
        'natural_target_met':best['natural']['accuracy']>=.9 and best['natural']['eot_f1']>=.9}


def run(language):
    output=ROOT/language;output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    initial=Path(INITIAL[language]);manifest=json.loads((DATA/language/'manifest.json').read_text())
    for s in ['train','dev']:assert file_hash(DATA/language/(s+'.jsonl'))==manifest['split_sha256'][s]
    train=list(read_jsonl(DATA/language/'train.jsonl'));dev=list(read_jsonl(DATA/language/'dev.jsonl'));assert {r['language'] for r in train+dev}=={language}
    silver_path=ROOT/'qwen/silver.jsonl';silver=[r for r in read_jsonl(silver_path) if r['language']==language] if silver_path.exists() else []
    protected={input_id(r) for r in dev};seen={input_id(r) for r in train};silver=[r for r in silver if input_id(r) not in protected and input_id(r) not in seen];train+=silver
    torch.set_num_threads(2);torch.manual_seed(20261025);np.random.seed(20261025);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();tok=AutoTokenizer.from_pretrained(initial,local_files_only=True)
    ids,lengths=encode(train,tok);dx,dl=encode(dev,tok);labels=np.array([LABEL2ID[r['label']] for r in train],np.int64);weights=np.array([r.get('sample_weight',1.) for r in train],np.float32)
    model=load_student(initial,'cuda');params=torch.load(initial/'quant_params.pt',map_location='cpu',weights_only=True)
    scales={n:v.to('cuda') for n,v in params['scales'].items()};prepare_qat(model,params['group_size'],scales)
    baseline_probs=infer(model,dx,dl);baseline=describe(dev,baseline_probs);np.savez_compressed(output/'baseline_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=baseline_probs)
    write_json(output/'baseline.json',baseline)
    pools=defaultdict(list)
    for i,row in enumerate(train):pools[(row['pool'],row['label'])].append(i)
    sources=sorted({r['pool'] for r in train});source_weights={'natural':.50,'replay':.30,'expanded_dialogue':.15,'crawl_silver':.05}
    probability=np.array([source_weights[s] for s in sources]);probability/=probability.sum()
    # Refuse unsupported class pools instead of silently relabeling them.
    pools={k:np.array(v) for k,v in pools.items()};reports=[];reference_model=None
    identity={'language':language,'GPU':torch.cuda.get_device_name(0),'training_languages':[language],'shared_trainable_weights':False,
        'initial_student_sha256':file_hash(initial/'student.pt'),'initial_quant_params_sha256':file_hash(initial/'quant_params.pt'),
        'data_manifest_sha256':file_hash(DATA/language/'manifest.json'),'source_sha256':file_hash(Path(__file__)),'reward_source_sha256':file_hash(Path('turn_v2/reward_v10_objective.py')),
        'train_counts':{s:sum(r['pool']==s for r in train) for s in sources},'dev_counts':{s:sum(r['evaluation_slice']==s for r in dev) for s in {r['evaluation_slice'] for r in dev}},
        'no_held_evaluation_or_selection':True,'training':'INT4 fake quant every forward, own frozen scales; FP32 activations',
        'RL':'Offline one-step sampled REINFORCE with detached all-action baseline; not PPO, online interaction or a measured latency simulator',
        'rewards':'Correct+1; premature stop-2; wait on EOT-0.5. Automatic/proxy labels. Crawl semantic silver excluded from policy reward.',
        'matched_controls':'ce_control,cost_ce,reinforce start from same selected data_sft weights; same batch-index RNG, steps225, lr2e-4',
        'input':'Only current/past visible text; tail truncation256; no audio/timing/future in model input','seed':20261025}
    write_json(output/'identity.json',identity)
    for phase,steps,lr in [('data_sft',450,5e-4),('ce_control',225,2e-4),('cost_ce',225,2e-4),('reinforce',225,2e-4)]:
        folder=output/phase;folder.mkdir();torch.manual_seed(20261025);rng=np.random.default_rng(20261025)
        if phase!='data_sft':
            # Load effective latent matrices by removing and rebuilding only
            # process-local parametrizations; scales remain the original ones.
            del model;gc.collect();torch.cuda.empty_cache();model=load_student(output/'data_sft','cuda');prepare_qat(model,params['group_size'],scales)
        if phase=='reinforce':
            reference_model=load_student(output/'data_sft','cuda');prepare_qat(reference_model,params['group_size'],scales);reference_model.eval()
            for p in reference_model.parameters():p.requires_grad_(False)
        best=baseline if phase=='data_sft' else json.loads((output/'data_sft/best_dev.json').read_text());best_score=best['selection_score'];best_step=0;history=[]
        # Every phase includes its own unchanged starting point as a control.
        torch.save(latent_state(model),folder/'student.pt');torch.save(params,folder/'quant_params.pt')
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:shutil.copy2(initial/name,folder/name)
        starting_probs=baseline_probs if phase=='data_sft' else np.load(output/'data_sft/dev_predictions.npz')['prob_EOT']
        np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=starting_probs)
        write_json(folder/'best_dev.json',best);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        # Full sequences remain modest for a GRU; batch256 makes useful GPU
        # work. Qwen labeling is the memory-intensive stage of the pipeline.
        gradient_verified=False
        for step in range(1,steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            chosen=rng.choice(sources,size=128,p=probability);index=[]
            for source in chosen:
                for label in ['NEOT','EOT']:
                    pool=pools.get((source,label))
                    if pool is None:pool=np.concatenate([v for (s,_),v in pools.items() if s==source])
                    index.append(rng.choice(pool))
            index=np.array(index);width=int(lengths[index].max());x=torch.from_numpy(ids[index,:width]).cuda();mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index]).cuda()[:,None]
            y=torch.from_numpy(labels[index]).cuda();w=torch.from_numpy(weights[index]).cuda();logits=model(x,mask.long()).float()
            ce=F.cross_entropy(logits,y,reduction='none',label_smoothing=.02)
            if phase=='cost_ce':ce=ce*torch.where(y==0,2.,1.)
            loss=(ce*w).sum()/w.sum();reward_values={}
            if phase=='reinforce':
                with torch.inference_mode():ref=reference_model(x,mask.long()).float().softmax(-1)
                eligible=torch.tensor([train[i]['pool']!='crawl_silver' for i in index],device='cuda')
                actor,stats=actor_loss(logits[eligible],y[eligible],ref[eligible])
                loss=.5*loss+.5*actor;reward_values={k:float(v) for k,v in stats.items()}
            assert torch.isfinite(loss);loss.backward()
            if not gradient_verified:
                gradient_verified=all(getattr(model.get_submodule(name.rsplit('.',1)[0]).parametrizations,name.rsplit('.',1)[1]).original.grad is not None for name in params['weight_names'])
                assert gradient_verified,'QAT gradient missing'
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
            for group in optimizer.param_groups:group['lr']=lr*min(1.,step/20)*(.2+.8*.5*(1+math.cos(math.pi*step/steps)))
            optimizer.step()
            if step%25==0:
                progress={'language':language,'phase':phase,'step':step,'steps_max':steps,'batch_size':len(index),'loss':float(loss.detach()),'reward':reward_values,
                    'CUDA_peak_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started}
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if step%75==0 or step==steps:
                probs=infer(model,dx,dl);record={'step':step,**describe(dev,probs)};history.append(record)
                if record['selection_score']>best_score:
                    best=record;best_score=record['selection_score'];best_step=step
                    torch.save(latent_state(model),folder/'student.pt');np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs);write_json(folder/'best_dev.json',record)
                write_json(folder/'history.json',history)
        contract=json.loads((initial/'turn_detector_config.json').read_text());contract.update(threshold=best['threshold'],language=language,current_truncation='tail',
            quality_status='V10_DEV_ONLY_UNSEEN_REAL_TARGET_NOT_CERTIFIED',training_method='QAT_DATA_AND_'+phase.upper())
        contract.pop('language_thresholds',None);write_json(folder/'turn_detector_config.json',contract)
        report={'phase':phase,'steps':steps,'best_step':best_step,'dev':best,'gradient_verified':gradient_verified,'student_sha256':file_hash(folder/'student.pt'),'quality_target_certified':False}
        reports.append(report);write_json(folder/'report.json',report);del optimizer
        if reference_model is not None:del reference_model;reference_model=None
    winner=max(reports,key=lambda r:r['dev']['selection_score']);write_json(output/'selection.json',{'selected_phase':winner['phase'],'selection_basis':'minimum dev-slice joint score; no regression/test tuning','baseline':baseline,'phases':reports,'quality_target_certified':False})
    write_json(output/'report.json',{'status':'MATCHED_QAT_DATA_REWARD_ABLATION_COMPLETE','language':language,'baseline':baseline,'phases':reports,'selected_phase':winner['phase'],
        'RL_beats_matched_CE_on_selection_score':next(r['dev']['selection_score'] for r in reports if r['phase']=='reinforce')>next(r['dev']['selection_score'] for r in reports if r['phase']=='ce_control'),
        'RL_beats_cost_CE_on_selection_score':next(r['dev']['selection_score'] for r in reports if r['phase']=='reinforce')>next(r['dev']['selection_score'] for r in reports if r['phase']=='cost_ce'),
        'local_disk_policy':'No full teacher download; small students/results only','elapsed_seconds':time.monotonic()-started,'quality_target_certified':False})
    del model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--language',choices=['vi','en','zh'],required=True);run(p.parse_args().language)
