"""Text-only teacher adaptation after matched-window Vietnamese ASR repair."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from .common import LABEL2ID,file_hash,input_id,read_jsonl,write_json
from .natural_v5_train import summarize
from .natural_v6_finetune import encode_rows
from .stream_v4_train import infer


def run(args):
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(20261028);rng=np.random.default_rng(20261028)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((args.data/'manifest.json').read_text())
    for split in ['train','dev']:assert file_hash(args.data/(split+'.jsonl'))==manifest['files'][split]['sha256']
    train=[{**r,'training_source':'natural'} for r in read_jsonl(args.data/'train.jsonl')]
    dev=[{**r,'evaluation_slice':'natural'} for r in read_jsonl(args.data/'dev.jsonl')]
    legacy=[{**r,'evaluation_slice':'legacy_vi'} for r in read_jsonl(Path('data_v4/causal_20261025/dev.jsonl')) if r['language']=='vi']
    dev+=legacy
    # Exclude exact test inputs from replay without loading test labels into
    # any training/selection objective. Keep all held cases, including errors.
    protected={r['language']+input_id(r) for r in read_jsonl(args.data/'sealed.jsonl')}
    protected.update(r['language']+input_id(r) for r in dev)
    replay=[{**r,'training_source':'replay'} for r in read_jsonl(Path('data_v4/causal_20261025/train.jsonl')) if r['language']+input_id(r) not in protected]
    train=[r for r in train if r['language']+input_id(r) not in protected]+replay
    baselines=[Path('runs/stream_v4_campaign_20261025/round_02/best'),Path('runs/natural_v5_train_20261025/round_02/best')]
    tokenizer=AutoTokenizer.from_pretrained(baselines[0],local_files_only=True)
    x,lengths=encode_rows(train,tokenizer);dx,dl=encode_rows(dev,tokenizer)
    labels=np.array([LABEL2ID[r['label']] for r in train],dtype=np.int64)
    initial=[];best=-1.;best_path=None;best_record=None;best_predictions=None
    for baseline_id,path in enumerate(baselines):
        model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,dtype=torch.float32).cuda().eval()
        probs=infer(model,dx,dl,batch_size=32);record={'baseline_path':str(path),**summarize(dev,probs)}
        predictions_path=args.output/f'initial_baseline_{baseline_id:02d}.npz'
        np.savez_compressed(predictions_path,labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
        initial.append(record)
        if record['minimum_dev_slice_joint_score']>best:
            best=record['minimum_dev_slice_joint_score'];best_path=str(path);best_record=record;best_predictions=str(predictions_path)
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(args.output/'initial_dev.json',{'candidates':initial,'selected_initial_path':best_path,'selection_basis':'Same development min slice score'})
    model=AutoModelForSequenceClassification.from_pretrained(best_path,local_files_only=True,dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    groups=defaultdict(list)
    for i,row in enumerate(train):groups[(row['language'],row['training_source'],row['label'])].append(i)
    groups={k:np.array(v) for k,v in groups.items()}
    write_json(args.output/'identity.json',{'source_sha256':file_hash(Path(__file__)),'codec_sha256':file_hash(Path('turn_v2/natural_v6_finetune.py')),
        'data_manifest_sha256':file_hash(args.data/'manifest.json'),'initial_path':best_path,'initial_sha256':file_hash(Path(best_path)/'model.safetensors'),
        'candidate_baselines':[{ 'path':str(p),'sha256':file_hash(p/'model.safetensors')} for p in baselines],
        'input':'Only current/past-partner normalized text, latest254currenttokens, latestcontext to512total',
        'dev_samples':len(dev),'train_samples_with_replay':len(train),'replay':len(replay),'rounds_max':3,'steps_per_round':300,
        'microbatch':8,'accumulation':3,'language_batch':'Each optimizer step has one microbatch per language',
        'held_labels_used_as_targets_or_for_selection':False,'held_text_hashes_used_for_replay_dedup':True,
        'stop':'4 stale evaluations after at least one complete round, strict dev target, or maxrounds',
        'quality_protocol':'Do not consume limited held cohort if dev stays below target; all prior pilots regression only'})
    history=[];stale=0;reason='MAX_ROUNDS'
    for round_id in range(1,4):
        lr=[5e-6,2e-6,1e-6][round_id-1];folder=args.output/f'round_{round_id:02d}';folder.mkdir()
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        for step in range(1,301):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[]
            for lang in ['vi','en','zh']:
                source='natural' if rng.random()<(.7 if lang=='vi' else .8) else 'replay'
                index=np.concatenate([rng.choice(groups[(lang,source,label)],4,replace=len(groups[(lang,source,label)])<4) for label in ['NEOT','EOT']]);rng.shuffle(index)
                width=int(lengths[index].max());ids=torch.from_numpy(x[index,:width].astype(np.int64)).cuda()
                mask=torch.arange(width,device='cuda')[None]<torch.from_numpy(lengths[index].astype(np.int64)).cuda()[:,None]
                logits=model(input_ids=ids,attention_mask=mask.long()).logits.float()
                loss=torch.nn.functional.cross_entropy(logits,torch.from_numpy(labels[index]).cuda(),label_smoothing=.02)/3
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*3)
            factor=min(1.,step/30)*(.2+.8*.5*(1+math.cos(math.pi*step/300)))
            for group in optimizer.param_groups:group['lr']=lr*factor
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%25==0:
                progress={'stage':'MATCHED_ASR_TEXT_TRAINING','round':round_id,'step':step,'steps_per_round':300,
                    'loss':float(np.mean(losses)),'elapsed_seconds':time.monotonic()-started}
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if step%100==0:
                probs=infer(model,dx,dl,batch_size=32);record={'round':round_id,'step':step,**summarize(dev,probs)}
                history.append(record);write_json(args.output/'history.json',history)
                if record['minimum_dev_slice_joint_score']>best:
                    best=record['minimum_dev_slice_joint_score'];best_path=str(folder/'best');best_record=record;stale=0
                    model.save_pretrained(best_path,safe_serialization=True);tokenizer.save_pretrained(best_path)
                    write_json(folder/'best_dev.json',record);np.savez_compressed(folder/'dev_predictions.npz',labels=np.array([LABEL2ID[r['label']] for r in dev]),prob_EOT=probs)
                    best_predictions=str(folder/'dev_predictions.npz')
                else:stale+=1
                write_json(args.output/'selection.json',{'best_path':best_path,'best_dev':best_record,'stale_evaluations':stale,'held_classifier_scored':False})
                print(json.dumps({'stage':'MATCHED_ASR_TEXT_DEV',**record}),flush=True)
                if record['development_all_slices_over_90'] and record['natural_hold_false_eot_below_10_percent']:reason='DEV_TARGET';break
                if round_id>=2 and stale>=4:reason='DEV_PLATEAU';break
        del optimizer,model;gc.collect();torch.cuda.empty_cache()
        if reason!='MAX_ROUNDS':break
        model=AutoModelForSequenceClassification.from_pretrained(best_path,dtype=torch.float32).cuda();model.gradient_checkpointing_enable()
    write_json(args.output/'report.json',{'status':'MATCHED_ASR_TEXT_CAMPAIGN_FINISHED','stop_reason':reason,'best_path':best_path,'best_dev':best_record,
        'history':history,'checkpoint_sha256':file_hash(Path(best_path)/'model.safetensors'),'held_classifier_scored':False,'best_dev_predictions_path':best_predictions,
        'elapsed_seconds':time.monotonic()-started,'human_verified_quality_certified':False,'real_target_achieved':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data_v6/pho_text_20261025'))
    parser.add_argument('--output',type=Path,default=Path('runs/natural_v7_text_20261025'));run(parser.parse_args())
