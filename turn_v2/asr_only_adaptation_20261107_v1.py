"""Bounded one-full-epoch ASR-only warm adaptation; paused v7 stays untouched."""
from collections import Counter
from datetime import datetime,timezone
import copy
import fcntl
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import traceback
import numpy as np

from .uploaded_vi_en_data_v5 import StreamOrder, initialize, clean, gaps, key
from .tiny_compression_v1.common import encode_pair

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
MAIN=ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
SOURCE=ROOT/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
BASE=ROOT/'outputs/sentence_mix_end_only_test_20261107_v1'
OUT=ROOT/'outputs/asr_only_adaptation_20261107_v1'
STOP=False
CODE=['asr_only_adaptation_20261107_v1.py','uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py',
      'uploaded_vi_en_sentence_mix_evaluation_v7.py','malbert_turn_student_v1.py','tiny_compression_v1/common.py']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('w',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)


def status(stage,**v):
    raw=Path('/proc/self/stat').read_text();v=dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),at_utc=datetime.now(timezone.utc).isoformat(),**v)
    save('status.json',v);print(json.dumps(v),flush=True)


def reserve():
    mount=next(s.split() for s in Path('/proc/mounts').read_text().splitlines() if s.split()[1]=='/mnt/hdd500')
    assert mount[0]=='/dev/sda1' and 'rw' in mount[3].split(',')
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/mnt/hdd500').free)>=13*2**30


def atomic_torch(p,x):
    import torch
    reserve();p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');torch.save(x,t)
    with t.open('rb') as f:os.fsync(f.fileno())
    if p.name=='latest_resume.pt' and p.exists():
        previous=p.with_name('previous_resume.pt');link=p.with_name('previous_snapshot.tmp')
        assert not link.exists();os.link(p,link);link.replace(previous)
    t.replace(p)


def rng_state():
    import torch
    return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())


def restore_rng(x):
    import torch
    torch.set_rng_state(x['torch_rng']);torch.cuda.set_rng_state(x['cuda_rng']);np.random.set_state(x['numpy_rng']);random.setstate(x['python_rng'])


def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def parent_paused():
    launch=json.loads((MAIN/'launcher.json').read_text());assert not Path('/proc',str(launch['pid'])).exists()
    assert json.loads((MAIN/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT' and (MAIN/'STOP_REQUESTED').exists()
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['boot_id']
    return launch


def prepare():
    if (OUT/'dataset_manifest.json').exists():
        manifest=json.loads((OUT/'dataset_manifest.json').read_text());assert sha(OUT/'train_streams.json.gz')==manifest['train_sha256']
        return manifest
    reserve();parent_paused();source_manifest=json.loads((SOURCE/'dataset_manifest.json').read_text())
    assert sha(SOURCE/'train_streams.json.gz')==source_manifest['files']['train_streams.json.gz']
    all_streams=json.load(gzip.open(SOURCE/'train_streams.json.gz','rt',encoding='utf-8'))
    streams=[s for s in all_streams if s['domain']=='asr/run2b'];assert len(streams)==210
    initialize(streams);counts=Counter();masked=Counter();retained_label_bits={};families=set()
    for s in streams:
        assert s['source_split']=='train';families.add(s['family']);counts['source_rows']+=s['source_rows'];counts['words']+=len(s['words'])
        for seg in s['segments']:counts[seg['language']+':turns']+=1
        for b,k,y,lang in gaps(s):
            if b in s['conflict_mask'][str(k)]:masked[lang+':'+str(y)]+=1;continue
            counts[lang+':k'+str(k)+':'+str(y)]+=1
            h=key(s,b,k);retained_label_bits[h]=retained_label_bits.get(h,0)|(1<<y);assert retained_label_bits[h]!=3
    assert counts['words']==169011
    for n in ['dev_calibration.json.gz','dev_monitor.json.gz','dev_full.json.gz']:
        dev=json.load(gzip.open(MAIN/n,'rt',encoding='utf-8'))
        assert families.isdisjoint({s['family'] for s in dev})
        if n!='dev_full.json.gz' and not (OUT/n).exists():os.link(MAIN/n,OUT/n)
    canonical=lambda value:re.sub(r'-r\d+-\d+$','',re.sub(r'\.(?:vi|en)$','',value))
    canonical_train={canonical(f) for f in families}
    dev_full=json.load(gzip.open(MAIN/'dev_full.json.gz','rt',encoding='utf-8'))
    assert canonical_train.isdisjoint({canonical(s['family']) for s in dev_full if s['domain']=='asr'})
    held_source=json.load(gzip.open(ROOT/'outputs/malbert_uploaded_test_20261105_v1/prepared_streams.json.gz','rt',encoding='utf-8'))
    assert canonical_train.isdisjoint({'asr:'+canonical(s['dialogue_id']) for s in held_source if s['domain']=='asr'})
    with gzip.open(OUT/'train_streams.json.gz','wt',encoding='utf-8') as f:json.dump(clean(streams),f,ensure_ascii=False,separators=(',',':'))
    expected=sum(v for k,v in counts.items() if ':k' in k)
    manifest=dict(version=1,domain='asr/run2b',languages=['vi','en'],streams=len(streams),families=len(families),counts=dict(counts),
        eligible_points_per_epoch=expected,masked_gaps=dict(masked),train_sha256=sha(OUT/'train_streams.json.gz'),
        source_manifest_sha256=sha(SOURCE/'dataset_manifest.json'),source_train_sha256=sha(SOURCE/'train_streams.json.gz'),
        dev_files={n:sha(OUT/n) for n in ['dev_calibration.json.gz','dev_monitor.json.gz']},
        inherited_conflict_masks_unchanged=True,source_order_and_actual_turn_future_preserved=True,
        synthetic_sentence_repacking=False,source_family_exclusions_reused_from_verified_uploaded_train=True,
        train_families_disjoint_from_all_dev=True,canonical_video_families=len(canonical_train),
        test_source_family_metadata_used_only_for_exclusion=True,test_labels_or_predictions_not_used=True,
        change_scope='ASR-only exposure AND original source-turn ordering instead of v7 synthetic packs; not a single-factor matched-budget comparison',
        label_source='Unambiguous source fullstop silver, not human/audio gold',loss={'NEOT':1.,'EOT':2.},learning_rate=4e-6,
        budget_full_epochs=1,batch=16,code_sha256={n:sha(ROOT/'turn_v2'/n) for n in CODE})
    save('dataset_manifest.json',manifest)
    save('independent_preparation_verification.json',dict(complete=True,streams=len(streams),words=counts['words'],
        all_eligible_gaps_labels_masks_recounted=True,source_subset_unchanged=True,surviving_conflict_keys_label_consistent=True,
        expected_scored_points=expected,train_dev_family_disjoint=True,test_not_used=True))
    return manifest


def step_batch(model,opt,runner,tok,device='cuda'):
    import torch
    from torch.nn import functional as F
    rows,states=runner.rows()
    if not rows:return None
    seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in rows]
    width=max(map(len,seqs));ids=torch.zeros((len(rows),width),dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for j,seq in enumerate(seqs):ids[j,:len(seq)]=torch.as_tensor(seq,device=device);mask[j,:len(seq)]=1
    scored=torch.tensor([r['scored'] for r in rows],dtype=torch.bool,device=device);y=torch.tensor([r['label'] for r in rows],device=device)
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=(device=='cuda')):logits=model(ids,mask)
    logits=logits.float();probs=logits.detach().softmax(-1)[:,1].cpu().tolist();updated=bool(scored.any())
    if updated:
        loss=F.cross_entropy(logits[scored],y[scored],weight=logits.new_tensor([1.,2.]));assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_value=float(loss.detach())
    else:loss_value=0.
    runner.advance(states,probs,.1)
    return rows,seqs,probs,updated,loss_value


def technical_resume_fixture():
    import torch
    if (OUT/'technical_fixture_verification.json').exists():return
    class Tok:
        cls_token_id=1;sep_token_id=3
        def encode(self,text,add_special_tokens=False):return [4+int(w[1:]) for w in text.split()]
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(32,8);self.drop=torch.nn.Dropout(.2);self.head=torch.nn.Linear(8,2)
        def forward(self,ids,mask):return self.head(self.drop((self.emb(ids)*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]))
    source=initialize([dict(key='fixture:'+str(i),domain='fixture',source_rows=0,source_split='train',family=str(i),
        words=['w'+str(j) for j in range(1,12)],eots=[4,8,11],unknown=[2],conflict_mask={'1':[6],'2':[]},segments=[dict(start=0,end=11,language='vi')]) for i in range(3)])
    order=np.asarray([1,4,0,3,5,2]);torch.manual_seed(917);m=Tiny();opt=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01)
    initial=copy.deepcopy(m.state_dict());initial_opt=copy.deepcopy(opt.state_dict());initial_rng=torch.get_rng_state().clone()
    def run(interrupt):
        m=Tiny();m.load_state_dict(initial);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);o.load_state_dict(copy.deepcopy(initial_opt));torch.set_rng_state(initial_rng)
        runner=StreamOrder(source,order,lanes=2);updates=seen=0
        while True:
            result=step_batch(m,o,runner,Tok(),'cpu')
            if result is None:break
            rs,_,_,u,_=result;updates+=int(u);seen+=sum(r['scored'] for r in rs)
            if interrupt and updates==3:
                state=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),runner=runner.state(),rng=torch.get_rng_state(),updates=updates,seen=seen))
                m=Tiny();m.load_state_dict(state['model']);o=torch.optim.AdamW(m.parameters(),lr=4e-6,weight_decay=.01);o.load_state_dict(state['optimizer']);runner=StreamOrder(source,order,state['runner'],lanes=2);torch.set_rng_state(state['rng']);interrupt=False
        return dict(model=m.state_dict(),optimizer=o.state_dict(),runner=runner.state(),rng=torch.get_rng_state(),updates=updates,seen=seen)
    a,b=run(False),run(True);assert equal(a,b)
    save('technical_fixture_verification.json',dict(complete=True,technical_only=True,actual_step_batch_and_source_order_used=True,
        interrupted_active_lane_resume_bitwise_matches=True,model_optimizer_torch_rng_cursor_lanes_equal=True,updates=a['updates'],scored_points=a['seen']))


def fingerprint(manifest):
    return dict(dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),train_sha256=manifest['train_sha256'],
                code_sha256={n:sha(ROOT/'turn_v2'/n) for n in CODE})


def initialize_child(manifest):
    import torch
    path=OUT/'runs/finetune/latest_resume.pt'
    if path.exists():return
    parent_paused();cp=json.loads((BASE/'checkpoint_receipt.json').read_text())
    assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
    frozen=OUT/'frozen';frozen.mkdir(exist_ok=True);pin=frozen/'parent_full_resume.pt';os.link(MAIN/'runs/finetune/latest_resume.pt',pin)
    x=torch.load(pin,map_location='cpu',weights_only=False,mmap=True)
    assert x['step']==3731173 and len(x['optimizer']['state'])==27 and not x['quant_scales']
    assert x['parameters']==11941122 and {int(v['step']) for v in x['optimizer']['state'].values()}=={x['step']}
    for n,h in x['fingerprint']['code_sha256'].items():assert sha(ROOT/'turn_v2'/n)==h
    sampler=np.random.default_rng();sampler.bit_generator.state=copy.deepcopy(x['sampler_rng']);order=sampler.permutation(manifest['streams']*2)
    child=dict(x);child.update(parent_step=x['step'],parent_fingerprint=x['fingerprint'],step=x['step'],branch_step=0,
        epoch=0,order=order,cursor=0,stream_state=None,sampler_rng=sampler.bit_generator.state,fingerprint=fingerprint(manifest),
        scored_seen=0,seen_by_key={},training_trace=[],threshold=.1,training_domain='asr/run2b',budget_full_epochs=1)
    atomic_torch(path,child);os.link(path,frozen/'initial_child_resume.pt')
    actual=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    fields=['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng']
    assert all(equal(x[k],actual[k]) for k in fields)
    assert sorted(map(int,actual['order']))==list(range(manifest['streams']*2))
    for name in ['dev_calibration.json.gz','dev_monitor.json.gz']:assert sha(OUT/name)==manifest['dev_files'][name]
    baseline=json.loads((BASE/'frozen_dev.json').read_text());assert baseline['threshold']==.1
    shutil.copyfile(BASE/'frozen_dev.json',OUT/'baseline_dev.json')
    save('migration_receipt.json',dict(complete=True,parent_step=x['step'],parent_full_sha256=sha(pin),initial_child_sha256=sha(path),
        preserved_exact_fields=fields,optimizer_entries=27,parameters=x['parameters'],new_asr_order_and_fresh_lanes_explicitly_versioned=True,
        parent_order_cursor_lanes_sampler_kept_in_immutable_full_pin=True,baseline_reused_not_rescored=True,
        baseline_dev_sha256=sha(OUT/'baseline_dev.json'),shared_training_threshold=.1,base_reset=False,test_used=False))


def make_model():
    from transformers import AlbertConfig
    from .malbert_turn_student_v1 import MalbertTurnStudent
    m=MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    assert m.parameter_count()==11941122<20000000
    return m.cuda()


def train(manifest,streams,tok):
    import torch
    if (OUT/'training_report.json').exists():return
    path=OUT/'runs/finetune/latest_resume.pt';x=torch.load(path,map_location='cpu',weights_only=False)
    assert x['fingerprint']==fingerprint(manifest)
    model=make_model();model.load_state_dict(x['model'],strict=True);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=4e-6,weight_decay=.01);opt.load_state_dict(x['optimizer'])
    assert all(abs(g['lr']-4e-6)<1e-15 and g['weight_decay']==.01 for g in opt.param_groups)
    order=x['order'];sampler=x['sampler_rng'];runner=StreamOrder(streams,order,x['stream_state'],lanes=16)
    step=x['step'];branch=x['branch_step'];seen=x['scored_seen'];by=Counter(x['seen_by_key']);trace=x.get('training_trace',[]);restore_rng(x)
    parent_step=x['parent_step'];del x
    first=OUT/'frozen/first_100_resume.pt'
    if branch==100 and not first.exists():os.link(path,first)
    if first.exists() and not (OUT/'first_training_trace.json').exists():
        first_state=torch.load(first,map_location='cpu',weights_only=False,mmap=True);save('first_training_trace.json',first_state['training_trace']);del first_state
    def checkpoint(epoch=0):
        value=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),step=step,branch_step=branch,
            parent_step=parent_step,parameters=11941122,epoch=epoch,order=order,cursor=runner.next,stream_state=runner.state(),
            sampler_rng=sampler,scored_seen=seen,seen_by_key=dict(by),training_trace=trace,quant_scales={},threshold=.1,
            loss_policy={'NEOT':1.,'EOT':2.},training_domain='asr/run2b',budget_full_epochs=1,fingerprint=fingerprint(manifest),**rng_state())
        atomic_torch(path,value)
        save('resume_receipt.json',dict(step=step,branch_step=branch,epoch=epoch,scored_seen=seen,expected_scored_points=manifest['eligible_points_per_epoch'],
            optimizer_entries=len(opt.state),cursor=runner.next,active_lanes=len(runner.slots),checkpoint_sha256=sha(path)))
    while True:
        if STOP or (OUT/'STOP_REQUESTED').exists():checkpoint();raise InterruptedError('User stop saved full ASR adaptation state')
        result=step_batch(model,opt,runner,tok)
        if result is None:break
        rows,seqs,probs,updated,loss=result
        for r,seq,prob in zip(rows,seqs,probs):
            if r['scored']:seen+=1;by[r['language']+':k'+str(r['k'])+':'+str(r['label'])]+=1
            if branch<100:trace.append(dict(stream=r['stream'],boundary=r['boundary'],k=r['k'],start=r['start'],label=r['label'],scored=r['scored'],
                p_eot=prob,input_ids_sha256=hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()))
        if updated:step+=1;branch+=1
        if updated and branch%50==0:status('TRAINING_ASR_ONLY',step=step,branch_step=branch,scored_seen=seen,expected_points=manifest['eligible_points_per_epoch'],loss=loss,epoch_display=1)
        if updated and (branch==100 or branch%500==0):
            checkpoint()
            if branch==100:
                os.link(path,OUT/'frozen/first_100_resume.pt');save('first_training_trace.json',trace)
    assert seen==manifest['eligible_points_per_epoch']
    assert dict(by)=={k:v for k,v in manifest['counts'].items() if ':k' in k}
    assert runner.next==len(order) and not runner.slots
    assert {int(v['step']) for v in opt.state.values()}=={step} and step==parent_step+branch
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    checkpoint(epoch=1);atomic_torch(OUT/'runs/finetune/weights.pt',{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    save('epoch_0001_coverage.json',dict(complete=True,expected=manifest['eligible_points_per_epoch'],seen=seen,by_key=dict(by),all_420_tasks_completed=True))
    save('training_report.json',dict(complete=True,full_epochs=1,updates=branch,parent_step=parent_step,global_step=step,scored_seen=seen,
        optimizer_entries=len(opt.state),weights_sha256=sha(OUT/'runs/finetune/weights.pt'),full_resume_sha256=sha(path),test_used=False))
    del model,opt;gc.collect();torch.cuda.empty_cache()


def audit_training(streams,tok):
    import torch
    trace=json.loads((OUT/'first_training_trace.json').read_text());init=torch.load(OUT/'frozen/initial_child_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    after=torch.load(OUT/'frozen/first_100_resume.pt',map_location='cpu',weights_only=False,mmap=True)
    runner=StreamOrder(streams,init['order'],lanes=16);offset=0;updates=0
    while offset<len(trace):
        rows,states=runner.rows();part=trace[offset:offset+len(rows)];assert len(part)==len(rows)
        for r,saved in zip(rows,part):
            assert all(saved[k]==r[k] for k in ['stream','boundary','k','label','scored']) and saved['start']==r['start']
            seq=encode_pair(tok,r['context'],r['text'],256)['input_ids']
            assert hashlib.sha256(np.asarray(seq,dtype='<u4').tobytes()).hexdigest()==saved['input_ids_sha256']
        updates+=int(any(r['scored'] for r in rows));runner.advance(states,[r['p_eot'] for r in part],.1);offset+=len(rows)
    assert updates==100 and equal(runner.state(),after['stream_state'])
    assert len(after['optimizer']['state'])==27 and {int(v['step']) for v in after['optimizer']['state'].values()}=={init['step']+100}
    assert all(torch.isfinite(v).all() for v in after['model'].values()) and any(not torch.equal(v,init['model'][k]) for k,v in after['model'].items())
    save('independent_actual_training_verification.json',dict(complete=True,actual_training_inputs=len(trace),updates=100,labels_masks_encoding_causal_future_resets_verified=True,
        checkpoint_optimizer_rng_order_lanes_present=True,weights_finite_and_changed=True,trace_sha256=sha(OUT/'first_training_trace.json')))


def evaluate(manifest,tok):
    import torch
    from . import uploaded_vi_en_sentence_mix_evaluation_v7 as ev
    if (OUT/'dev_report.json').exists():return json.loads((OUT/'dev_report.json').read_text())
    model=make_model();model.load_state_dict(torch.load(OUT/'runs/finetune/weights.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    calibration=json.load(gzip.open(OUT/'dev_calibration.json.gz','rt',encoding='utf-8'));monitor=json.load(gzip.open(OUT/'dev_monitor.json.gz','rt',encoding='utf-8'))
    ev.OUT=OUT;ev.save=save;ev.status=status;predict=ev.predictor(model,tok)
    curves=[]
    for t in np.linspace(.1,.9,9):
        name='calibration_%.1f.json'%t
        if (OUT/name).exists():r=json.loads((OUT/name).read_text())
        else:
            status('CALIBRATING_ASR_ADAPTED_MODEL',threshold=float(t));r=ev.rollout(predict,calibration,float(t));save(name,r)
        curves.append(r)
    selected=max(curves,key=lambda r:(r['score'],sum(m['eot_f1'] or 0 for m in r['by_language'].values())))
    status('SCORING_FIXED_ALL_DOMAIN_DEV',threshold=selected['threshold'])
    chosen=ev.rollout(predict,monitor,selected['threshold'],'selected_dev_trace.jsonl.gz')
    if selected['threshold']==.1:fixed=chosen;fixed_trace='selected_dev_trace.jsonl.gz'
    else:
        status('SCORING_FIXED_BASELINE_THRESHOLD_DEV',threshold=.1);fixed=ev.rollout(predict,monitor,.1,'fixed_threshold_dev_trace.jsonl.gz');fixed_trace='fixed_threshold_dev_trace.jsonl.gz'
    result=dict(complete=True,selected=chosen,fixed_baseline_threshold=fixed,fixed_threshold_trace=fixed_trace,
        calibration=[dict(threshold=r['threshold'],by_language=r['by_language']) for r in curves],
        threshold_policy_unchanged=True,monitor_not_used_to_choose_threshold=True,test_used=False,
        weights_sha256=sha(OUT/'runs/finetune/weights.pt'),train_full_epoch_sha256=sha(OUT/'epoch_0001_coverage.json'))
    save('dev_report.json',result);del model;gc.collect();torch.cuda.empty_cache();return result


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False;u.is_librosa_available=im.is_librosa_available=lambda:False
    from transformers import AutoTokenizer
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5)
    assert not (OUT/'report.json').exists(),'Never rerun a complete ASR experiment'
    parent_paused();reserve();status('PREPARING_ASR_ONLY_TRAIN');manifest=prepare();technical_resume_fixture();initialize_child(manifest)
    streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
    tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    train(manifest,streams,tok);audit_training(streams,tok);dev=evaluate(manifest,tok)
    baseline=json.loads((OUT/'baseline_dev.json').read_text());before={(x['domain'],x['language']):x for x in baseline['by_domain']}
    comparisons=[]
    for mode in ['selected','fixed_baseline_threshold']:
        for x in dev[mode]['by_domain']:
            old=before[x['domain'],x['language']]
            comparisons.append(dict(mode=mode,domain=x['domain'],language=x['language'],before=old,after=x,
                f1_delta_pp=100*((x['eot_f1'] or 0)-(old['eot_f1'] or 0))))
    parent_paused();cp=json.loads((BASE/'checkpoint_receipt.json').read_text());assert sha(MAIN/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
    report=dict(complete=True,training=json.loads((OUT/'training_report.json').read_text()),dev=dev,comparison=comparisons,
        baseline_step=3731173,baseline_dev_sha256=sha(OUT/'baseline_dev.json'),baseline_reused_not_rescored=True,
        original_v7_remains_paused=True,no_model_default_replacement=True,no_test_scored=True,
        change_scope=manifest['change_scope'],under20m=True,not_int4=True)
    save('report.json',report);status('COMPLETE',report_sha256=sha(OUT/'report.json'))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True);os.nice(10)
    def request_stop(*args):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except InterruptedError as exc:status('STOPPED_WITH_CHECKPOINT',reason=str(exc))
        except Exception as exc:
            save('error.json',dict(error=repr(exc),traceback=traceback.format_exc(),at_utc=datetime.now(timezone.utc).isoformat()));status('ERROR',error=repr(exc));raise
