"""Warm-start VI/EN training with nested whole-family curriculum and full-state resume."""
from collections import Counter
import copy
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np

from .uploaded_vi_en_curriculum_data_v6 import ROOT, OUT, PARENT, LANGS, MODULES, StreamOrder, initialize, prepare, reserve, save, sha, status, make_order, check_order
from .uploaded_vi_en_curriculum_evaluation_v6 import evaluate, predictor, retained, rollout
from .tiny_compression_v1.common import encode_pair

STOP=False
PARAMETERS=11941122
PRETRAINED_SHA='3040f56fef13a7c0d6b0e6f608daf9d8cd8cc4f4a163844d652d0b3e80cbdc84'
TEACHER_SHA='15d4ebfc7fcba7c3c842b7b7815ac309cf349f0436dca69522d9d529295954cc'
OLD=ROOT/'outputs/multilingual_tiny_compression_20261102_v1'


def stopped():
    return STOP or (OUT/'STOP_REQUESTED').exists()


def atomic_torch(path, value):
    import torch
    reserve();path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    if path.exists() and path.name=='latest_resume.pt':
        link=path.with_name('previous_resume.link');assert not link.exists()
        os.link(path,link);link.replace(path.with_name('previous_resume.pt'))
    tmp.replace(path)


def equal(a,b):
    import torch
    if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def fingerprint(manifest):
    return dict(dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),
        train_streams_sha256=manifest['files']['train_streams.json.gz'],
        active_languages=list(LANGS),parameters=PARAMETERS,drive_replay_rows=0,
        curriculum_plan_sha256=sha(OUT/'curriculum_plan.json'),
        code_sha256={n:sha(ROOT/'turn_v2'/n) for n in MODULES})


def migrate(manifest):
    import torch
    if (OUT/'migration_receipt.json').exists():
        return json.loads((OUT/'migration_receipt.json').read_text())
    launch=json.loads((PARENT/'launcher.json').read_text())
    assert launch['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert not Path('/proc',str(launch['pid'])).exists(), 'Parent must checkpoint and exit'
    assert (PARENT/'STOP_REQUESTED').exists()
    assert json.loads((PARENT/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
    assert json.loads((OUT/'independent_preparation_verification.json').read_text())['complete']
    for name in ['pretrained','tokenizer']:
        if not (OUT/name).exists():
            shutil.copytree(PARENT/name,OUT/name,copy_function=os.link)
    frozen=OUT/'frozen'
    frozen.mkdir(exist_ok=True)
    pin=frozen/'parent_full_resume.pt'
    if not pin.exists():
        os.link(PARENT/'runs/finetune/latest_resume.pt',pin)
    parent=torch.load(pin,map_location='cpu',weights_only=False,mmap=True)
    assert parent['parameters']==PARAMETERS and len(parent['optimizer']['state'])==27 and not parent['quant_scales']
    assert {int(value['step']) for value in parent['optimizer']['state'].values()}=={parent['step']}
    assert sha(PARENT/'dataset_manifest.json')==parent['fingerprint']['dataset_manifest_sha256']
    assert sha(OUT/'train_streams.json.gz')==parent['fingerprint']['train_streams_sha256']
    assert parent['loss_policy']=={'NEOT':1.,'EOT':2.}
    for name,digest in parent['fingerprint']['code_sha256'].items():
        assert sha(ROOT/'turn_v2'/name)==digest
    plan=json.loads((OUT/'curriculum_plan.json').read_text())
    rng=np.random.default_rng()
    rng.bit_generator.state=copy.deepcopy(parent['sampler_rng'])
    order=make_order(plan['stages'][0],rng)
    child=dict(parent)
    child.update(branch_step=0,parent_step=parent['step'],epoch=0,epoch_scored_seen=0,
        order=order,cursor=0,stream_state=None,sampler_rng=rng.bit_generator.state,
        curriculum_stage=0,curriculum_stage_step=0,curriculum_stage_seen=0,curriculum_stage_pass=0,
        cutover_partial_epoch=False,partial_epoch_expected=0,parent_epoch=parent['epoch'],
        parent_uploaded_rows_seen=parent['uploaded_rows_seen'],best_key=(-1.,-1.,-1.),
        fingerprint=fingerprint(manifest),loss_policy={'NEOT':1.,'EOT':2.})
    path=OUT/'runs/finetune/latest_resume.pt'
    if path.exists():
        existing=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert existing['branch_step']==0 and existing['parent_step']==parent['step']
        assert equal(existing,child), 'An interrupted migration must preserve the exact child'
    else:
        atomic_torch(path,child)
    initial=frozen/'initial_child_resume.pt'
    if not initial.exists():
        os.link(path,initial)
    actual=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    preserved=['model','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','best_threshold']
    assert all(equal(parent[key],actual[key]) for key in preserved)
    assert equal(actual['order'],order) and equal(actual['sampler_rng'],rng.bit_generator.state)
    check_order(plan['stages'][0],actual['order'])
    receipt=dict(complete=True,parent_step=parent['step'],parent_branch_step=parent['branch_step'],
        parent_epoch=parent['epoch'],parent_full_sha256=sha(pin),initial_child_full_sha256=sha(path),
        parameters=PARAMETERS,optimizer_entries=27,preserved_fields=preserved,
        parent_order_cursor_lanes_and_sampler_retained_in_immutable_full_pin=True,
        new_order_cursor_and_fresh_conversation_lanes_explicitly_versioned=True,
        sampler_advanced_only_by_deterministic_initial_stage_permutation=True,
        curriculum_stage=0,initial_stage_families=plan['stages'][0]['family_count'],
        active_loss={'NEOT':1.,'EOT':2.},base_model_reset=False,test_used=False,
        full_filtered_epochs_complete=0,partial_parent_tail_archived_not_counted_as_new_epoch=True)
    save('migration_receipt.json',receipt)
    return receipt


def new_model():
    from transformers import AlbertConfig
    from .malbert_turn_student_v1 import MalbertTurnStudent
    assert sha(OUT/'pretrained/model.safetensors')==PRETRAINED_SHA
    model=MalbertTurnStudent(AlbertConfig.from_pretrained(OUT/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    assert model.parameter_count()==PARAMETERS<20_000_000
    return model


def baseline(calibration,monitor):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    p=OUT/'teacher_new_dev.json'
    if p.exists():return json.loads(p.read_text())
    assert sha(OLD/'teacher/model.safetensors')==TEACHER_SHA
    status('EVALUATING_FROZEN_CONTROL_ON_NEW_DEV',not_training_old_teacher=True)
    tok=AutoTokenizer.from_pretrained(OLD/'teacher',local_files_only=True,token=False)
    teacher=AutoModelForSequenceClassification.from_pretrained(OLD/'teacher',local_files_only=True,token=False).cuda().eval()
    result=evaluate(predictor(teacher,tok,teacher=True),calibration,monitor,'teacher_new_dev')
    result['frozen_teacher_sha256']=TEACHER_SHA;save('teacher_new_dev.json',result)
    del teacher;gc.collect();torch.cuda.empty_cache();return result


def fit(name,manifest,streams,calibration,monitor,teacher,initial=None,group=None,min_epochs=6):
    import torch
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from .multilingual_malbert_vi_en_checked_train_v3 import attach_qat,latent
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True);path=folder/'latest_resume.pt'
    plan=json.loads((OUT/'curriculum_plan.json').read_text())
    if (folder/'training_report.json').exists():return folder
    tok=AutoTokenizer.from_pretrained(OUT/'tokenizer',local_files_only=True,token=False)
    model=new_model().cuda();rng=np.random.default_rng(43)
    if path.exists():x=torch.load(path,map_location='cpu',weights_only=False)
    else:
        assert initial is not None
        x=dict(model=torch.load(initial,map_location='cpu',weights_only=True),step=0,branch_step=0,parent_step=0,
               epoch=0,cursor=0,order=make_order(plan['stages'][3],rng),sampler_rng=rng.bit_generator.state,stream_state=None,
               curriculum_stage=3,curriculum_stage_step=0,curriculum_stage_seen=0,curriculum_stage_pass=0,
               uploaded_rows_seen=0,uploaded_origin_exposures={},epoch_scored_seen=0,best_key=(-1.,-1.,-1.),best_threshold=.5,
               fingerprint=fingerprint(manifest),parameters=PARAMETERS,quant_scales={})
    assert x['fingerprint']==fingerprint(manifest) and x['parameters']==PARAMETERS
    model.load_state_dict(x['model'],strict=True)
    q=attach_qat(model,group,x['quant_scales'] or None) if group else {}
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
    if 'optimizer' in x:opt.load_state_dict(x['optimizer'])
    rng.bit_generator.state=x['sampler_rng'];epoch=x['epoch'];step=x['step'];branch=x.get('branch_step',0)
    parent_step=x.get('parent_step',0);order=x['order'];best=tuple(x['best_key']);threshold=x.get('best_threshold',.5)
    total_seen=x.get('uploaded_rows_seen',0);origins=Counter(x.get('uploaded_origin_exposures',{}));epoch_seen=x.get('epoch_scored_seen',0)
    partial_tail=x.get('cutover_partial_epoch',False);partial_expected=x.get('partial_epoch_expected',0);parent_epoch=x.get('parent_epoch',0);parent_exposures=x.get('parent_uploaded_rows_seen',0)
    stage_index=x['curriculum_stage'];stage_step=x['curriculum_stage_step']
    stage_seen=x['curriculum_stage_seen'];stage_pass=x['curriculum_stage_pass']
    stage=plan['stages'][stage_index];check_order(stage,order)
    runner=StreamOrder(streams,order,x.get('stream_state'),lanes=16)
    for key,setter in [('torch_rng',torch.set_rng_state),('cuda_rng',torch.cuda.set_rng_state),
                       ('numpy_rng',np.random.set_state),('python_rng',random.setstate)]:
        if key in x:setter(x[key])
    del x
    current_lr=opt.param_groups[0]['lr'];passed_best=best[0]>.5
    exposures_at_start=total_seen
    training_trace=[]
    def checkpoint():
        full=dict(model=latent(model,q),optimizer=opt.state_dict(),step=step,branch_step=branch,parent_step=parent_step,
            epoch=epoch,cursor=runner.next,order=order,sampler_rng=rng.bit_generator.state,stream_state=runner.state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            best_key=best,best_threshold=threshold,parameters=PARAMETERS,fingerprint=fingerprint(manifest),
            quant_scales={k:v.scale.detach().cpu() for k,v in q.items()},uploaded_rows_seen=total_seen,
            uploaded_origin_exposures=dict(origins),epoch_scored_seen=epoch_seen,drive_replay_rows=0,
            cutover_partial_epoch=partial_tail,partial_epoch_expected=partial_expected,parent_epoch=parent_epoch,
            parent_uploaded_rows_seen=parent_exposures,loss_policy={'NEOT':1.,'EOT':2.},
            curriculum_stage=stage_index,curriculum_stage_step=stage_step,
            curriculum_stage_seen=stage_seen,curriculum_stage_pass=stage_pass)
        atomic_torch(path,full)
        save(str(folder.relative_to(OUT)/'resume_receipt.json'),dict(step=step,branch_step=branch,epoch=epoch,
            tasks_assigned=runner.next,active_lanes=len(runner.slots),optimizer_entries=len(opt.state),
            uploaded_rows_seen=total_seen,uploaded_origin_exposures=dict(origins),parameters=PARAMETERS,
            source_order_cursor_and_all_rng_preserved=True,drive_replay_rows=0,
            curriculum_stage=stage_index,curriculum_family_fraction=stage['fraction_of_families'],
            curriculum_stage_step=stage_step,curriculum_stage_seen=stage_seen,
            full_filtered_epochs_complete=epoch))
        if branch>=100 and not (OUT/'independent_training_verification.json').exists():
            y=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert len(y['optimizer']['state'])==27 and y['step']>parent_step and y['uploaded_rows_seen']>exposures_at_start
            assert y['fingerprint']==fingerprint(manifest)
            check_order(stage,y['order'])
            assert {int(v['step']) for v in y['optimizer']['state'].values()}=={y['step']}
            save('independent_training_verification.json',dict(verified=True,step=step,branch_step=branch,epoch=epoch,
                uploaded_rows_seen=total_seen,uploaded_origin_exposures=dict(origins),optimizer_entries=27,
                checkpoint_sha256=sha(path),full_resume_load_passed=True,parameters=PARAMETERS,drive_replay_rows=0,
                causal_predicted_reset_stream_state_preserved=True,all_sources_trained_once=epoch>=1,
                curriculum_stage=stage_index,active_stage_families=stage['family_count'],
                active_stage_streams=len(stage['stream_indices']),curriculum_stage_step=stage_step))
            save('first_training_causal_trace.json',training_trace)
    def eval_now():
        nonlocal best,passed_best,threshold
        model.eval();status('EVALUATING_NEW_DEV',run=name,step=step,branch_step=branch,epoch=epoch+1)
        result=evaluate(predictor(model,tok),calibration,monitor,'runs/'+name+'/dev_%08d'%branch)
        passed,checks=retained(result,teacher)
        record=dict(run=name,step=step,branch_step=branch,epoch=epoch+1,dev=result,
                    retention_passed=passed,retention_by_language=checks,uploaded_rows_seen=total_seen,
                    curriculum_stage=stage_index,curriculum_family_fraction=stage['fraction_of_families'],
                    curriculum_stage_step=stage_step,full_filtered_epochs_complete=epoch,
                    uploaded_origin_exposures=dict(origins),test_used=False,new_uploaded_only_dev=True)
        save('runs/'+name+'/dev_step_%08d.json'%branch,record)
        key=(float(passed),result['score'],float(np.mean([v['eot_f1'] or 0 for v in result['by_language'].values()])))
        threshold=result['threshold']
        if key>best and branch>0:
            best=key;passed_best=passed;atomic_torch(folder/'student.pt',latent(model,q));save('runs/'+name+'/best_report.json',record)
            if q:atomic_torch(folder/'quant_params.pt',dict(group_size=group,scales={k:v.scale.detach().cpu() for k,v in q.items()}))
        save('latest_dev.json',record);model.train();checkpoint();return record
    if branch==0 and not (OUT/'migration_receipt.json').exists():
        # These new-DEV results are a warm-start baseline, never reported as new-data improvement.
        eval_now();save('warm_start_new_dev_baseline.json',json.loads((OUT/'latest_dev.json').read_text()))
    while stage_index<3 or epoch<min_epochs or (name=='finetune' and not passed_best):
        if stopped():
            checkpoint()
            raise InterruptedError('User stop with full model/optimizer/RNG/curriculum checkpoint')
        if stage_index<3 and stage_step>=stage['update_budget']:
            selected=folder/'student.pt'
            archived=folder/('curriculum_stage_%02d_best.pt'%stage_index)
            if selected.exists() and not archived.exists():
                os.link(selected,archived)
            selected_report=folder/'best_report.json'
            if selected_report.exists():
                save('runs/'+name+'/curriculum_stage_%02d_best_report.json'%stage_index,
                     json.loads(selected_report.read_text()))
            save('runs/'+name+'/curriculum_stage_%02d_complete.json'%stage_index,
                 dict(complete=True,stage=stage_index,family_fraction=stage['fraction_of_families'],
                      optimizer_updates=stage_step,scored_exposures=stage_seen,
                      complete_subset_passes=stage_pass,partial_pass_scored_points=epoch_seen,
                      full_uploaded_epoch=False,step=step,branch_step=branch))
            stage_index+=1
            stage=plan['stages'][stage_index]
            stage_step=0;stage_seen=0;stage_pass=0;epoch_seen=0
            best=(-1.,-1.,-1.);passed_best=False
            order=make_order(stage,rng)
            runner=StreamOrder(streams,order,lanes=16)
            checkpoint()
            status('CURRICULUM_STAGE_OPENED',run=name,step=step,branch_step=branch,
                   curriculum_stage=stage_index,family_fraction=stage['fraction_of_families'],
                   families=stage['family_count'],eligible_points_per_pass=stage['eligible_points_per_pass'])
        model.train();rows,states=runner.rows()
        if not rows:
            expected=stage['eligible_points_per_pass']
            assert epoch_seen==expected,(epoch_seen,expected)
            stage_pass+=1
            coverage=dict(complete=True,curriculum_stage=stage_index,subset_pass=stage_pass,
                          scored_points=epoch_seen,expected_eligible_points=expected,
                          all_active_subset_eligible_gaps_visited=True,full_uploaded_epoch=stage_index==3,
                          drive_replay_rows=0)
            save('runs/'+name+'/stage_%02d_pass_%04d_coverage.json'%(stage_index,stage_pass),coverage)
            if stage_index==3:
                epoch+=1
                coverage['all_filtered_eligible_gaps_visited']=True
                save('runs/'+name+'/epoch_%04d_coverage.json'%epoch,coverage)
            order=make_order(stage,rng);runner=StreamOrder(streams,order,lanes=16)
            epoch_seen=0;checkpoint();continue
        if branch%100==0:reserve()
        seqs=[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in rows];width=max(map(len,seqs))
        ids=torch.zeros((len(rows),width),dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for j,seq in enumerate(seqs):ids[j,:len(seq)]=torch.as_tensor(seq,device='cuda');mask[j,:len(seq)]=1
        scored=torch.tensor([r['scored'] for r in rows],device='cuda',dtype=torch.bool)
        y=torch.tensor([r['label'] for r in rows],device='cuda')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not bool(group)):logits=model(ids,mask)
        logits=logits.float();probs=logits.detach().softmax(-1)[:,1].cpu().tolist()
        if name=='finetune' and branch<100:
            import hashlib
            for r,p,seq in zip(rows,probs,seqs):
                training_trace.append(dict(stream=r['stream'],boundary=r['boundary'],k=r['k'],start=r['start'],
                    label=r['label'],scored=r['scored'],origin=r['origin'],p_eot=p,threshold=threshold,
                    input_ids_sha256=hashlib.sha256(np.asarray(seq,dtype='<u2').tobytes()).hexdigest()))
        updated=bool(scored.any())
        if updated:
            loss=F.cross_entropy(logits[scored],y[scored],weight=logits.new_tensor([1.,2.]));assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            # Preserve the inherited learning rate and AdamW state across the source migration.
            opt.step();step+=1;branch+=1;stage_step+=1
        else:loss=logits.sum()*0
        runner.advance(states,probs,threshold)
        for r in rows:
            if r['scored']:origins[r['origin']]+=1;total_seen+=1;epoch_seen+=1;stage_seen+=1
        if updated and branch and branch%50==0:
            status('QAT_UPLOADED_MALBERT_INT4' if group else 'TRAINING_UPLOADED_ONLY_MALBERT',run=name,
                step=step,branch_step=branch,epoch=epoch+1,uploaded_rows_seen=total_seen,
                uploaded_origin_exposures=dict(origins),loss=float(loss.detach()),learning_rate=current_lr,
                parameters=PARAMETERS,group_size=group,drive_replay_rows=0,loss_policy={'NEOT':1.,'EOT':2.},
                curriculum_stage=stage_index,curriculum_family_fraction=stage['fraction_of_families'],
                curriculum_stage_step=stage_step,curriculum_stage_budget=stage['update_budget'],
                curriculum_stage_seen=stage_seen,full_filtered_epochs_complete=epoch,
                tasks_assigned=runner.next,total_tasks=len(order))
        if updated and (branch==100 or (branch and branch%500==0)):checkpoint()
        if updated and branch and branch%10000==0:eval_now()
    save('runs/'+name+'/training_report.json',dict(complete=True,epochs=epoch,step=step,branch_step=branch,
        uploaded_rows_seen=total_seen,selected=json.loads((folder/'best_report.json').read_text()),test_used=False))
    del model,opt;gc.collect();torch.cuda.empty_cache();return folder


def export_candidate(folder,name,bits,calibration,monitor,teacher):
    import torch
    from transformers import AutoTokenizer
    from .malbert_turn_export_v2 import export
    from .multilingual_tiny_compress_v3 import ort_predictor
    if (OUT/'candidates'/name/'dev_report.json').exists():return json.loads((OUT/'candidates'/name/'dev_report.json').read_text())
    tok=AutoTokenizer.from_pretrained(OUT/'tokenizer',local_files_only=True,token=False)
    model=new_model();model.load_state_dict(torch.load(folder/'student.pt',map_location='cpu',weights_only=True),strict=True)
    probes=[]; languages=set()
    for s in monitor:
        for p in s['segments']:
            b=p['start']+1
            if b+2<=p['end']:
                probes.append(dict(context=' '.join(s['words'][:b]),text=' '.join(s['words'][b:b+2]),language=p['language']))
                languages.add(p['language'])
            if len(probes)>=128 and languages==set(LANGS):break
        if len(probes)>=128 and languages==set(LANGS):break
    path=OUT/'candidates'/name;quant=torch.load(folder/'quant_params.pt',map_location='cpu',weights_only=True) if bits==4 else None
    assert json.loads((folder/'best_report.json').read_text())['branch_step']>0,'A selected trained artifact is required'
    export(model,path,OUT/'tokenizer',probes,tok,bits=bits,quant=quant);del model;gc.collect()
    threshold=json.loads((folder/'best_report.json').read_text())['dev']['threshold']
    cpu=ort_predictor(path,tok)
    def p(rows):return list(cpu(rows)),[encode_pair(tok,r['context'],r['text'],256)['input_ids'] for r in rows]
    result=rollout(p,monitor,threshold,'cpu_'+name+'_dev_trace.jsonl.gz');passed,checks=retained(result,teacher)
    record=dict(candidate=name,path=str(path),parameters=PARAMETERS,storage_bits=bits,model_bytes=(path/'model.onnx').stat().st_size,
        model_sha256=sha(path/'model.onnx'),dev=result,retention_passed=passed,retention_by_language=checks,test_used=False)
    save('candidates/'+name+'/dev_report.json',record);return record


def finalize(candidates, monitor):
    passing=[c for c in candidates if c['retention_passed'] and c['parameters']<20_000_000]
    if not passing:
        save('final_report.json',dict(complete=True,accepted=False,reason='No under20M CPU candidate passed unchanged new-DEV retention',candidates=candidates))
        status('COMPLETE_NO_ACCEPTED_UNDER20M_MODEL');return
    winner=min(passing,key=lambda c:c['model_bytes'])
    package=ROOT/'models/turn_vi_en_malbert_curriculum_20261106_v6';target=package/'onnx'
    target.mkdir(parents=True,exist_ok=False);source=Path(winner['path'])
    config=dict(parameters=PARAMETERS,hard_parameter_limit_exclusive=20_000_000,languages=list(LANGS),
        threshold=winner['dev']['threshold'],max_length=256,lookahead_words=[1,2],padding_to_multiple_of=8,
        normalization='NFKC lowercase punctuation-to-space collapse-whitespace',label2id={'NEOT':0,'EOT':1},
        storage_precision='FP32' if winner['storage_bits']==32 else 'INT'+str(winner['storage_bits']),
        compute_precision='FP32',gold_reset_in_inference=False,language_id_supplied=False,
        onnx_session_options={'session.disable_quant_qdq':'1'},smoke_only=False,
        quality_status='UPLOADED_DEV_SILVER_RETENTION_PASS_NOT_HUMAN_AUDIO_GOLD')
    save('candidates/'+winner['candidate']+'/turn_detector_config.json',config)
    shutil.copyfile(ROOT/'turn_v2/tiny_compression_v1/portable_runner.py',source/'run_model.py')
    (source/'requirements.txt').write_text('numpy>=2.1\nonnxruntime==1.23.2\ntokenizers==0.23.3\n')
    names=['model.onnx','config.json','tokenizer.json','tokenizer_config.json','turn_detector_config.json','run_model.py','requirements.txt','export_receipt.json']
    for n in names:shutil.copyfile(source/n,target/n)
    cases=[]
    for s in monitor:
        for p in s['segments']:
            b=p['start']+1
            if b+2<=p['end']:
                cases.append(dict(context=' '.join(s['words'][:b]),text=' '.join(s['words'][b:b+2]),language=p['language'],label='EOT' if b in s['_eots'] else 'NEOT'))
            if len(cases)>=128:break
        if len(cases)>=128:break
    with (OUT/'benchmark_cases.jsonl').open('w') as f:
        for r in cases:f.write(json.dumps(r)+'\n')
    benchmark={}
    for name,kind,path in [('teacher','teacher',OLD/'teacher'),('student','onnx',target)]:
        result=subprocess.run([sys.executable,'-m','turn_v2.multilingual_tiny_benchmark_v2','--kind',kind,'--model',str(path),
            '--cases',str(OUT/'benchmark_cases.jsonl')],cwd=ROOT,capture_output=True,text=True,timeout=600)
        save('benchmark_'+name+'.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        assert result.returncode==0
        benchmark[name]=json.loads(result.stdout.strip().splitlines()[-1])
    files={n:dict(bytes=(target/n).stat().st_size,sha256=sha(target/n)) for n in names}
    model_manifest=dict(winner=winner,files=files,benchmark=benchmark,parameters=PARAMETERS,
        packed_int4=winner['storage_bits']==4,compute_precision='FP32 dequantized packed matrices',
        source_dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),test_used=False,drive_training_rows=0,
        backed_up_to_drive=False,active_default_replaced=False)
    (package/'model_manifest.json').write_text(json.dumps(model_manifest,indent=2)+'\n')
    archive=package.with_suffix('.zip');reserve()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(package.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        import hashlib
        for n,receipt in files.items():assert hashlib.sha256(z.read(package.name+'/onnx/'+n)).hexdigest()==receipt['sha256']
    save('final_report.json',dict(complete=True,accepted=True,winner=winner,candidates=candidates,benchmark=benchmark,
         package=str(package),archive_sha256=sha(archive),manifest_sha256=sha(package/'model_manifest.json'),test_used=False))
    status('COMPLETE',storage_bits=winner['storage_bits'],package=str(package),parameters=PARAMETERS)


def run():
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False
    u.is_librosa_available=im.is_librosa_available=lambda:False
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.5);os.nice(5)
    assert not (OUT/'final_report.json').exists(),'Never restart a completed campaign'
    manifest=prepare();migration=migrate(manifest)
    streams=initialize(json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8')))
    calibration=initialize(json.load(gzip.open(OUT/'dev_calibration.json.gz','rt',encoding='utf-8')))
    monitor=initialize(json.load(gzip.open(OUT/'dev_monitor.json.gz','rt',encoding='utf-8')))
    teacher=baseline(calibration,monitor)
    fine=fit('finetune',manifest,streams,calibration,monitor,teacher)
    candidates=[export_candidate(fine,'finetuned_fp32',32,calibration,monitor,teacher)]
    save('candidate_results.json',candidates)
    assert candidates[0]['retention_passed'],'Preserve training; investigate CPU retention before QAT'
    control=fit('fp32_control',manifest,streams,calibration,monitor,teacher,initial=fine/'student.pt',min_epochs=2)
    for name,bits in [('fp32_control',32),('int8_control',8)]:
        candidates.append(export_candidate(control,name,bits,calibration,monitor,teacher));save('candidate_results.json',candidates)
    for group in (64,32):
        folder=fit('qat_g'+str(group),manifest,streams,calibration,monitor,teacher,initial=fine/'student.pt',group=group,min_epochs=2)
        candidates.append(export_candidate(folder,'int4_g'+str(group),4,calibration,monitor,teacher));save('candidate_results.json',candidates)
    finalize(candidates,monitor)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    def request_stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    with (OUT/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except InterruptedError as e:status('STOPPED_WITH_CHECKPOINT',reason=str(e))
        except Exception as e:
            save('error.json',dict(error=repr(e),traceback=traceback.format_exc()));status('ERROR',error=repr(e));raise
