"""Verify finished artifacts and numerical agreement with the existing predictor."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
JOB=ROOT/'outputs/malbert_eval_zip_20261104_v1'
MAIN=ROOT/'outputs/multilingual_malbert_vi_en_checked_20261103_v3'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def main():
    report=json.loads((JOB/'report.json').read_text());completion=json.loads((JOB/'completion_verification.json').read_text())
    launch=json.loads((JOB/'launcher.json').read_text());source_audit=json.loads((JOB/'input_source_order_audit.json').read_text())
    assert report['complete'] and completion['complete'] and source_audit['complete']
    assert sha(JOB/'report.json')==completion['report_sha256']
    assert sha(ROOT/'turn_v2/malbert_eval_zip_v1.py')==report['evaluator_source_sha256']==launch['source_sha256']
    for name,key in [('student_resume_pin.pt','full_checkpoint_sha256'),('weights.pt','weights_sha256')]:
        assert sha(JOB/'frozen'/name)==report['checkpoint'][key]
    assert sha(JOB/'inputs/eval.zip')==report['input_manifest']['zip_sha256']
    assert sha(JOB/'prepared_streams.json.gz')==source_audit['prepared_streams_sha256']==report['input_manifest']['prepared_streams_sha256']
    assert sha(JOB/'predictions.jsonl.gz')==report['predictions_sha256']
    assert sha(MAIN/'train_vi_en.jsonl')==report['main_dataset_sha256']
    streams=json.load(gzip.open(JOB/'prepared_streams.json.gz','rt',encoding='utf-8'))
    selected={};counts=Counter();overlap_streams=set();minimum=1.;maximum=0.
    with gzip.open(JOB/'predictions.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);p=row['p_eot'];assert math.isfinite(p) and 0<=p<=1
            assert row['predicted_eot']==(p>=report['checkpoint']['threshold'])
            counts[row['lookahead_words']]+=1;minimum=min(minimum,p);maximum=max(maximum,p)
            if row['known_train_source_overlap']:overlap_streams.add(row['stream'])
            stream=streams[row['stream']];b=row['boundary']
            segment=next(s for s in stream['segments'] if s['start']<=b-1<s['end'])
            bucket='short' if row['input_length']<=32 else 'medium' if row['input_length']<=128 else 'long'
            key=(stream['domain'],segment['language'],row['lookahead_words'],bucket)
            if key not in selected:selected[key]=row
    assert counts=={1:586048,2:555469} and sum(counts.values())==report['inferences']==completion['all_inferences_audited']
    assert len(overlap_streams)==54
    import torch
    import transformers.utils as u
    import transformers.utils.import_utils as im
    u.is_torchvision_available=im.is_torchvision_available=lambda:False
    u.is_librosa_available=im.is_librosa_available=lambda:False
    from transformers import AlbertConfig,AutoTokenizer
    from .malbert_turn_student_v1 import MalbertTurnStudent
    from .multilingual_tiny_compress_v3 import tensor_predictor
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    tok=AutoTokenizer.from_pretrained(MAIN/'tokenizer',local_files_only=True,token=False)
    assert tok.pad_token_id==0 and tok.sep_token_id==3
    model=MalbertTurnStudent(AlbertConfig.from_pretrained(MAIN/'pretrained',local_files_only=True,token=False),sep_token_id=3)
    model.load_state_dict(torch.load(JOB/'frozen/weights.pt',map_location='cpu',weights_only=True),strict=True)
    assert model.parameter_count()==11941122
    model.cuda().eval();predict=tensor_predictor(model,tok,'cuda');rows=list(selected.values());delta=0.
    for start in range(0,len(rows),16):
        group=rows[start:start+16]
        inputs=[{'context':' '.join(streams[r['stream']]['words'][r['context_start']:r['boundary']]),
                 'text':' '.join(streams[r['stream']]['words'][r['boundary']:r['future_end']])} for r in group]
        observed=predict(inputs)
        for a,b in zip(group,observed):
            delta=max(delta,abs(a['p_eot']-b))
            assert (b>=report['checkpoint']['threshold'])==a['predicted_eot']
    assert delta<1e-4,delta
    main_launch=json.loads((MAIN/'launcher.json').read_text());proc=Path('/proc')/str(main_launch['pid'])
    stat=(proc/'stat').read_text();ticks=stat[stat.rfind(')')+2:].split()[19]
    cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert ticks==str(main_launch['start_ticks']) and 'turn_v2.multilingual_malbert_vi_en_checked_train_v3' in cmdline
    mount=next(s for s in Path('/proc/mounts').read_text().splitlines() if s.split()[1]=='/mnt/hdd500')
    assert mount.split()[0]=='/dev/sda1' and 'rw' in mount.split()[3].split(',')
    free={'ssd':shutil.disk_usage(ROOT).free/2**30,'hdd':shutil.disk_usage('/mnt/hdd500').free/2**30}
    assert min(free.values())>=13
    proof={'complete':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'report_sha256':completion['report_sha256'],
        'all_probabilities_finite_and_in_range':True,'probability_min_max':[minimum,maximum],
        'inferences_by_lookahead':dict(counts),'known_overlap_streams_actually_scored':len(overlap_streams),
        'canonical_training_dev_predictor_parity_cases':len(rows),'canonical_predictor_max_probability_delta':delta,
        'canonical_predictor_all_decisions_identical':True,'verification_only_no_new_performance_metrics_or_selection':True,
        'frozen_zip_streams_weights_fullpin_trace_source_sha_verified':True,
        'main_process':{'pid':main_launch['pid'],'start_ticks':ticks,'cmdline':cmdline},
        'main_status':json.loads((MAIN/'status.json').read_text()),'disk_free_gib':free,'hdd_mount':mount,
        'main_training_dataset_unchanged':True,'backed_up_to_drive':False}
    target=JOB/'independent_completion_verification.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(proof))


if __name__=='__main__':main()
