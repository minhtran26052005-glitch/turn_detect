"""Freeze selected shared teacher and prepare TRAIN-only tiny-student inputs."""
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

from .factory_stream_data_v1 import Sampler, sha
from .multilingual_three_teacher_distill_v1 import DATA, rows_after, validate_row

ROOT=Path('/home/edabk/hoangbpm/turn_detect')
SOURCE=ROOT/'outputs/multilingual_stream_v2_20261102'
OUT=ROOT/'outputs/multilingual_tiny_compression_20261102_v1'


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def status(stage,**extra):save('preparation_status.json',{'stage':stage,'pid':os.getpid(),'at_unix':time.time(),**extra})


def run():
    assert shutil.disk_usage(OUT).free>15*2**30
    teacher=OUT/'teacher';teacher.mkdir(exist_ok=True)
    status('PINNING_DEV_SELECTED_TEACHER')
    if not (OUT/'teacher_receipt.json').exists():
        before=(SOURCE/'best_report.json').read_bytes();path=SOURCE/'best/model.safetensors'
        assert path.stat().st_mtime_ns<=(SOURCE/'best_report.json').stat().st_mtime_ns,'Best update in progress; retry safely before pinning'
        os.link(path,teacher/'model.safetensors')
        after=(SOURCE/'best_report.json').read_bytes();assert before==after and path.stat().st_ino==(teacher/'model.safetensors').stat().st_ino,'Teacher changed while pinning'
        report=json.loads(before)
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:shutil.copyfile(SOURCE/'best'/name,teacher/name)
        save('teacher_selected_report.json',report)
        save('teacher_receipt.json',{'branch_step':report['branch_step'],'step':report['step'],'weights_sha256':sha(teacher/'model.safetensors'),
             'weight_bytes':(teacher/'model.safetensors').stat().st_size,'selection':'Existing campaign best, DEV-selected before compression; no tests used',
             'source_dataset_manifest_sha256':sha(SOURCE/'dataset/manifest.json'),'source_main_training_continues':True,'optimizer_not_required_for_frozen_inference_teacher':True})
    deps=OUT/'runtime_deps';deps.mkdir(exist_ok=True)
    status('INSTALLING_ISOLATED_EXPORT_DEPENDENCIES')
    if not (deps/'onnx').exists():
        command=[sys.executable,'-m','pip','install','--no-deps','--no-cache-dir','--target',str(deps),
                 'onnx==1.19.1','onnxruntime==1.23.2','flatbuffers==25.9.23','coloredlogs==15.0.1','humanfriendly==10.0']
        with (OUT/'dependency_install.log').open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    status('PREPARING_TRAIN_ONLY_DISTILL_ROWS')
    train=OUT/'train.jsonl'
    if not train.exists():
        sampler=Sampler(SOURCE/'dataset',seed=202611029);db=sqlite3.connect(DATA.as_uri()+'?mode=ro',uri=True);cursor='';counts=Counter();labels=Counter()
        tmp=train.with_suffix('.partial')
        with tmp.open('w',encoding='utf-8') as f:
            for block in range(100):
                raw=rows_after(db,'zh','train',cursor,1000);zh=[validate_row(r) for r in raw];assert len(zh)==1000;cursor=raw[-1][0]
                pools={'vi':[sampler.row('vi') for _ in range(1000)],'en':[sampler.row('en') for _ in range(1000)],'zh':zh}
                for i in range(1000):
                    for lang in ['vi','en','zh']:
                        row=pools[lang][i];record={'language':lang,'context':row['context'],'text':row['text'],'label':row['label'],'split':'train',
                            'label_source':'SYNTHETIC_PUNCTUATION_SILVER' if lang!='zh' else 'LEGACY_PUNCTUATION_SILVER','source_index':counts[lang]}
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');counts[lang]+=1;labels[lang+':'+row['label']]+=1
                if block%5==0:f.flush();status('PREPARING_TRAIN_ONLY_DISTILL_ROWS',counts=dict(counts))
            f.flush();os.fsync(f.fileno())
        tmp.replace(train);db.close()
        save('train_receipt.json',{'rows':300000,'by_language':dict(counts),'labels':dict(labels),'sha256':sha(train),
              'source':'Frozen source722 accepted TRAIN VI/EN sampler plus accepted legacy ZH TRAIN; no new spokenv9 or DEV/test rows.',
              'subset_not_full_corpus':True,'new_spoken_generator_data_used':False,'sampler_state_after':sampler.state_dict(),'zh_cursor_after':cursor})
    status('BUILDING_TRAIN_ONLY_BYTE_BPE')
    vocabulary=OUT/'vocabulary'
    if not (vocabulary/'tokenizer.json').exists():
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
        from transformers import PreTrainedTokenizerFast
        vocabulary.mkdir(exist_ok=True)
        backend=Tokenizer(models.BPE(unk_token='[UNK]'));backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);backend.decoder=decoders.ByteLevel()
        trainer=trainers.BpeTrainer(vocab_size=8192,min_frequency=2,special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        def texts():
            with train.open(encoding='utf-8') as f:
                for line in f:
                    row=json.loads(line);yield row['context'];yield row['text']
        backend.train_from_iterator(texts(),trainer=trainer)
        tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',unk_token='[UNK]',cls_token='[CLS]',sep_token='[SEP]',model_max_length=256)
        tok.save_pretrained(vocabulary)
        save('vocabulary/config.json',{'model_type':'bert','vocab_size':len(tok),'pad_token_id':0})
        save('vocabulary_report.json',{'size':len(tok),'train_only':True,'train_sha256':sha(train),'algorithm':'byte-level BPE with256byte fallback alphabet',
             'teacher_tokenization_is_different':True,'no_teacher_embedding_pca_mapping_claim':True,'tokenizer_sha256':sha(vocabulary/'tokenizer.json')})
    status('COMPLETE',train_rows=300000,teacher=json.loads((OUT/'teacher_receipt.json').read_text()),train_sha256=sha(train))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'PREPARATION_LOCK').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run()
        except Exception as e:save('preparation_error.json',{'error':repr(e),'traceback':traceback.format_exc()});status('ERROR',error=repr(e));raise
