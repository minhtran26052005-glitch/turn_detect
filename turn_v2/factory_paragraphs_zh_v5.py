"""Batched Qwen factory-dialogue pilot; separate from all training sets."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata

SOURCES = [
    {'url':'https://huggingface.co/datasets/UKPLab/TexPrax', 'kind':'german_worker_dialogues', 'license':'CC BY-NC 4.0; no source text copied'},
    {'url':'https://zenodo.org/records/8268928', 'kind':'industrial_scripts_and_professionally_generated_english_audio', 'license':'See source record; no source text copied'},
    {'url':'https://englishladder.com/efsp-manufacturing.html', 'kind':'fictional_language_training_dialogues', 'license':'No source text copied'},
]
SCENARIOS = [
    ('quality','质量专家到工厂，与生产主管在产线讨论产品表面缺陷、抽样方法、缺陷数量和复核计划。'),
    ('cycle_time','顾问工程师和生产线主管比较两个工序的周期时间，讨论堵料位置和进一步测量。'),
    ('customer_audit','客户工程师参观工厂，询问质量主管来料检查记录、批次追溯和隔离数量。'),
    ('maintenance','设备专家在车间与维修主管讨论停机计划、振动现象以及检查记录。'),
    ('delivery','工厂主管与外方项目经理交流生产进度、缺料、瓶颈工序和下一次更新时间。'),
    ('process','工艺工程师与车间班长讨论显示屏上的参数、允许范围和文件核对。'),
    ('training','技术转移专家向工厂主管询问培训人数、换班和员工尚不清楚的问题。'),
    ('site_inspection','客户代表和车间主管现场核对检查记录、需要补充的照片和回复时间。'),
]


def normalize(value):
    value = unicodedata.normalize('NFKC', value).lower()
    value = ''.join(' ' if unicodedata.category(ch).startswith('P') else ch for ch in value)
    return re.sub(r'\s+', ' ', value).strip()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def validate_dialogue(obj, scenario_id):
    if not isinstance(obj, dict) or not isinstance(obj.get('turns'), list):
        raise ValueError('expected object with turns list')
    turns = obj['turns']
    if not 4 <= len(turns) <= 12: raise ValueError('expected 4-12 turns')
    language=scenario_id.split('_',1)[0]
    if language not in ('vi','en','zh'): raise ValueError('missing language identifier')
    result = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get('speaker'), str) or not isinstance(turn.get('sentences'), list):
            raise ValueError('invalid turn schema')
        speaker = turn['speaker'].strip()
        if not speaker or not 1 <= len(turn['sentences']) <= 8: raise ValueError('invalid speaker or sentence count')
        sentences = []
        for item in turn['sentences']:
            if not isinstance(item, str): raise ValueError('sentence must be string')
            original = item.strip()
            if not original or len(original) > 400 or '\n' in original: raise ValueError('invalid sentence')
            # Preserve model-authored punctuation and split obvious merged sentences.
            # Do not trust the generated split as human or audio ground truth.
            pieces=re.split(r'(?<=[。！？])\s*|(?<=[.!?])\s+(?=\S)',original)
            for piece in pieces:
                piece=piece.strip()
                if not piece:continue
                clean=normalize(piece)
                if len(clean)<2: raise ValueError('sentence too short')
                if re.search(r'[<>\[\]{}]', clean): raise ValueError('model control characters in text')
                sentences.append({'display':piece,'normalized':clean})
        if len(sentences)<2:raise ValueError('speaker paragraph must contain at least two sentences')
        result.append({'speaker':speaker,'sentences':sentences})
    if len({x['speaker'] for x in result}) != 2: raise ValueError('exactly two speakers required')
    joined=' '.join(s['normalized'] for t in result for s in t['sentences'])
    has_han=bool(re.search(r'[\u3400-\u9fff]',joined))
    if language=='zh' and not has_han:raise ValueError('Chinese dialogue lacks Chinese characters')
    if language in ('vi','en') and has_han:raise ValueError('unexpected Chinese characters')
    greeting=re.compile(r'^(?:xin chào|chào |hello\b|hi\b|good morning|good afternoon|你好|您好|大家好)',re.I)
    greetings=sum(bool(greeting.search(s['normalized'])) for t in result for s in t['sentences'])
    if greetings:raise ValueError('greeting present')
    return {'dialogue_id':scenario_id,'language':language,'turns':result,'source':'QWEN_SYNTHETIC','is_human_gold':False}


def make_examples(dialogue):
    flat = [];language=dialogue['language']
    if language=='zh':
        import jieba
        def words(text):return [x for x in (''.join(c for c in token if c.isalnum()) for token in jieba.lcut(text,HMM=False)) if x]
    else:
        def words(text):return re.findall(r'[^\W_]+',text,re.UNICODE)
    for turn_idx, turn in enumerate(dialogue['turns']):
        for sent_idx, sent in enumerate(turn['sentences']):
            flat.append((turn_idx, sent_idx, words(sent['normalized']),sent['display']))
    examples = []
    for si,(turn_idx,sent_idx,current,display) in enumerate(flat):
        next_words=[word for tid,_,more,_ in flat[si+1:] if tid==turn_idx for word in more]
        candidates=[]
        # Full stop is the target; punctuation is absent from model input.
        if display.rstrip().endswith(('.', '。')):candidates.append((current,next_words,'EOT','full_stop'))
        for match in re.finditer(r'[,，;；]',display):
            left=words(normalize(display[:match.start()]));right=words(normalize(display[match.end():]))
            if left and right:candidates.append((left,right,'NEOT','internal_comma_or_semicolon'))
        for idx,(left,right,label,kind) in enumerate(candidates):
            for k in (1,2):
                future=right[:k]
                if not left or len(future)!=k:continue
                identity=f"{dialogue['dialogue_id']}:{si}:{idx}:{k}"
                examples.append({'example_id':digest(identity),'dialogue_id':dialogue['dialogue_id'],
                    'language':language,'context':' '.join(left),'text':' '.join(future),'label':label,
                    'lookahead_words':k,'sentence_index':si,'turn_index':turn_idx,'boundary_type':kind,
                    'crosses_speaker':False,'label_source':'QWEN_PUNCTUATION_DERIVED_WEAK_LABEL',
                    'punctuation_in_model_input':False,'requires_human_review':True,
                    'is_human_gold':False,'split':'pilot_review_not_train'})
    return examples


def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def run(output, count):
    if sys.platform == 'win32': raise RuntimeError('Pilot Qwen inference is restricted to Colab Linux')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert Path('/content').is_dir() and torch.cuda.is_available(), 'Requires a Colab GPU runtime'
    props=torch.cuda.get_device_properties(0)
    assert props.total_memory >= 22*1024**3, '8B BF16 pilot needs a GPU with at least 22 GiB VRAM'
    assert shutil.disk_usage('/content').free >= 25*1024**3, 'Keep 25 GiB available on Colab'
    output.mkdir(parents=True,exist_ok=False)
    checkpoint='Qwen/Qwen3-8B'; revision='b968826d9c46dd6066d109eabc6255188de91218'
    write_json(output/'status.json',{'stage':'LOADING_QWEN','model':checkpoint,'revision':revision})
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=revision,token=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(checkpoint,revision=revision,dtype=torch.bfloat16,device_map='auto',token=False).eval()
    system=('你要为工厂口译研究编写虚构的普通话对话。只用简体中文，不要使用越南语或英文。'
      '场景是顾问、工程师或客户代表与工厂主管、生产监督或工厂工程师在现场讨论工作。'
      '从话题中间直接开始，所有轮次都不要问候或自我介绍。语言自然，讨论明确的数据、问题、确认和下一步。'
      '只返回JSON对象，字段turns是4至6轮两位角色交替讲话。每个turn包含speaker和sentences，'
      'sentences是3至5个字符串，每个字符串是一句完整的话，用中文句号结束。一人一轮应是由多句话组成的段落。'
      '不要编造真实姓名和联系方式，不要输出危险的操作步骤，不要用Markdown。')
    dialogues=[];examples=[];errors=[];batch_sizes=[]
    languages=[('zh','普通话')]
    tasks=[(f'{language}_{sid}_v{variant}',scenario,description) for language,description in languages for sid,scenario in SCENARIOS[:count] for variant in (1,2,3,4)]
    def generate_batch(batch):
        prompts=[]
        for scenario_id,scenario,description in batch:
            messages=[{'role':'system','content':system},{'role':'user','content':'请只使用'+description+'写一段工厂现场对话。场景：'+scenario}]
            prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
        oom=False
        try:
            inputs=tokenizer(prompts,padding=True,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                tokens=model.generate(**inputs,max_new_tokens=1024,do_sample=True,temperature=.85,top_p=.92,pad_token_id=tokenizer.eos_token_id)
            raw_outputs=tokenizer.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
            batch_sizes.append(len(batch))
            del inputs,tokens
        except torch.cuda.OutOfMemoryError:
            oom=True
        if oom:
            if 'inputs' in locals(): del inputs
            torch.cuda.empty_cache()
            if len(batch)==1: raise RuntimeError('GPU out of memory even at batch size 1')
            middle=len(batch)//2
            generate_batch(batch[:middle]);generate_batch(batch[middle:]);return
        for (scenario_id,scenario,_description),raw in zip(batch,raw_outputs):
            raw=re.sub(r'^```(?:json)?\s*|\s*```$', '',raw.strip(),flags=re.I)
            try:
                d=validate_dialogue(json.loads(raw),scenario_id);d['scenario']=scenario
                rows=make_examples(d)
                assert {r['label'] for r in rows}=={'EOT','NEOT'}
                dialogues.append(d);examples.extend(rows)
            except Exception as exc:errors.append({'scenario_id':scenario_id,'error':repr(exc),'raw':raw[:5000]})
        write_json(output/'status.json',{'stage':'GENERATING','done':len(dialogues)+len(errors),'accepted_dialogues':len(dialogues),'rejected':len(errors),'examples':len(examples),'batch_sizes':batch_sizes,'peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,2)})
    for offset in range(0,len(tasks),32):generate_batch(tasks[offset:offset+32])
    for name,rows in [('dialogues.jsonl',dialogues),('examples.jsonl',examples),('rejected.jsonl',errors)]:
        with (output/name).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for language,_ in languages:
        for prefix,rows in [('dialogues',dialogues),('examples',examples)]:
            with (output/f'{prefix}_{language}.jsonl').open('w',encoding='utf-8') as f:
                for row in rows:
                    if row['language']==language:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts=Counter((x['language'],x['label'],x['lookahead_words']) for x in examples)
    write_json(output/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'model':checkpoint,'revision':revision,'gpu':props.name,
       'source_type':'QWEN_SYNTHETIC_NOT_REAL_CONVERSATION_OR_HUMAN_GOLD','examples_used_for_training':False,
       'sources_for_scenario_inspiration':SOURCES,'requested_dialogues':len(tasks),'accepted_dialogues':len(dialogues),'rejected_dialogues':len(errors),
       'batch_sizes':batch_sizes,'gpu_total_gib':round(props.total_memory/2**30,2),'gpu_peak_reserved_gib':round(torch.cuda.max_memory_reserved()/2**30,2),
       'counts':{f'{lang}_{label}_k{k}':n for (lang,label,k),n in counts.items()},'test_or_dev_selected':False,
       'caveat':'Monolingual same-speaker lookahead; EOT at generated full stops, NEOT at internal commas or semicolons. This is weak punctuation-derived supervision, not verified semantic completeness or human/audio gold.'})
    write_json(output/'status.json',{'stage':'COMPLETE','accepted_dialogues':len(dialogues),'rejected':len(errors),'examples':len(examples)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=8,choices=range(1,len(SCENARIOS)+1));args=p.parse_args();run(args.output,args.count)

if __name__=='__main__':main()
