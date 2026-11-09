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
    ('chuyen_gia_chat_luong','Chuyên gia chất lượng đến nhà máy và chủ quản sản xuất trao đổi tại dây chuyền về lỗi bề mặt, phương pháp lấy mẫu, số mẫu lỗi và kế hoạch xác nhận lại.'),
    ('ky_su_nang_suat','Kỹ sư tư vấn và giám sát chuyền so sánh thời gian chu kỳ giữa hai công đoạn, hỏi vị trí ùn hàng và bàn cách đo thêm.'),
    ('khach_hang_kiem_tra','Kỹ sư khách hàng thăm nhà máy hỏi trưởng bộ phận chất lượng về hồ sơ kiểm tra đầu vào, truy xuất lô và số lượng cách ly.'),
    ('chuyen_gia_thiet_bi','Chuyên gia thiết bị làm việc tại xưởng với quản lý bảo trì về lịch dừng máy, dấu hiệu rung và việc xác minh bằng phiếu kiểm tra.'),
    ('chu_quan_giao_hang','Chủ quản nhà máy trao đổi với giám sát dự án từ nước ngoài về tiến độ lô, hàng còn thiếu, công đoạn nghẽn và thời điểm cập nhật lại.'),
    ('ky_su_quy_trinh','Kỹ sư quy trình đến khảo sát và tổ trưởng thảo luận về thông số trên bảng hiển thị, phạm vi được phép và cách đối chiếu tài liệu.'),
    ('chuyen_gia_dao_tao','Chuyên gia chuyển giao hỏi giám sát nhà máy về số người được đào tạo, thay ca và những điểm nhân viên còn chưa rõ.'),
    ('kiem_tra_hien_truong','Giám sát khách hàng và chủ quản xưởng đối chiếu thực tế nhà xưởng với biên bản kiểm tra, chỗ cần bổ sung ảnh và thời hạn phản hồi.'),
    ('kiem_soat_lo','Chuyên gia tư vấn và kỹ sư chất lượng thảo luận về việc phân loại hàng lỗi, mã lô và cách xác nhận số lượng đã kiểm.'),
    ('nha_cung_cap','Kỹ sư nhà cung cấp làm việc với giám sát nhà máy về linh kiện không khớp bản vẽ và tài liệu cần đối chiếu.'),
    ('du_an_moi','Chuyên gia dự án và trưởng bộ phận sản xuất trao đổi về chạy thử, trách nhiệm từng bên và kết quả cần ghi nhận.'),
    ('ke_hoach_bao_tri','Quản lý bảo trì và kỹ sư cố vấn trao đổi lịch kiểm tra định kỳ, thiết bị ưu tiên và thời gian có thể dừng máy.'),
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
    system=('Bạn viết hội thoại HƯ CẤU để nghiên cứu phiên dịch tại nhà máy. '
      'Bối cảnh là chuyên gia tư vấn, kỹ sư hoặc giám sát đến nhà máy làm việc trực tiếp với chủ quản, '
      'giám sát chuyền hoặc kỹ sư nhà máy; hai người cần trao đổi công việc qua phiên dịch. '
      'Bắt đầu giữa cuộc trao đổi, đi thẳng vào vấn đề, TUYỆT ĐỐI không có lời chào hỏi hay giới thiệu ở bất kỳ lượt nào. '
      'Giọng nói đời thường nhưng chuyên môn phù hợp: hỏi lại, giải thích, xác nhận số liệu, phản hồi và nêu việc tiếp theo. '
      'Dùng chi tiết cụ thể phù hợp tình huống, giữ số liệu nhất quán, không lên lớp, không lặp câu mẫu hay vai. '
      'Chỉ trả JSON object có trường turns gồm 4-6 lượt luân phiên của đúng hai vai cụ thể. '
      'Mỗi lượt là MỘT ĐOẠN người đó nói, chứa 3-5 câu liên quan, có thông tin cụ thể, không phải vài từ ngắn. '
      'Mỗi lượt là object có speaker và sentences; sentences là danh sách 3-5 chuỗi, mỗi chuỗi MỘT câu hoàn chỉnh kết thúc bằng dấu chấm. '
      'Một người có thể nói nhiều câu trước khi người khác nói. Dùng dấu phẩy tự nhiên trong câu. '
      'Không chép nguyên văn ví dụ, không nhắc nguồn. Không dùng tên người thật, email, số điện thoại; '
      'không đưa hướng dẫn vận hành nguy hiểm hoặc thông số bịa đặt được coi là quy trình. Không dùng markdown.')
    dialogues=[];examples=[];errors=[];batch_sizes=[]
    languages=[('vi','tiếng Việt'),('en','tiếng Anh'),('zh','tiếng Trung phổ thông')]
    tasks=[(f'{language}_{sid}_v{variant}',scenario,description) for language,description in languages for sid,scenario in SCENARIOS[:count] for variant in (1,2,3,4)]
    def generate_batch(batch):
        prompts=[]
        for scenario_id,scenario,description in batch:
            messages=[{'role':'system','content':system},{'role':'user','content':'Tất cả các lượt trong hội thoại này phải dùng '+description+' và không dùng bất kỳ ngôn ngữ nào khác. Tình huống: '+scenario}]
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
