"""Bounded Qwen synthetic factory-dialogue pilot; separate from all training sets."""
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
    ('bang_tai','Băng tải đóng gói có lúc dừng ngắn; người vận hành báo bảo trì và hỏi tình trạng ca.'),
    ('cam_bien','Cảm biến trên dây chuyền báo lỗi không ổn định; hai người trao đổi về lúc lỗi xuất hiện.'),
    ('ban_giao_ca','Hai người bàn giao ca về số lần dừng máy, lô hàng đang dở và việc cần theo dõi.'),
    ('dong_co','Người vận hành báo động cơ có tiếng lạ; kỹ thuật viên hỏi thêm dấu hiệu và thời điểm.'),
    ('chat_luong','Nhân viên chất lượng báo sản phẩm bị lệch kích thước; tổ trưởng hỏi số lượng ảnh hưởng.'),
    ('vat_tu','Công nhân phát hiện thiếu vật tư đóng gói; kho và tổ trưởng phối hợp thời gian cấp thêm.'),
    ('robot','Nhân viên báo robot gắp hàng bỏ sót sản phẩm; kỹ thuật viên xác nhận mã báo lỗi.'),
    ('an_toan','Người vận hành thấy thiết bị che chắn có vấn đề; báo quản lý và chờ người có thẩm quyền kiểm tra.'),
    ('may_nen_khi','Hai người trao đổi về tiếng ồn của máy nén khí và phiếu yêu cầu bảo trì.'),
    ('lich_bao_tri','Người lập kế hoạch và kỹ thuật viên trao đổi lịch dừng máy để kiểm tra định kỳ.'),
    ('may_cnc','Nhân viên vận hành thông báo chi tiết gia công có bề mặt không đều và đề nghị kiểm tra.'),
    ('dong_goi','Nhân viên phát hiện nhãn trên thùng bị lệch; hai người trao đổi cách ghi nhận lô.'),
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
    result = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get('speaker'), str) or not isinstance(turn.get('sentences'), list):
            raise ValueError('invalid turn schema')
        speaker = turn['speaker'].strip()
        if not speaker or not 1 <= len(turn['sentences']) <= 3: raise ValueError('invalid speaker or sentence count')
        sentences = []
        for item in turn['sentences']:
            if not isinstance(item, str): raise ValueError('sentence must be string')
            original = item.strip()
            if not original or len(original) > 400 or '\n' in original: raise ValueError('invalid sentence')
            # Exactly one sentence per list entry: punctuation within is allowed,
            # but sentence terminal punctuation must appear only at the end.
            stripped = re.sub(r'[.!?…]+$', '', original).strip()
            if not stripped or re.search(r'[.!?…]', stripped): raise ValueError('multiple sentences in one entry')
            clean = normalize(original)
            if len(clean.split()) < 2: raise ValueError('sentence too short')
            if re.search(r'[<>\[\]{}]', clean): raise ValueError('model control characters in text')
            sentences.append({'display': original, 'normalized': clean})
        result.append({'speaker':speaker,'sentences':sentences})
    if len({x['speaker'] for x in result}) < 2: raise ValueError('two speakers required')
    return {'dialogue_id':scenario_id,'language':'vi','turns':result,'source':'QWEN_SYNTHETIC','is_human_gold':False}


def make_examples(dialogue):
    flat = []
    for turn_idx, turn in enumerate(dialogue['turns']):
        for sent_idx, sent in enumerate(turn['sentences']):
            flat.append((turn_idx, sent_idx, sent['normalized'].split()))
    examples = []
    for si,(turn_idx,sent_idx,words) in enumerate(flat):
        later = words + [word for _,_,more in flat[si+1:] for word in more]
        for cut in range(1,len(words)+1):
            for k in (1,2):
                future = later[cut:cut+k]
                if len(future) != k: continue
                label = 'EOT' if cut == len(words) else 'NEOT'
                identity = f"{dialogue['dialogue_id']}:{si}:{cut}:{k}"
                examples.append({'example_id':digest(identity),'dialogue_id':dialogue['dialogue_id'],
                    'language':'vi','context':' '.join(words[:cut]),'text':' '.join(future),
                    'label':label,'lookahead_words':k,'sentence_index':si,'turn_index':turn_idx,
                    'boundary_type':'synthetic_sentence_end' if label=='EOT' else 'synthetic_within_sentence',
                    'crosses_speaker':bool(label=='EOT' and si+1<len(flat) and flat[si+1][0]!=turn_idx),
                    'label_source':'QWEN_SYNTHETIC_SENTENCE_STRUCTURE','punctuation_in_model_input':False,
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
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,revision=revision,token=False)
    model=AutoModelForCausalLM.from_pretrained(checkpoint,revision=revision,torch_dtype=torch.bfloat16,device_map='auto',token=False).eval()
    system=('Bạn viết hội thoại HƯ CẤU giữa người làm việc trong nhà máy Việt Nam cho bộ dữ liệu nghiên cứu. '
      'Chỉ trả JSON object có trường turns: một danh sách 4-10 lượt. Mỗi lượt là object có speaker và sentences; '
      'sentences là danh sách 1-3 chuỗi, mỗi chuỗi đúng MỘT câu nói hoàn chỉnh. '
      'Đối thoại tự nhiên, ngắn gọn, có câu hỏi và câu trả lời, từ ngữ vận hành phù hợp, có thể ngập ngừng nhẹ. '
      'Không chép nguyên văn ví dụ, không nhắc nguồn. Không dùng tên người thật, email, số điện thoại, '
      'không đưa hướng dẫn thao tác nguy hiểm. Không dùng markdown.')
    dialogues=[];examples=[];errors=[]
    for scenario_id,scenario in SCENARIOS[:count]:
        messages=[{'role':'system','content':system},{'role':'user','content':'Viết một hội thoại theo tình huống: '+scenario}]
        try:
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tokenizer(prompt,return_tensors='pt').to(model.device)
            with torch.inference_mode():
                tokens=model.generate(**inputs,max_new_tokens=1300,do_sample=True,temperature=.75,top_p=.9,pad_token_id=tokenizer.eos_token_id)
            raw=tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
            raw=re.sub(r'^```(?:json)?\s*|\s*```$', '',raw,flags=re.I)
            d=validate_dialogue(json.loads(raw),scenario_id);d['scenario']=scenario
            rows=make_examples(d)
            assert {r['label'] for r in rows}=={'EOT','NEOT'}
            dialogues.append(d);examples.extend(rows)
        except Exception as exc:
            errors.append({'scenario_id':scenario_id,'error':repr(exc),'raw':locals().get('raw','')[:5000]})
        write_json(output/'status.json',{'stage':'GENERATING','done':len(dialogues)+len(errors),'accepted_dialogues':len(dialogues),'rejected':len(errors),'examples':len(examples)})
    for name,rows in [('dialogues.jsonl',dialogues),('examples.jsonl',examples),('rejected.jsonl',errors)]:
        with (output/name).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts=Counter((x['label'],x['lookahead_words']) for x in examples)
    write_json(output/'manifest.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'model':checkpoint,'revision':revision,'gpu':props.name,
       'source_type':'QWEN_SYNTHETIC_NOT_REAL_CONVERSATION_OR_HUMAN_GOLD','examples_used_for_training':False,
       'sources_for_scenario_inspiration':SOURCES,'requested_dialogues':count,'accepted_dialogues':len(dialogues),'rejected_dialogues':len(errors),
       'counts':{f'{a}_k{b}':n for (a,b),n in counts.items()},'test_or_dev_selected':False,
       'caveat':'Sentence ends use generated sentence structure; cross-speaker lookahead assumes a continuous word stream. No human/audio gold.'})
    write_json(output/'status.json',{'stage':'COMPLETE','accepted_dialogues':len(dialogues),'rejected':len(errors),'examples':len(examples)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=8,choices=range(1,len(SCENARIOS)+1));args=p.parse_args();run(args.output,args.count)

if __name__=='__main__':main()
