"""Separate EN/ZH datasets; keep the accepted Vietnamese deployment intact."""
from collections import Counter
from pathlib import Path
import argparse
import json

from .common import file_hash, input_id, read_jsonl, write_json, write_jsonl


def encode_tail(tokenizer, context, text, max_length=256):
    from .common import normalize
    if max_length < 3:raise ValueError('max_length must be >=3')
    current=tokenizer.encode(normalize(text),add_special_tokens=False)[-(max_length-3):] if max_length>3 else []
    budget=max_length-3-len(current)
    history=tokenizer.encode(normalize(context),add_special_tokens=False)[-budget:] if budget else []
    ids=[tokenizer.cls_token_id]+history+[tokenizer.sep_token_id]+current+[tokenizer.sep_token_id]
    return {'input_ids':ids,'attention_mask':[1]*len(ids)}


def group(row):
    return row['language']+':'+row.get('session_group',row['source_id'].split('__turn_')[0])


def partition_language(language, natural, replay, protected):
    if language not in {'en','zh'}:raise ValueError('VI keeps its established model; prepare EN/ZH separately')
    result={};seen=set(protected);dev_groups=set();removed=Counter()
    for split in ['dev','train']:
        rows=[]
        for source,values in [('natural',natural[split]),('replay',replay[split])]:
            for old in values:
                if old['language']!=language:continue
                row={**old,'training_source':source,'evaluation_slice':'natural' if source=='natural' else old['source']}
                key=language+':'+input_id(row)
                if key in seen or (split=='train' and group(row) in dev_groups):removed[split+':protected_or_duplicate']+=1;continue
                seen.add(key);rows.append(row)
                if split=='dev':dev_groups.add(group(row))
        result[split]=rows
    assert not {group(r) for r in result['train']}.intersection(group(r) for r in result['dev'])
    assert all(r['language']==language for rows in result.values() for r in rows)
    assert {r['label'] for r in result['train']}=={'EOT','NEOT'}
    return result,dict(removed)


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    paths={'natural':Path('data_v6/pho_text_20261025'),'replay':Path('data_v4/causal_20261025')}
    pools={name:{split:list(read_jsonl(root/(split+'.jsonl'))) for split in ['train','dev']} for name,root in paths.items()}
    protected={r['language']+':'+input_id(r) for root in paths.values() for r in read_jsonl(root/'sealed.jsonl')}
    for language in ['en','zh']:
        rows,removed=partition_language(language,pools['natural'],pools['replay'],protected)
        folder=output/language;folder.mkdir()
        for split,values in rows.items():write_jsonl(folder/(split+'.jsonl'),values)
        write_json(folder/'manifest.json',{'language':language,'scope':'MONOLINGUAL_TEXT_ONLY; natural automatic floor references plus causal proxy replay; dev only',
            'split_sha256':{s:file_hash(folder/(s+'.jsonl')) for s in rows},'counts':{s:dict(Counter(r['evaluation_slice']+':'+r['label'] for r in values)) for s,values in rows.items()},
            'removed':removed,'source_files':{str(root/(s+'.jsonl')):file_hash(root/(s+'.jsonl')) for root in paths.values() for s in ['train','dev','sealed']},
            'sealed_input_hashes_used_only_for_dedup':True,'seen_regressions_not_used_for_selection':True,'held_evaluated':False})
    vi=Path('models/turn_v2_distilled_qat4_20261024/onnx')
    write_json(output/'recipe.json',{'architecture':'THREE_INDEPENDENT_MODELS; no shared trainable weights or shared student vocabulary',
        'vi':{'action':'PRESERVE_ESTABLISHED_QAT_MODEL','path':str(vi),'weights_sha256':file_hash(vi/'model.onnx'),
            'historical_dev_accuracy':.935457,'historical_dev_eot_f1':.901934,'historical_scope':'Frozen derived-prefix dev; not natural-conversation test'},
        'en':{'data':str(output/'en'),'pipeline':'independent teacher -> own distillation -> own INT4 QAT'},
        'zh':{'data':str(output/'zh'),'pipeline':'independent teacher -> own distillation -> own INT4 QAT'},
        'new_quality_claim':False,'source_sha256':file_hash(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data_v9/monolingual_20261025'));prepare(p.parse_args().output)
