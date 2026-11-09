"""Use more train dialogue; keep all development and seen regressions protected."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
import requests
from .common import digest,file_hash,input_id,normalize,read_jsonl,write_json,write_jsonl
from .trilingual_data import load_dialogues,turn_rows

ROOT=Path('data_v10/reward_20261025')
TITLES={'vi':['Hội An','Nha Trang','Đà Lạt','Bánh mì'],
    'en':['Melbourne','Tea','Bicycle','Museum'],'zh':['南京市','苏州市','茶','自行车']}


def split_protected(language):
    rows=[]
    for path in [Path('data_v6/pho_text_20261025/dev.jsonl'),Path('data_v6/pho_text_20261025/sealed.jsonl'),
        Path('data_v4/causal_20261025/dev.jsonl'),Path('data_v4/causal_20261025/sealed.jsonl'),Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')]:
        rows.extend(r for r in read_jsonl(path) if r['language']==language)
    if language=='vi':
        rows.extend(read_jsonl(Path('data_v2/frozen/dev.jsonl')))
        rows.extend(read_jsonl(Path('data_v2/frozen/holdout.jsonl')))
    return {input_id(r) for r in rows}


def add_unique(rows,protected):
    seen=set(protected);result=[];conflicts=0
    for row in rows:
        key=input_id(row)
        if key in seen:conflicts+=1;continue
        seen.add(key);result.append(row)
    return result,conflicts


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False);rng=random.Random(20261025)
    natural={s:list(read_jsonl(Path('data_v6/pho_text_20261025')/(s+'.jsonl'))) for s in ['train','dev']}
    source=Path('data_v3/sources_20261024');manifest={}
    for language in ['vi','en','zh']:
        protected=split_protected(language)
        real=[{**r,'pool':'natural','sample_weight':1.,'evaluation_slice':'natural'} for r in natural['train'] if r['language']==language]
        dev=[{**r,'evaluation_slice':'natural'} for r in natural['dev'] if r['language']==language]
        if language=='vi':
            replay=[{**r,'language':'vi','pool':'replay','sample_weight':.5,'label_scope':'LEGACY_PREFIX_PROXY'} for r in read_jsonl(Path('data_v2/frozen/train.jsonl'))]
            rng.shuffle(replay);replay=replay[:24000]
            olddev=list(read_jsonl(Path('data_v2/frozen/dev.jsonl')));rng.shuffle(olddev)
            dev.extend({**r,'language':'vi','evaluation_slice':'legacy_vi'} for r in olddev[:3600])
            raw_receipts=[{'path':'data_v2/frozen/train.jsonl','sha256':file_hash(Path('data_v2/frozen/train.jsonl'))}]
        else:
            old=Path('data_v9/monolingual_20261025')/language
            replay=[{**r,'pool':'replay','sample_weight':.5} for r in read_jsonl(old/'train.jsonl') if r['training_source']!='natural']
            # Keep exactly the prior dev subset for matched comparisons.
            dev.extend(r for r in read_jsonl(old/'dev.jsonl') if r['evaluation_slice']!='natural')
            dataset='multiwoz' if language=='en' else 'crosswoz';dialogs=load_dialogues(source,dataset,'train')
            rng.shuffle(dialogs);expanded=[]
            for did,turns in dialogs:
                for row in turn_rows(dataset,did,turns,'train'):
                    expanded.append({**row,'pool':'expanded_dialogue','sample_weight':.35,'label_scope':'OBSERVED_TURN_END_OR_CONTINUATION_PROXY_NOT_SEMANTIC_GOLD'})
                if len(expanded)>=24000:break
            replay.extend(expanded)
            raw_receipts=[{'path':str(p),'sha256':file_hash(p)} for p in sorted((source/'raw').glob('en_train_*.json' if language=='en' else 'zh_train.json.zip'))]
        protected.update(input_id(r) for r in dev)
        train,duplicates=add_unique(real+replay,protected)
        folder=ROOT/language;folder.mkdir()
        write_jsonl(folder/'train.jsonl',train);write_jsonl(folder/'dev.jsonl',dev)
        record={'language':language,'counts':{'train':dict(Counter(r['pool']+':'+r['label'] for r in train)),'dev':dict(Counter(r['evaluation_slice']+':'+r['label'] for r in dev))},
            'split_sha256':{s:file_hash(folder/(s+'.jsonl')) for s in ['train','dev']},'duplicates_or_protected_inputs_removed':duplicates,'raw_train_sources':raw_receipts,
            'source_sha256':file_hash(Path(__file__)),'held_targets_used':False,'protected_input_hashes_only':True,'fresh_test_available':False,
            'scope':'Mixed automatic observed-floor labels and low-weight structural continuation proxies; no human semantic gold'}
        write_json(folder/'manifest.json',record);manifest[language]=record
    write_json(ROOT/'manifest.json',manifest)
    print(json.dumps({lang:record['counts'] for lang,record in manifest.items()}),flush=True)


def crawl():
    output=ROOT/'crawl';output.mkdir(parents=True,exist_ok=False);http=requests.Session();http.headers['User-Agent']='TurnDetectorResearch/0.10 (bounded text-only research; provenance recorded)'
    sources=[];candidates=[];errors=[]
    old=list(read_jsonl(Path('data_v4/fresh_crawl_20261025/review_cases.jsonl')));excluded={r['source_id'] for r in old}
    for language,titles in TITLES.items():
        for title in titles:
            try:
                response=http.get(f'https://{language}.wikipedia.org/w/api.php',params={'action':'query','format':'json','formatversion':2,'prop':'extracts|info|revisions','explaintext':1,'exintro':1,'inprop':'url','rvprop':'ids','redirects':1,'titles':title},timeout=35)
                response.raise_for_status()
                for page in response.json().get('query',{}).get('pages',[]):
                    extract=page.get('extract','');sid=language+'wiki:'+str(page['pageid'])
                    if sid in excluded or not extract:continue
                    source={'language':language,'source_id':sid,'title':page['title'],'url':page.get('fullurl'),'page_revision':page.get('revisions',[{}])[0].get('revid'),
                        'text':extract,'sha256':digest(extract),'license':'Wikipedia CC BY-SA terms; attribution retained'};sources.append(source)
                    positions=list(re.finditer(r'[,，、;；:：.。!?！？]',extract))[:12]
                    for pos in positions:
                        text=extract[:pos.end()]
                        if len(normalize(text))<15:continue
                        candidates.append({'candidate_id':digest(sid+'\0'+str(pos.end())),'language':language,'source_id':sid,'source_url':source['url'],'source_title':page['title'],
                            'text':text,'context':'','punctuation':pos.group(),'label':None,'is_gold':False,'label_status':'PENDING_QWEN_TWO_VIEW_SILVER_ONLY',
                            'observed_suffix_sha256':digest(extract[pos.end():])})
            except (requests.RequestException,ValueError) as error:errors.append({'language':language,'title':title,'error':type(error).__name__+':'+str(error)[:150]})
            time.sleep(.3)
    write_jsonl(output/'sources.jsonl',sources);write_jsonl(output/'candidates.jsonl',candidates)
    write_json(output/'manifest.json',{'counts':dict(Counter(r['language'] for r in candidates)),'sources':len(sources),'errors':errors,'labels_created':False,
        'prior204_review_sources_excluded':True,'split':'TRAINING_CANDIDATE_POOL_ONLY; NOT_EVALUATION','source_sha256':file_hash(Path(__file__))})
    print(json.dumps({'crawl_counts':dict(Counter(r['language'] for r in candidates)),'errors':errors}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','crawl']);a=p.parse_args();prepare() if a.action=='prepare' else crawl()
