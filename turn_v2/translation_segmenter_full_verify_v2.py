"""Independent all-row encoded-input/provenance gate before any optimizer step."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np

from . import translation_segmenter_full_data_v5 as data
from .translation_segmenter_v15c import atomic_json,file_sha


def check_encoded(row,raw,checksum,cls,sep,pad,vocab_size):
    if not raw or len(raw)%4 or len(raw)>256*4:return 'bad_token_array_length'
    if hashlib.sha256(raw).hexdigest()!=checksum:return 'encoded_hash_mismatch'
    ids=np.frombuffer(raw,dtype='<u4')
    if len(ids)<5 or ids[0]!=cls or ids[-1]!=sep:return 'bad_special_token_shape'
    separators=np.flatnonzero(ids==sep)
    if len(separators)!=2 or separators[0]<=1 or separators[0]>=len(ids)-2:return 'empty_left_or_future_slot'
    if np.any(ids==pad) or int(ids.max())>=vocab_size:return 'invalid_or_padding_token_inside_input'
    if not row.get('context') or len(row.get('text','').split())!=row['lookahead_words']:return 'bad_raw_context_or_future'
    return None


def verify(manifest):
    from transformers import AutoTokenizer
    folder=data.JOB/'datasets'/manifest['generation'];receipt=folder/'verification_receipt.json'
    code_sha=file_sha(Path(__file__))
    if receipt.exists():
        old=json.loads(receipt.read_text())
        assert old['complete'] and old['database_sha256']==manifest['database_sha256'] and old['verifier_source_sha256']==code_sha
        return old
    assert manifest['complete'] and manifest['all_train_dev_sources_examined']
    assert file_sha(folder/'dataset.sqlite')==manifest['database_sha256']
    tokenizer_hashes={lang:file_sha(data.ROOT/'teachers'/lang/'best/tokenizer.json') for lang in data.LANGUAGES}
    assert len(set(tokenizer_hashes.values()))==1,'Three model tokenizers differ; cannot share encoded inputs'
    tok=AutoTokenizer.from_pretrained(data.ROOT/'teachers/vi/best',local_files_only=True)
    db=sqlite3.connect('file:'+str(folder/'dataset.sqlite')+'?mode=ro',uri=True)
    sources={lang+':'+pid:(checksum,split,rejected,key) for lang,pid,checksum,split,rejected,key in db.execute('SELECT language,page_id,text_sha256,split,rejected,source_key FROM processed_sources')}
    protected={(lang,h) for lang,h in db.execute('SELECT language,input_hash FROM protected')}
    counts=Counter();bad=Counter();examples=[];n=0;last=0
    db.execute('PRAGMA cache_size=-262144')
    indexes={r[1]:r for r in db.execute('PRAGMA index_list(records)')}
    assert indexes['accepted_encoded_input_unique'][2]==1 and indexes['accepted_encoded_input_unique'][4]==1
    assert [r[2] for r in db.execute('PRAGMA index_info(accepted_encoded_input_unique)')]==['language','input_hash']
    previous=None;unique_rows=0
    for lang,h in db.execute("SELECT language,input_hash FROM records INDEXED BY accepted_encoded_input_unique WHERE disposition='accepted' ORDER BY language,input_hash"):
        assert (lang,h)!=previous,'Duplicate accepted encoded input'
        previous=(lang,h);unique_rows+=1
    for eid,lang,split,label,k,sid,h,raw,record in db.execute("SELECT example_id,language,split,label,k,source_id,input_hash,tokens,record_json FROM records NOT INDEXED WHERE disposition='accepted'"):
        row=json.loads(record);reason=check_encoded(row,raw,h,tok.cls_token_id,tok.sep_token_id,tok.pad_token_id,len(tok))
        if (lang,h) in protected:reason='protected_test_input_in_train_or_dev'
        source=sources.get(sid)
        if source is None or source[1]!=split or source[2] is not None:reason='invalid_or_quarantined_source'
        elif source[0]!=row.get('source_text_sha256') or source[3]!=row.get('source_key'):reason='source_provenance_mismatch'
        if any(row.get(key)!=value for key,value in [('example_id',eid),('language',lang),('split',split),('label',label),('lookahead_words',k),('source_id',sid)]):reason='record_and_sql_metadata_disagree'
        if label not in ('EOT','NEOT') or k not in (1,2):reason='invalid_label_or_horizon'
        if reason:
            bad[reason]+=1
            if len(examples)<50:examples.append({'example_id':eid,'language':lang,'split':split,'reason':reason,'source_id':sid})
        counts[(lang,split,label,k)]+=1;n+=1
        if time.monotonic()-last>15:
            atomic_json(data.JOB/'verification_status.json',{'stage':'VERIFYING_EVERY_ACCEPTED_ENCODED_INPUT','generation':manifest['generation'],'verified_rows':n,'violations':sum(bad.values()),'language':lang});last=time.monotonic()
    assert n==unique_rows,'Sequential row scan and unique-key scan disagree'
    fingerprints={}
    for lang in data.LANGUAGES:
        fingerprints[lang]={}
        for split in ('train','dev'):
            digest=hashlib.sha256();total=0
            for eid,h,label,k in db.execute("SELECT example_id,input_hash,label,k FROM records INDEXED BY accepted_fingerprint_cover WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id",(lang,split)):
                digest.update((eid+'|'+h+'|'+label+'|'+str(k)+'\n').encode());total+=1
            actual={'rows':total,'sha256':digest.hexdigest()};expected=manifest['fingerprints'][lang][split]
            assert actual['rows']==expected['rows'] and actual['sha256']==expected['sha256']
            fingerprints[lang][split]=actual
    db.close()
    actual_counts={tuple((r['language'],r['split'],r['label'],r['lookahead_words'])):r['rows'] for r in manifest['counts']}
    assert dict(counts)==actual_counts,'Manifest class/horizon counts disagree with actual rows'
    result={'complete':not bad,'generation':manifest['generation'],'accepted_rows_checked':n,
      'database_sha256':manifest['database_sha256'],'verifier_source_sha256':code_sha,'tokenizer_file_hashes':tokenizer_hashes,
      'fingerprints':fingerprints,'violations':dict(bad),'violation_examples':examples,
      'all_rows_checked_for_encoded_slots_token_bounds_hash_and_provenance':True,
      'primary_all_row_normalization_check':manifest['lowercase_punctuation_free'],
      'unique_index_key_rows_checked':unique_rows,'sequential_accepted_record_scan':True,'model_inference_or_test_scoring':False,'human_label_quality_certified':False}
    atomic_json(receipt if not bad else folder/'verification_failure.json',result)
    atomic_json(data.JOB/'verification_status.json',{'stage':'COMPLETE' if not bad else 'FAILED_PRESERVED','generation':manifest['generation'],'verified_rows':n,'violations':dict(bad)})
    if bad:raise RuntimeError('Accepted input contract violations; no training allowed: '+str(dict(bad)))
    return result
