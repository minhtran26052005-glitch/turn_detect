"""Full source-script inventory, not a language model or human adjudication."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import unicodedata
from .translation_segmenter_v15c import atomic_json

JOB=Path('outputs/translation_segmenter_v15d_20261026/full_epoch_v1')


def main():
    request=json.loads((JOB/'data_build_request.json').read_text());folder=JOB/'datasets'/request['generation']
    db=sqlite3.connect('file:'+str(folder/'source_snapshot.sqlite')+'?mode=ro',uri=True)
    cache={};counts=Counter();flags=[];examples=[];done=0;last=0
    for language,pid,text,checksum,split in db.execute("SELECT language,page_id,text,text_sha256,split FROM sources WHERE split IN ('train','dev') ORDER BY language,page_id"):
        assert hashlib.sha256(text.encode()).hexdigest()==checksum
        chars=Counter(c for c in text if c.isalpha());scripts=Counter()
        for c,n in chars.items():
            if c not in cache:
                name=unicodedata.name(c,'');cache[c]=next((s for s in ('LATIN','CYRILLIC','ARABIC','HEBREW','HIRAGANA','KATAKANA','HANGUL','CJK') if s in name),'OTHER')
            scripts[cache[c]]+=n
        letters=sum(scripts.values());reason=None
        if letters>=100:
            if language in ('vi','en') and scripts['LATIN']/letters<.8:reason='NON_LATIN_DOMINANT_IN_LATIN_LANGUAGE_SOURCE'
            if language=='zh' and (scripts['CJK']/letters<.1 or (scripts['HIRAGANA']+scripts['KATAKANA']+scripts['HANGUL'])/letters>.2):reason='NON_CHINESE_SCRIPT_DOMINANT'
        counts[language+':pages']+=1
        if reason:
            counts[language+':script_flags']+=1
            flags.append({'language':language,'source_id':language+':'+pid,'split':split,'reason':reason,'scripts':dict(scripts),'source_sha256':checksum})
            if sum(x['language']==language for x in examples)<3:examples.append({'language':language,'source_id':language+':'+pid,'excerpt':text[:240],'scripts':dict(scripts)})
        done+=1
        if time.monotonic()-last>15:
            atomic_json(JOB/'script_audit_status.json',{'stage':'SCANNING_ALL_SOURCE_SCRIPTS','pages':done,'counts':dict(counts)});last=time.monotonic()
    db.close()
    atomic_json(JOB/'script_audit_report.json',{'complete':True,'pages':done,'counts':dict(counts),'flags':flags,'examples':examples,
      'scope':'Conservative script screen; Latin-script language identity and conversational label quality are not certified',
      'model_inference':False,'data_or_labels_modified':False})
    atomic_json(JOB/'script_audit_status.json',{'stage':'COMPLETE','pages':done,'counts':dict(counts)})


if __name__=='__main__':main()
