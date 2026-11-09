"""Quarantine derived TRAIN rows before the immutable generation is published.

Original corpus and source snapshot are read-only. Hold the derived-DB writer
lock before briefly pausing its writer, so no uncommitted source is interrupted.
"""
import os,sys,json,sqlite3,signal,hashlib
from pathlib import Path
root=Path('/content/turn_detect_v2');sys.path.insert(0,str(root));os.chdir(root)
from turn_v2.translation_segmenter_full_campaign_v1 import live
from turn_v2.translation_segmenter_causal_extraction_v2 import examples_from_page_causal
from turn_v2.translation_segmenter_v15c import atomic_json,ensure_jieba
run=root/'outputs/translation_segmenter_v15d_20261026';job=run/'full_epoch_v1'
assert not (job/'script_quarantine_receipt_v1.json').exists()
assert not (job/'current_dataset.json').exists(),'Generation is already immutable; never edit it'
report=json.loads((job/'script_audit_report.json').read_text());assert report['complete'] and report['pages']==40647
request=json.loads((job/'data_build_request.json').read_text());folder=job/'datasets'/request['generation']
assert not (folder/'manifest.json').exists()
source=sqlite3.connect('file:'+str(folder/'source_snapshot.sqlite')+'?mode=ro',uri=True)
jieba=ensure_jieba();plans=[]
for flag in report['flags']:
    lang=flag['language'];pid=flag['source_id'].split(':',1)[1]
    row=source.execute('SELECT title,url,text,text_sha256,split FROM sources WHERE language=? AND page_id=?',(lang,pid)).fetchone()
    title,url,text,checksum,split=row;assert checksum==flag['source_sha256']
    assert split in ('train','dev')
    examples=examples_from_page_causal(lang,pid,title,url,text,split,jieba)
    plans.append((flag,sorted({r['example_id'] for r in examples})))
source.close()
writer=json.loads((job/'data_launcher_v2.json').read_text());assert live(writer)
db=sqlite3.connect(folder/'dataset.sqlite',timeout=30);paused=False;result=[]
try:
    db.execute('BEGIN IMMEDIATE')
    assert not (folder/'manifest.json').exists() and not (job/'current_dataset.json').exists()
    processed=db.execute('SELECT COUNT(*) FROM processed_sources').fetchone()[0];assert processed<40647
    assert live(writer);os.kill(writer['pid'],signal.SIGSTOP);paused=True
    db.execute("CREATE TABLE IF NOT EXISTS quarantined_records_v1 AS SELECT records.*,CAST('' AS TEXT) AS quarantine_reason FROM records WHERE 0")
    for flag,ids in plans:
        language=flag['language'];pid=flag['source_id'].split(':',1)[1]
        old=db.execute('SELECT rejected,generated,text_sha256 FROM processed_sources WHERE language=? AND page_id=?',(language,pid)).fetchone()
        if old is None or old[0] is not None:continue
        assert old[2]==flag['source_sha256']
        count=0
        for eid in ids:
            existing=db.execute('SELECT source_id,split FROM records WHERE example_id=?',(eid,)).fetchone()
            if existing is None:continue
            assert existing==(flag['source_id'],flag['split'])
            db.execute('INSERT INTO quarantined_records_v1 SELECT records.*,? FROM records WHERE example_id=?',(flag['reason'],eid))
            db.execute('DELETE FROM records WHERE example_id=?',(eid,));count+=1
        db.execute('UPDATE processed_sources SET rejected=? WHERE language=? AND page_id=?',('script_audit_quarantine_v1:'+flag['reason'],language,pid))
        result.append({'language':language,'source_id':flag['source_id'],'source_sha256':flag['source_sha256'],
                       'split':flag['split'],'reason':flag['reason'],'generated_before_quarantine':old[1],'distinct_records_quarantined':count})
    receipt={'source_pages_flagged':len(report['flags']),'additional_pages_quarantined':len(result),
       'distinct_records_quarantined':sum(x['distinct_records_quarantined'] for x in result),'sources':result,
       'original_sources_and_test_unchanged':True,'derived_records_preserved_in':'quarantined_records_v1',
       'labels_not_reassigned':True,'curation_before_any_new_optimizer_step':True,
       'script_report_sha256':hashlib.sha256((job/'script_audit_report.json').read_bytes()).hexdigest()}
    db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',('script_quarantine_receipt_v1',json.dumps(receipt)))
    db.commit();atomic_json(job/'script_quarantine_receipt_v1.json',receipt)
    print(json.dumps(receipt))
finally:
    if db.in_transaction:db.rollback()
    db.close()
    if paused and live(writer):os.kill(writer['pid'],signal.SIGCONT)
