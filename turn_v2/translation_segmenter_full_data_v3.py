"""Build an immutable, fully audited TRAIN/DEV generation on Colab.

Original sources, SQLite, scored tests and reserved test pages are never edited.
Each source page is re-extracted with contiguous lookahead. Exact encoded-input
collisions are resolved before training; conflicting targets are quarantined.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import unicodedata
import zipfile

from .common import normalize
from . import translation_segmenter_causal_extraction_v2 as extraction
from .monolingual_data import encode_tail
from .translation_segmenter_v15c import LABELS, file_sha, atomic_json

ROOT = Path('outputs/translation_segmenter_v15d_20261026')
JOB = ROOT / 'full_epoch_v1'
LANGUAGES = ('vi', 'en', 'zh')
RECIPE = 'FULL_EPOCH_CONTIGUOUS_V1'


def now(): return datetime.now(timezone.utc).isoformat()


def state(stage, **fields):
    atomic_json(JOB / 'data_status.json', {'at_utc': now(), 'stage': stage, **fields})


def input_hash(ids):
    # Explicit fixed-width little endian, independent of Python int encoding.
    import struct
    return hashlib.sha256(struct.pack('<' + 'I' * len(ids), *ids)).hexdigest()


def valid_record(row):
    if row['language'] not in LANGUAGES or row['split'] not in ('train', 'dev'):
        return 'invalid_language_or_split'
    if row['label'] not in LABELS or row['lookahead_words'] not in (1, 2):
        return 'invalid_target_or_horizon'
    if not row['context'] or len(row['text'].split()) != row['lookahead_words']:
        return 'invalid_left_or_future_length'
    if any(normalize(row[k]) != row[k] for k in ('context', 'text')):
        return 'not_lowercase_punctuation_free'
    mark = row.get('target_punctuation', '')
    if row['label'] == 'EOT' and mark not in '.!?。！？': return 'bad_eot_mark'
    if row['label'] == 'NEOT' and mark not in ',，;；:：、': return 'bad_neot_mark'
    return None


def obvious_wrong_script(language, text):
    from collections import Counter
    letters=Counter(c for c in text if c.isalpha());total=sum(letters.values())
    if total<100:return False
    if language in ('vi','en'):
        latin=sum(n for c,n in letters.items() if 'LATIN' in unicodedata.name(c,''))
        return latin/total<.8
    han=sum(n for c,n in letters.items() if extraction.base.is_han(c))
    foreign=sum(n for c,n in letters.items() if any(s in unicodedata.name(c,'') for s in ('HIRAGANA','KATAKANA','HANGUL')))
    return han/total<.1 or foreign/total>.2


def schema(db):
    db.executescript('''
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=FULL;
    CREATE TABLE IF NOT EXISTS records(
      example_id TEXT PRIMARY KEY,language TEXT,split TEXT,label TEXT,k INTEGER,
      source_id TEXT,input_hash TEXT,tokens BLOB,record_json TEXT,
      disposition TEXT NOT NULL DEFAULT 'pending');
    CREATE INDEX IF NOT EXISTS records_input ON records(language,input_hash);
    CREATE INDEX IF NOT EXISTS records_split ON records(language,split,disposition,example_id);
    CREATE TABLE IF NOT EXISTS protected(language TEXT,input_hash TEXT,PRIMARY KEY(language,input_hash));
    CREATE TABLE IF NOT EXISTS processed_sources(language TEXT,page_id TEXT,text_sha256 TEXT,
      source_key TEXT,split TEXT,generated INTEGER,rejected TEXT,
      PRIMARY KEY(language,page_id));
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
    ''')


def resolve_collisions(db):
    """No target is voted/relabelled. Ambiguous exact inputs are all excluded."""
    db.executescript('''
    CREATE TEMP TABLE input_summary AS
      SELECT language,input_hash,COUNT(DISTINCT label) labels,
        MIN(CASE WHEN split='dev' THEN example_id END) dev_id,
        MIN(CASE WHEN split='train' THEN example_id END) train_id
      FROM records GROUP BY language,input_hash;
    CREATE UNIQUE INDEX temp.input_summary_key ON input_summary(language,input_hash);
    UPDATE records AS r SET disposition=CASE
      WHEN EXISTS(SELECT 1 FROM protected p WHERE p.language=r.language AND p.input_hash=r.input_hash)
        THEN 'protected_test_input'
      WHEN (SELECT labels FROM input_summary s WHERE s.language=r.language AND s.input_hash=r.input_hash)>1
        THEN 'conflicting_targets'
      WHEN r.example_id=(SELECT COALESCE(dev_id,train_id) FROM input_summary s
           WHERE s.language=r.language AND s.input_hash=r.input_hash) THEN 'accepted'
      WHEN r.split='train' AND (SELECT dev_id FROM input_summary s
           WHERE s.language=r.language AND s.input_hash=r.input_hash) IS NOT NULL THEN 'train_dev_overlap'
      ELSE 'duplicate_input' END;
    DROP TABLE input_summary;
    ''')
    db.commit()


def build(source_path=None, generation=None):
    import numpy as np
    from transformers import AutoTokenizer
    from .translation_segmenter_v15c import ensure_jieba
    JOB.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 45 * 1024**3:
        raise RuntimeError('Need 45 GiB headroom to build; preserve 25 GiB reserve')
    source_path = Path(source_path or ROOT / 'corpus.sqlite')
    source = sqlite3.connect('file:' + str(source_path) + '?mode=ro', uri=True)
    cycle_row = source.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()
    cycle = int(cycle_row[0]) - 1
    generation = generation or f'cycle{cycle:06d}_{RECIPE.lower()}'
    folder = JOB / 'datasets' / generation
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / 'manifest.json'
    if completed.exists():
        manifest = json.loads(completed.read_text())
        assert manifest['complete'] and file_sha(folder / 'dataset.sqlite') == manifest['database_sha256']
        source.close(); return manifest
    snapshot = folder / 'source_snapshot.sqlite'
    if not snapshot.exists():
        state('PINNING_SQLITE_SOURCE_SNAPSHOT', generation=generation, source_cycle=cycle)
        tmp = snapshot.with_suffix('.partial.sqlite')
        target = sqlite3.connect(tmp); source.backup(target); target.close(); tmp.replace(snapshot)
    source.close()
    source = sqlite3.connect('file:' + str(snapshot) + '?mode=ro', uri=True)
    assert source.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    saved_cycle = source.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()
    # A resumed build always uses its pinned snapshot even if crawling advanced.
    cycle = int(saved_cycle[0]) - 1
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'teachers/vi/best', local_files_only=True)
    assert tokenizer.pad_token_id == 0
    jieba = ensure_jieba()
    path = folder / 'dataset.sqlite'
    db = sqlite3.connect(path, timeout=60); schema(db)
    prior = db.execute("SELECT value FROM meta WHERE key='protected_complete'").fetchone()
    if not prior:
        count = 0
        # Inputs only. Test labels are neither used for cleaning nor scoring.
        for language, raw in source.execute("SELECT language,record_json FROM examples WHERE split='test'"):
            row = json.loads(raw); ids = encode_tail(tokenizer, row['context'], row['text'])['input_ids']
            db.execute('INSERT OR IGNORE INTO protected VALUES(?,?)', (language, input_hash(ids)))
            count += 1
            if count % 10000 == 0:
                db.commit(); state('AUDITING_PROTECTED_INPUT_HASHES', generation=generation, input_rows=count)
        db.execute("INSERT OR REPLACE INTO meta VALUES('protected_complete','true')"); db.commit()
    pages = source.execute("SELECT COUNT(*) FROM sources WHERE split IN ('train','dev')").fetchone()[0]
    page_done = db.execute('SELECT COUNT(*) FROM processed_sources').fetchone()[0]
    generated = db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    last = 0
    query = '''SELECT s.language,s.page_id,s.title,s.url,s.text,s.text_sha256,s.split,
      COALESCE(p.source_key,'unknown') FROM sources s LEFT JOIN source_provenance p
      ON s.language=p.language AND s.page_id=p.page_id
      WHERE s.split IN ('train','dev') ORDER BY s.language,s.page_id'''
    for language, page_id, title, url, text, checksum, split, source_key in source.execute(query):
        old = db.execute('SELECT text_sha256 FROM processed_sources WHERE language=? AND page_id=?', (language,page_id)).fetchone()
        if old:
            assert old[0] == checksum; continue
        if file_sha_bytes(text.encode()) != checksum: raise RuntimeError('Stored source hash mismatch')
        reason = 'quarantined_baai_en' if language == 'en' and 'industrycorpus2' in source_key else None
        if source_key == 'unknown': reason = 'missing_source_provenance'
        if obvious_wrong_script(language,text): reason = 'obvious_wrong_script'
        added = 0
        if not reason:
            for row in extraction.examples_from_page_causal(language,page_id,title,url,text,split,jieba):
                row['split'] = split  # Stored in SQL by the old schema, not inside its JSON.
                reason_row = valid_record(row)
                if reason_row: raise RuntimeError('New extractor violated contract: ' + reason_row)
                row['full_epoch_recipe'] = RECIPE
                row['source_text_sha256'] = checksum
                row['source_key'] = source_key
                ids = encode_tail(tokenizer,row['context'],row['text'])['input_ids']
                tokens = np.asarray(ids,dtype='<u4').tobytes()
                db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',
                  (row['example_id'],language,split,row['label'],row['lookahead_words'],row['source_id'],
                   input_hash(ids),tokens,json.dumps(row,ensure_ascii=False,separators=(',',':')),'pending'))
                added += 1
        db.execute('INSERT INTO processed_sources VALUES(?,?,?,?,?,?,?)',
                   (language,page_id,checksum,source_key,split,added,reason))
        # Source and every derived row commit together, so a restart is exact.
        db.commit(); page_done += 1; generated += added
        if time.monotonic() - last > 15:
            state('REEXTRACTING_ALL_TRAIN_DEV_SOURCES',generation=generation,source_cycle=cycle,
                  language=language,processed_pages=page_done,total_pages=pages,generated_rows=generated)
            last = time.monotonic()
        if shutil.disk_usage(ROOT).free < 25 * 1024**3: raise RuntimeError('Colab reserve reached; preserve build')
    state('RESOLVING_ALL_ENCODED_INPUT_COLLISIONS',generation=generation,generated_rows=generated)
    resolve_collisions(db)
    counts = [dict(zip(('language','split','label','lookahead_words','rows'),r)) for r in db.execute(
        "SELECT language,split,label,k,COUNT(*) FROM records WHERE disposition='accepted' GROUP BY 1,2,3,4 ORDER BY 1,2,3,4")]
    rejected = [dict(zip(('language','split','reason','rows'),r)) for r in db.execute(
        "SELECT language,split,disposition,COUNT(*) FROM records GROUP BY 1,2,3 ORDER BY 1,2,3")]
    source_rejected = [dict(zip(('language','source_key','reason','pages'),r)) for r in db.execute(
        'SELECT language,source_key,rejected,COUNT(*) FROM processed_sources WHERE rejected IS NOT NULL GROUP BY 1,2,3')]
    fingerprints = {}
    for language in LANGUAGES:
        fingerprints[language] = {}
        for split in ('train','dev'):
            h = hashlib.sha256(); n = 0; distribution = Counter()
            for eid, ih, label, k in db.execute("SELECT example_id,input_hash,label,k FROM records WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id",(language,split)):
                h.update((eid+'|'+ih+'|'+label+'|'+str(k)+'\n').encode()); n += 1; distribution[label+':'+str(k)] += 1
            assert all(distribution[label+':'+str(k)] >= 100 for label in LABELS for k in (1,2)), (language,split,distribution)
            fingerprints[language][split] = {'rows':n,'sha256':h.hexdigest(),'strata':dict(distribution)}
    overlap = db.execute("SELECT COUNT(*) FROM records a JOIN records b ON a.language=b.language AND a.input_hash=b.input_hash WHERE a.disposition='accepted' AND b.disposition='accepted' AND a.split='train' AND b.split='dev'").fetchone()[0]
    assert overlap == 0
    original_counts = [dict(zip(('language','split','rows'),r)) for r in source.execute('SELECT language,split,COUNT(*) FROM examples GROUP BY 1,2')]
    source.close(); db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); integrity = db.execute('PRAGMA integrity_check').fetchone()[0]; assert integrity == 'ok'; db.close()
    checksum = file_sha(path)
    state('PACKAGING_VERIFIED_DATASET',generation=generation,source_cycle=cycle)
    archive = folder / 'dataset.zip'; temporary = folder / 'dataset.partial.zip'
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as out: out.write(path,'dataset.sqlite')
    temporary.replace(archive)
    manifest = {'schema':1,'complete':True,'at_utc':now(),'generation':generation,'source_cycle':cycle,
       'recipe':RECIPE,'extraction_recipe':extraction.RECIPE,'database_integrity':integrity,
       'database_sha256':checksum,'archive_sha256':file_sha(archive),'archive_bytes':archive.stat().st_size,
       'fingerprints':fingerprints,'counts':counts,'dispositions':rejected,'source_quarantine':source_rejected,
       'processed_train_dev_pages':page_done,'original_example_counts':original_counts,
       'accepted_exact_encoded_train_dev_overlap':overlap,'max_encoded_tokens':256,
       'all_train_dev_sources_examined':page_done==pages,'protected_test_inputs_hash_only':True,
       'test_labels_used':False,'test_or_reserve_reextracted_or_scored':False,
       'lowercase_punctuation_free':True,'lookahead':[1,2],
       'language_check_scope':'source provenance and conservative script screening, not human language verification',
       'human_gold':False,'extractor_source_sha256':file_sha(Path(extraction.__file__)),'builder_source_sha256':file_sha(Path(__file__)), 'script_screen':'Latin-script majority for vi/en; Han with low Kana/Hangul for zh; conservative quarantine, not human language certification'}
    atomic_json(completed,manifest); atomic_json(JOB/'current_dataset.json',manifest)
    state('COMPLETE',generation=generation,source_cycle=cycle,fingerprints=fingerprints,counts=counts)
    return manifest


def file_sha_bytes(raw): return hashlib.sha256(raw).hexdigest()


def cache(manifest, language, split):
    """Deterministic bounded-memory cache; reconstructible from backed-up DB."""
    import numpy as np
    folder = JOB/'datasets'/manifest['generation']; output = folder/'encoded'/language/split
    output.mkdir(parents=True,exist_ok=True)
    receipt = output/'complete.json'; fingerprint = manifest['fingerprints'][language][split]
    if receipt.exists():
        assert json.loads(receipt.read_text())['fingerprint']==fingerprint
        return load_cache(output)
    n = fingerprint['rows']; arrays = {
      'ids':np.lib.format.open_memmap(output/'ids.npy',mode='w+',dtype=np.int32,shape=(n,256)),
      'lengths':np.lib.format.open_memmap(output/'lengths.npy',mode='w+',dtype=np.int64,shape=(n,)),
      'labels':np.lib.format.open_memmap(output/'labels.npy',mode='w+',dtype=np.int64,shape=(n,)),
      'horizons':np.lib.format.open_memmap(output/'horizons.npy',mode='w+',dtype=np.int64,shape=(n,))}
    db=sqlite3.connect('file:'+str(folder/'dataset.sqlite')+'?mode=ro',uri=True)
    # mDeBERTa padding ID=0 is verified by tokenizer config at dataset build.
    arrays['ids'][:]=0
    for i,(raw,label,k) in enumerate(db.execute("SELECT tokens,label,k FROM records WHERE language=? AND split=? AND disposition='accepted' ORDER BY example_id",(language,split))):
        ids=np.frombuffer(raw,dtype='<u4');arrays['ids'][i,:len(ids)]=ids;arrays['lengths'][i]=len(ids);arrays['labels'][i]=LABELS[label];arrays['horizons'][i]=k
        if i%20000==0:state('ENCODING_VERIFIED_DATASET_CACHE',generation=manifest['generation'],language=language,split=split,rows=i,total=n)
    db.close()
    for a in arrays.values():a.flush()
    atomic_json(receipt,{'fingerprint':fingerprint,'rows':n,'source_database_sha256':manifest['database_sha256']})
    return load_cache(output)


def load_cache(folder):
    import numpy as np
    return tuple(np.load(folder/(key+'.npy'),mmap_mode='r') for key in ('ids','lengths','labels','horizons'))


if __name__=='__main__':
    try:
        request = JOB/'data_build_request.json'
        build(generation=json.loads(request.read_text())['generation'] if request.exists() else None)
    except Exception as exc:
        import traceback
        atomic_json(JOB/'data_error_v3.json',{'at_utc':now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('FAILED_PRESERVED',error=repr(exc));raise
