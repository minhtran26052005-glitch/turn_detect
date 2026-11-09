import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from turn_v2 import translation_segmenter_cache_v1 as cached
from turn_v2.translation_segmenter_full_data_v5 import schema,resolve_collisions
from turn_v2.translation_segmenter_full_campaign_v8 import eligible_ready_languages,prefetch_candidates


class CacheTests(unittest.TestCase):
    def fixture(self,root,n=131):
        folder=root/'datasets'/'synthetic';folder.mkdir(parents=True)
        db=sqlite3.connect(folder/'dataset.sqlite');schema(db);digest=hashlib.sha256();strata={label+':'+str(k):0 for label in ('NEOT','EOT') for k in (1,2)}
        for i in range(n):
            eid=f'{i:06d}';label='EOT' if i%2 else 'NEOT';k=i%2+1;ids=[1,10+i%10,2,20,2]
            raw=struct.pack('<5I',*ids);h=hashlib.sha256(raw).hexdigest()
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,'vi','train',label,k,'vi:source',h,raw,'{}','accepted'))
            digest.update((eid+'|'+h+'|'+label+'|'+str(k)+'\n').encode());strata[label+':'+str(k)]+=1
        # This fixture tests ordinal cache behavior; repeated synthetic token
        # values do not need the real dataset's unique-input constraint.
        db.execute("CREATE INDEX accepted_fingerprint_cover ON records(language,split,disposition,example_id,input_hash,label,k) WHERE disposition='accepted'")
        db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
        m={'generation':'synthetic','database_sha256':hashlib.sha256((folder/'dataset.sqlite').read_bytes()).hexdigest(),'fingerprints':{'vi':{'train':{'rows':n,'sha256':digest.hexdigest(),'strata':strata}}}}
        output=cached.folder_for(m,'vi','train');output.mkdir(parents=True)
        return m,output

    def test_resume_preserves_prefix_and_covers_every_ordinal(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cached.data,'JOB',Path(d)):
            m,folder=self.fixture(Path(d));self.assertIsNone(cached.build_locked(m,'vi','train',row_budget=64))
            before=np.load(folder/'ids.npy',mmap_mode='r')[:64].copy()
            self.assertFalse(cached.ready(m,'vi'))
            arrays=cached.build_locked(m,'vi','train')
            np.testing.assert_array_equal(arrays[0][:64],before)
            self.assertEqual(len(arrays[0]),131);self.assertTrue(cached.ready(m,'vi'))
            for i in range(131):self.assertEqual(arrays[0][i,1],10+i%10)
            for array in arrays:array._mmap.close()

    def test_adopts_only_old_published_prefix_and_rebuilds_tail(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cached.data,'JOB',Path(d)):
            m,folder=self.fixture(Path(d));cached.build_locked(m,'vi','train',row_budget=90)
            (folder/'progress.json').unlink()
            proof=cached.adopt_prefix(m,'vi','train',80)
            self.assertEqual(proof['rows_committed'],80);self.assertEqual(proof['last_example_id'],'000079')
            arrays=cached.build_locked(m,'vi','train')
            self.assertEqual(len(arrays[0]),131)
            for i in range(131):self.assertEqual(arrays[0][i,1],10+i%10)
            for array in arrays:array._mmap.close()

    def test_wrong_saved_ordinal_is_rejected_without_resetting_arrays(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cached.data,'JOB',Path(d)):
            m,folder=self.fixture(Path(d));cached.build_locked(m,'vi','train',row_budget=64)
            before=hashlib.sha256((folder/'ids.npy').read_bytes()).hexdigest()
            p=folder/'progress.json';r=json.loads(p.read_text());r['last_example_id']='wrong';p.write_text(json.dumps(r))
            with self.assertRaises(AssertionError):cached.build_locked(m,'vi','train')
            self.assertEqual(hashlib.sha256((folder/'ids.npy').read_bytes()).hexdigest(),before)

    def test_ready_languages_continue_while_zh_cache_is_missing(self):
        histories={'vi':[],'en':[],'zh':[]};m={}
        result=eligible_ready_languages(histories,['vi','en','zh'],1,m,lambda _,lang,split:lang!='zh')
        self.assertEqual(result,['vi','en'])
        histories['vi']=[{}];histories['en']=[{}]
        self.assertEqual(eligible_ready_languages(histories,['vi','en','zh'],1,m,lambda *_:True),['zh'])

    def test_first_missing_train_cache_precedes_dev_prefetch(self):
        m={'fingerprints':{l:{'train':{'rows':n},'dev':{'rows':100}} for l,n in [('vi',1600),('en',800),('zh',4800)]}}
        result=prefetch_candidates(m,lambda _,lang,split:split=='train' and lang!='zh')
        self.assertEqual(result[0],('zh','train'))


if __name__=='__main__':unittest.main()
