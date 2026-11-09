"""Real SQLite scans with synthetic tokens; no tokenizer/model execution."""
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from turn_v2 import translation_segmenter_full_verify_v2 as verifier
from turn_v2.translation_segmenter_full_data_v5 import schema, resolve_collisions


class TokenizerShape:
    cls_token_id=1
    sep_token_id=2
    pad_token_id=0

    def __len__(self):return 100


class FullVerifyV2Tests(unittest.TestCase):
    def run_fixture(self,root,broken=False):
        job=root/'full_epoch_v1';folder=job/'datasets'/'synthetic';folder.mkdir(parents=True)
        db=sqlite3.connect(folder/'dataset.sqlite');schema(db)
        counts=[];fingerprints={}
        for i,lang in enumerate(('vi','en','zh')):
            tok=root/'teachers'/lang/'best/tokenizer.json';tok.parent.mkdir(parents=True);tok.write_text('{}')
            fingerprints[lang]={}
            for j,split in enumerate(('train','dev')):
                eid=lang+'-'+split;pid='source-'+split;sid=lang+':'+pid
                ids=[1,10+2*i+j,2,50,2]
                if broken and i==0 and j==0:ids=[1,2,50,2]
                raw=struct.pack('<'+'I'*len(ids),*ids);h=hashlib.sha256(raw).hexdigest()
                row={'example_id':eid,'language':lang,'split':split,'label':'EOT','lookahead_words':1,'source_id':sid,'source_text_sha256':'a'*64,'source_key':'synthetic','context':'hello','text':'next'}
                db.execute('INSERT INTO processed_sources VALUES(?,?,?,?,?,?,?)',(lang,pid,'a'*64,'synthetic',split,1,None))
                db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,lang,split,'EOT',1,sid,h,raw,json.dumps(row),'accepted'))
                fingerprints[lang][split]={'rows':1,'sha256':hashlib.sha256((eid+'|'+h+'|EOT|1\n').encode()).hexdigest()}
                counts.append({'language':lang,'split':split,'label':'EOT','lookahead_words':1,'rows':1})
        with patch('turn_v2.translation_segmenter_full_data_v5.state'):resolve_collisions(db)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
        manifest={'generation':'synthetic','complete':True,'all_train_dev_sources_examined':True,'database_sha256':hashlib.sha256((folder/'dataset.sqlite').read_bytes()).hexdigest(),'counts':counts,'fingerprints':fingerprints,'lowercase_punctuation_free':True}
        fake=types.ModuleType('transformers');fake.AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:TokenizerShape())
        with patch.dict(sys.modules,{'transformers':fake}),patch.object(verifier.data,'ROOT',root),patch.object(verifier.data,'JOB',job):
            return verifier.verify(manifest)

    def test_sequential_record_and_ordered_unique_key_scans_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            result=self.run_fixture(Path(directory))
        self.assertTrue(result['complete']);self.assertEqual(result['accepted_rows_checked'],6)
        self.assertEqual(result['unique_index_key_rows_checked'],6)
        self.assertTrue(result['sequential_accepted_record_scan'])
        self.assertFalse(result['model_inference_or_test_scoring'])

    def test_bad_encoded_slot_blocks_training_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(RuntimeError):self.run_fixture(root,broken=True)
            result=json.loads((root/'full_epoch_v1/datasets/synthetic/verification_failure.json').read_text())
            self.assertFalse(result['complete']);self.assertEqual(result['violations']['bad_special_token_shape'],1)


if __name__=='__main__':unittest.main()
