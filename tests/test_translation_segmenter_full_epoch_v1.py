import sqlite3
import unittest
import tempfile
import hashlib
from pathlib import Path
import numpy as np
from turn_v2.translation_segmenter_epoch_sampling_v1 import make_order,batches,planned_steps,verify_complete,epoch_seed
from turn_v2.translation_segmenter_full_data_v2 import schema,resolve_collisions,valid_record
from turn_v2.translation_segmenter_full_train_v3 import convergence,KEY
from turn_v2.translation_segmenter_causal_extraction_v2 import examples_from_page_causal
from turn_v2.translation_segmenter_full_train_v2 import preserve_previous_recipe
from turn_v2.translation_segmenter_full_data_v3 import obvious_wrong_script
from turn_v2.translation_segmenter_full_verify_v1 import check_encoded
from turn_v2.translation_segmenter_exact_encoding_v1 import ExactTailEncoder
from turn_v2.monolingual_data import encode_tail
from turn_v2.translation_segmenter_full_data_v5 import resolve_collisions as prove_unique_inputs
from unittest.mock import patch


class FullEpochTests(unittest.TestCase):
    def test_unique_accepted_key_proves_overlap_absent_after_resume(self):
        db=sqlite3.connect(':memory:');schema(db)
        for eid,split,h in [('a','train','one'),('b','dev','two')]:
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,'vi',split,'EOT',1,'source',h,b'', '{}','accepted'))
        with patch('turn_v2.translation_segmenter_full_data_v5.state'):
            proof=prove_unique_inputs(db)
        self.assertTrue(proof['resumed_committed_dispositions'])
        self.assertTrue(proof['zero_train_dev_overlap_implied'])
        with self.assertRaises(sqlite3.IntegrityError):
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',('c','vi','dev','EOT',1,'source','one',b'', '{}','accepted'))

    def test_unique_proof_rejects_preexisting_accepted_train_dev_overlap(self):
        db=sqlite3.connect(':memory:');schema(db)
        for eid,split in [('a','train'),('b','dev')]:
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,'en',split,'EOT',1,'source','same',b'', '{}','accepted'))
        with patch('turn_v2.translation_segmenter_full_data_v5.state'):
            with self.assertRaises(sqlite3.IntegrityError):prove_unique_inputs(db)

    def test_pending_rows_still_quarantine_conflicts_before_unique_proof(self):
        db=sqlite3.connect(':memory:');schema(db)
        for eid,label in [('a','EOT'),('b','NEOT')]:
            db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,'zh','train',label,1,'source','same',b'', '{}','pending'))
        with patch('turn_v2.translation_segmenter_full_data_v5.state'):
            proof=prove_unique_inputs(db)
        self.assertFalse(proof['resumed_committed_dispositions'])
        self.assertEqual(db.execute("SELECT COUNT(*) FROM records WHERE disposition='conflicting_targets'").fetchone()[0],2)

    def test_every_row_once_with_remainder_and_exact_resume(self):
        order=make_order(131,epoch_seed('vi','test',1));visits=np.zeros(131,dtype=np.uint8)
        first=list(batches(order));self.assertEqual(len(first),3)
        for start,stop,chunks in first[:1]:
            for chunk in chunks:visits[chunk]+=1
        # Resume from saved optimizer boundary, including final partial batch.
        for start,stop,chunks in batches(order,64):
            for chunk in chunks:visits[chunk]+=1
        proof=verify_complete(order,131,visits)
        self.assertEqual(proof['unique_rows_seen'],131);self.assertEqual(proof['repeated_rows'],0)
        self.assertEqual(planned_steps(131),3)

    def test_missing_or_duplicate_visits_fail_completion(self):
        order=make_order(10,1);visits=np.ones(10,dtype=np.uint8);visits[3]=0
        with self.assertRaises(RuntimeError):verify_complete(order,10,visits)
        visits[3]=2
        with self.assertRaises(RuntimeError):verify_complete(order,10,visits)

    def test_deterministic_epoch_but_new_order_for_next_epoch(self):
        a=make_order(100,epoch_seed('zh','g',1));b=make_order(100,epoch_seed('zh','g',1));c=make_order(100,epoch_seed('zh','g',2))
        np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,c))

    def test_production_size_vi_en_zh_permutations_have_exact_coverage(self):
        # Production C75 cardinalities, including non-divisible final batches.
        for language,n in [('vi',865075),('en',449514),('zh',1128617)]:
            order=make_order(n,epoch_seed(language,'scale_check',1))
            visits=np.zeros(n,dtype=np.uint8);cursor=0
            for begin,end,chunks in batches(order):
                for chunk in chunks:visits[chunk]+=1
                cursor=end
            proof=verify_complete(order,cursor,visits)
            self.assertEqual(proof['unique_rows_seen'],n)
            self.assertEqual(proof['total_row_visits'],n)

    def test_conflicting_inputs_excluded_and_dev_protected(self):
        db=sqlite3.connect(':memory:');schema(db)
        def put(eid,split,label,h):db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)',(eid,'vi',split,label,1,'source',h,b'', '{}','pending'))
        put('a','train','EOT','conflict');put('b','train','NEOT','conflict')
        put('c','train','EOT','overlap');put('d','dev','EOT','overlap')
        put('e','train','EOT','duplicate');put('f','train','EOT','duplicate')
        put('g','train','EOT','test');db.execute("INSERT INTO protected VALUES('vi','test')")
        resolve_collisions(db);states=dict(db.execute('SELECT example_id,disposition FROM records'))
        self.assertEqual(states['a'],'conflicting_targets');self.assertEqual(states['b'],'conflicting_targets')
        self.assertEqual(states['c'],'train_dev_overlap');self.assertEqual(states['d'],'accepted')
        self.assertEqual(states['e'],'accepted');self.assertEqual(states['f'],'duplicate_input');self.assertEqual(states['g'],'protected_test_input')

    def test_input_contract_rejects_wrong_future_or_punctuation(self):
        row={'language':'en','split':'train','label':'EOT','lookahead_words':1,'context':'it works','text':'next','target_punctuation':'.'}
        self.assertIsNone(valid_record(row));self.assertIsNotNone(valid_record({**row,'text':'two words'}))
        self.assertIsNotNone(valid_record({**row,'context':'It works.'}))

    def test_real_extractor_rows_get_sql_split_before_validation(self):
        rows=examples_from_page_causal('en','x','title','','One complete sentence ends here. Next sentence, with an internal break. End.','train',None)
        self.assertGreater(len(rows),0)
        # Legacy make_example stores split in SQL, not record_json.
        for row in rows:
            row['split']='train'
            self.assertIsNone(valid_record(row))

    def test_dev_saturation_reduces_lr_and_preserves_best_policy(self):
        history=[{'dev_cross_entropy':.5,'train_weighted_loss':.4} for _ in range(8)]
        state=convergence(history)
        self.assertTrue(state['converged']);self.assertLess(state['lr_scale'],1)
        self.assertFalse(state['loss_zero_required']);self.assertFalse(state['test_used'])

    def test_improving_dev_continues_full_epochs(self):
        history=[{'dev_cross_entropy':.5-i*.01,'train_weighted_loss':.4-i*.01} for i in range(8)]
        self.assertFalse(convergence(history)['converged'])

    def test_improving_boundary_dev_score_prevents_false_plateau_from_ce(self):
        history=[{'dev_cross_entropy':.5+i*.01,'train_weighted_loss':.4-i*.01,KEY:.6+i*.01} for i in range(10)]
        state=convergence(history)
        self.assertFalse(state['converged']);self.assertEqual(state['dev_stale_epochs'],0)

    def test_starting_optimizer_and_best_survive_atomic_checkpoint_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);(folder/'best').mkdir()
            (folder/'latest_resume.pt').write_bytes(b'old optimizer and rng control')
            (folder/'best/model.safetensors').write_bytes(b'old best weights control')
            preserve_previous_recipe(folder)
            replacement=folder/'new.pt';replacement.write_bytes(b'new full epoch optimizer')
            replacement.replace(folder/'latest_resume.pt')
            self.assertEqual((folder/'previous_recipe_v15d/latest_resume.pt').read_bytes(),b'old optimizer and rng control')
            self.assertEqual((folder/'previous_recipe_v15d/best/model.safetensors').read_bytes(),b'old best weights control')
            self.assertTrue((folder/'previous_recipe_v15d/archive_receipt.json').exists())

    def test_cyrillic_mixing_is_quarantined_without_dropping_vietnamese_accents(self):
        self.assertFalse(obvious_wrong_script('vi','đây là tiếng việt có dấu và chữ đ '+('ngôn ngữ '*30)))
        self.assertTrue(obvious_wrong_script('vi','русский текст '*40+' tiếng việt'))
        self.assertTrue(obvious_wrong_script('en','русский текст '*40+' english'))

    def test_japanese_kana_mixture_is_quarantined_in_zh_sources(self):
        self.assertTrue(obvious_wrong_script('zh','中文语言'*30+'あいうえお'*30))
        self.assertFalse(obvious_wrong_script('zh','中文语言'*50+' english words'))

    def test_encoded_slot_audit_rejects_lost_left_context_after_truncation(self):
        row={'context':'hello world','text':'next','lookahead_words':1}
        raw=np.array([1,5,2,6,2],dtype='<u4').tobytes()
        self.assertIsNone(check_encoded(row,raw,hashlib.sha256(raw).hexdigest(),1,2,0,100))
        raw=np.array([1,2,6,7,2],dtype='<u4').tobytes()
        self.assertEqual(check_encoded(row,raw,hashlib.sha256(raw).hexdigest(),1,2,0,100),'empty_left_or_future_slot')

    def test_encoded_slot_audit_rejects_corruption_or_empty_future(self):
        row={'context':'hello','text':'next','lookahead_words':1}
        raw=np.array([1,5,2,6,2],dtype='<u4').tobytes()
        self.assertEqual(check_encoded(row,raw,'bad-hash',1,2,0,100),'encoded_hash_mismatch')
        raw=np.array([1,5,6,2,2],dtype='<u4').tobytes()
        self.assertEqual(check_encoded(row,raw,hashlib.sha256(raw).hexdigest(),1,2,0,100),'empty_left_or_future_slot')

    def test_cached_encoder_matches_official_for_unicode_and_truncation(self):
        class Tokenizer:
            cls_token_id=1;sep_token_id=2
            def encode(self,text,add_special_tokens=False):return [10+ord(c)%500 for c in text]
        tok=Tokenizer();cache=ExactTailEncoder(tok)
        for context,text in [('Xin CHÀO, bạn!','Ngày mai'),('中文。继续说话','下一句'),('a '*1000,'z'*1000),('','hello')]:
            for size in (3,5,16,256):
                self.assertEqual(cache.encode(context,text,size),encode_tail(tok,context,text,size))

    def test_cached_encoder_reuses_same_left_for_two_lookahead_horizons(self):
        class Tokenizer:
            cls_token_id=1;sep_token_id=2
            def __init__(self):self.context_calls=0
            def encode(self,text,add_special_tokens=False):
                if text=='long source context':self.context_calls+=1
                return [10+ord(c)%500 for c in text]
        tok=Tokenizer();cache=ExactTailEncoder(tok)
        cache.encode('long source context','next');cache.encode('long source context','next word')
        self.assertEqual(tok.context_calls,1);self.assertEqual(cache.context_hits,1)


if __name__=='__main__':unittest.main()
