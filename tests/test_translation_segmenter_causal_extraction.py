import unittest
from turn_v2.translation_segmenter_causal_extraction_v1 import all_sentence_spans,examples_from_page_causal,audit_legacy_eot
from turn_v2.translation_segmenter_v15c import examples_from_page


class CausalExtractionTests(unittest.TestCase):
    def test_short_sentence_is_never_skipped_in_lookahead(self):
        text='This is a complete sentence with enough words. Yes. Another proper sentence has enough words here.'
        rows=examples_from_page_causal('en','a','a','',text,'train',None)
        first=[r for r in rows if r['sentence_index']==0 and r['label']=='EOT']
        self.assertEqual({r['lookahead_words']:r['text'] for r in first},{1:'yes',2:'yes another'})
        old=examples_from_page('en','a','a','',text,'train',None)[0]
        result=audit_legacy_eot(old,text,'en',None)
        self.assertTrue(result['short_or_unterminated_source_content_skipped'])
        self.assertEqual(result['expected_future'],'yes')

    def test_unterminated_tail_can_supply_future_words(self):
        text='This sentence has a proper final punctuation mark. then a partial'
        rows=examples_from_page_causal('en','a','a','',text,'train',None)
        self.assertEqual([r['text'] for r in rows if r['label']=='EOT'],['then','then a'])

    def test_decimals_and_abbreviations_are_not_sentence_boundaries(self):
        text='Dr. Smith has 3.14 units in his lab. Yes.'
        spans=list(all_sentence_spans(text))
        self.assertEqual(len(spans),2)
        self.assertIn('3.14',spans[0].text)

    def test_last_sentence_internal_punctuation_is_preserved(self):
        rows=examples_from_page_causal('en','a','a','','hello there, please wait now.','train',None)
        self.assertEqual([r['label'] for r in rows],['NEOT','NEOT'])

    def test_han_ascii_period_and_short_spans_are_kept(self):
        self.assertEqual([s.text for s in all_sentence_spans('你好.是。继续说话。')],['你好.','是。','继续说话。'])


if __name__=='__main__':unittest.main()
