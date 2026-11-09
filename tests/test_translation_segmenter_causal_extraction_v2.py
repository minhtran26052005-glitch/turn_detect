import unittest
from turn_v2.translation_segmenter_causal_extraction_v2 import all_sentence_spans,examples_from_page_causal


class LanguageAbbreviationTests(unittest.TestCase):
    def test_city_abbreviation_does_not_create_a_false_EOT(self):
        text='Nhóm khảo sát đã đi đến TP. Hồ Chí Minh trong chuyến công tác. Sau đó họ trở về nhà.'
        spans=list(all_sentence_spans(text,'vi'));self.assertEqual(len(spans),2);self.assertIn('TP. Hồ Chí Minh',spans[0].text)
        rows=examples_from_page_causal('vi','sample','sample','',text,'train',None)
        self.assertTrue(all(not r['context'].endswith(' tp') for r in rows if r['label']=='EOT'))

    def test_academic_title_is_not_a_terminal_period(self):
        text='Bài giảng do PGS.TS. Ngô Thế Bính biên soạn cho sinh viên. Đây là nội dung mới.'
        spans=list(all_sentence_spans(text,'vi'));self.assertEqual(len(spans),2);self.assertIn('PGS.TS. Ngô',spans[0].text)

    def test_short_future_and_unterminated_tail_remain_contiguous(self):
        text='This is a complete sentence with enough words. Yes. then a partial'
        rows=examples_from_page_causal('en','x','x','',text,'train',None)
        self.assertEqual([r['text'] for r in rows if r['label']=='EOT' and r['sentence_index']==0],['yes','yes then'])


if __name__=='__main__':unittest.main()
