import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from turn_v2.translation_segmenter_parser_audit_v1 import audit_language
from turn_v2.translation_segmenter_v15c import examples_from_page


class ParserAuditTests(unittest.TestCase):
    def audit(self,text):
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE sources(language,page_id,text)');db.execute('INSERT INTO sources VALUES(?,?,?)',('en','sample',text))
        rows=examples_from_page('en','sample','sample','',text,'test',None)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.jsonl.gz'
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                for i,row in enumerate(rows):stream.write(json.dumps({**row,'included_in_clean_test':True,'correct':i!=0})+'\n')
            result=audit_language(db,'en',path,None)
        db.close();return result

    def test_counts_discontinuous_lookahead_separately_from_model_errors(self):
        result=self.audit('This is a complete sentence with enough words. Yes. Another proper sentence has enough words here.')
        c=result['counts'];self.assertEqual(c['aligned_EOT_rows'],2);self.assertEqual(c['non_contiguous_lookahead_rows'],2)
        self.assertEqual(c['non_contiguous_lookahead_model_errors'],1)
        self.assertEqual({r['expected_future'] for r in result['examples']},{'yes','yes another'})

    def test_contiguous_source_is_not_flagged(self):
        result=self.audit('This is a complete sentence with enough words. Another proper sentence has enough words here.')
        self.assertEqual(result['counts']['contiguous_lookahead_rows'],2)
        self.assertEqual(result['counts'].get('non_contiguous_lookahead_rows',0),0)


if __name__=='__main__':unittest.main()
