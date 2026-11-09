"""Protocol checks only; no local model training or inference."""
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from turn_v2 import translation_segmenter_diagnostic_c75_v1 as diag


class DiagnosticPartitionTests(unittest.TestCase):
    def test_source_partition_excludes_old_test_and_keeps_reserve_unscored(self):
        db=sqlite3.connect(':memory:')
        db.executescript('CREATE TABLE sources(language,page_id,added_cycle,text_sha256,split); CREATE TABLE examples(language,split,example_id,record_json);')
        chosen={True:None,False:None}
        for i in range(100):
            sid='vi:'+str(i)
            chosen[diag.reserve_page(sid)]=str(i)
            if all(v is not None for v in chosen.values()):break
        for page,cycle in [('old',45),(chosen[True],46),(chosen[False],75)]:
            db.execute('INSERT INTO sources VALUES(?,?,?,?,?)',('vi',page,cycle,page,'test'))
            row={'source_id':'vi:'+page,'context':'plain input','text':'next'}
            db.execute('INSERT INTO examples VALUES(?,?,?,?)',('vi','test',page,json.dumps(row)))
        with tempfile.TemporaryDirectory() as folder,patch.object(diag,'JOB',Path(folder)):
            selected,protected,receipt=diag.select_sources(db,'vi')
        self.assertEqual([json.loads(raw)['source_id'] for raw in selected],['vi:'+chosen[False]])
        self.assertEqual(len(protected),2)
        self.assertFalse(receipt['reserve_scored'])
        self.assertEqual(receipt['old_test_rows_not_rescored'],1)
        db.close()

    def test_reserve_assignment_is_stable_and_source_grouped(self):
        self.assertEqual([diag.reserve_page('en:'+str(i)) for i in range(100)], [diag.reserve_page('en:'+str(i)) for i in range(100)])
        n=sum(diag.reserve_page('zh:'+str(i)) for i in range(2000))
        self.assertTrue(300 < n < 500)


if __name__=='__main__':unittest.main()
