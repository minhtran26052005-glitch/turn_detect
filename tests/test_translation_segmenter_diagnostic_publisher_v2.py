import json
from pathlib import Path
import tempfile
import unittest
from turn_v2.translation_segmenter_diagnostic_c75_v2 import pin_metadata


class PublisherMetadataTests(unittest.TestCase):
    def test_live_status_changes_cannot_mutate_already_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'status.json'
            source.write_text(json.dumps({'stage':'FIRST'}),encoding='utf-8')
            pinned=pin_metadata(source,root/'pins')
            replacement=root/'status.new';replacement.write_text(json.dumps({'stage':'COPYING_STATUS'}),encoding='utf-8');replacement.replace(source)
            self.assertEqual(json.loads(pinned.read_text())['stage'],'FIRST')
            self.assertEqual(json.loads(source.read_text())['stage'],'COPYING_STATUS')


if __name__=='__main__':unittest.main()
