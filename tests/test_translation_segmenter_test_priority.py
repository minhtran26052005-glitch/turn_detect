from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from turn_v2 import translation_segmenter_test_priority_v1 as priority


class PriorityResumeTests(unittest.TestCase):
    def test_pid_reuse_or_identity_mismatch_never_receives_continue(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(priority,'CONTROL',Path(folder)),patch.object(priority,'process_matches',return_value=False),patch.object(priority.os,'kill') as kill,patch.object(priority.signal,'SIGCONT',18,create=True):
            priority.resumed({'pid':12345},'test mismatch')
            kill.assert_not_called()

    def test_only_verified_campaign_parent_is_resumed(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(priority,'CONTROL',Path(folder)),patch.object(priority,'process_matches',return_value=True),patch.object(priority.os,'kill') as kill,patch.object(priority.signal,'SIGCONT',18,create=True):
            priority.resumed({'pid':12345,'start_ticks':'fixed'},'test complete')
            kill.assert_called_once_with(12345,18)
            self.assertTrue((Path(folder)/'resume_receipt.json').exists())


if __name__=='__main__':unittest.main()
