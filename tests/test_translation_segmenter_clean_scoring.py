import unittest
from unittest.mock import patch
import numpy as np
from turn_v2 import translation_segmenter_clean_scoring_v1 as clean


class CleanScoringTests(unittest.TestCase):
    def test_protected_overlapping_rows_never_reach_inference(self):
        encoded=(np.array([[11,12],[21,22],[31,32]]),np.array([2,2,2]),np.array([0,1,0]),np.array([1,2,1]))
        seen=[]
        def fake(model,subset,lang):
            seen.extend(subset[0].tolist());return np.array([.2,.8])
        with patch.object(clean.ev,'infer',fake):
            out=clean.infer_clean(None,encoded,np.array([0,2]),'en')
        self.assertEqual(seen,[[11,12],[31,32]])
        self.assertEqual(out.tolist(),[.2,.8])


if __name__=='__main__':unittest.main()
