import numpy as np
from turn_v2.natural_calibration import summarize_exact


def test_close_scores_are_not_limited_by_coarse_threshold_grid():
    rows=[];probs=[]
    for lang in ['vi','en','zh']:
        for label,p in [('NEOT',.501),('EOT',.503)]:
            rows.append({'language':lang,'evaluation_slice':'natural','label':label});probs.append(p)
    report=summarize_exact(rows,np.array(probs))
    assert report['minimum_dev_slice_joint_score']==1.
    assert all(.501<threshold<=.503 for threshold in report['thresholds'].values())


def test_legacy_and_natural_slice_both_constrain_calibration():
    rows=[];probs=[]
    for lang in ['vi','en','zh']:
        for label,p in [('NEOT',.2),('EOT',.8)]:
            rows.append({'language':lang,'evaluation_slice':'natural','label':label});probs.append(p)
    rows.extend([{'language':'vi','evaluation_slice':'legacy_vi','label':'NEOT'},
        {'language':'vi','evaluation_slice':'legacy_vi','label':'EOT'}]);probs.extend([.9,.1])
    report=summarize_exact(rows,np.array(probs))
    assert report['minimum_dev_slice_joint_score']<1.
    assert not report['development_all_slices_over_90']
