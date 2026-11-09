import copy

from turn_v2.natural_v5_canonical import canonical_rows, add_causal_history


def example(case,decision,start,source='a',label='EOT'):
    return {'case_id':case,'source_id':source,'session_group':'same_call','speaker':'a',
        'absolute_decision':decision,'absolute_window_start':start,'text':case,'label':label}


def test_future_asr_history_excluded():
    rows=[example('past',3,0),example('current',20,5,source='b'),example('future',21,6,source='c')]
    actual=add_causal_history(rows)
    assert actual[1]['context']=='speakera: past'
    assert all(item['observed_through']<5 for item in actual[1]['context_observations'])


def test_context_independent_of_eot_labels():
    rows=[example('past',3,0),example('current',20,5,source='b')]
    changed=copy.deepcopy(rows)
    for row in changed:row['label']='NEOT'
    assert [r['context'] for r in add_causal_history(rows)]==[r['context'] for r in add_causal_history(changed)]


def test_sidecar_unifies_both_channels_and_ending_types():
    rows=[];metadata={}
    for source,speaker in [('clip_a_response','a'),('clip_b_response','b')]:
        rows.append({'source_id':source,'session_group':source,'language':'en','score_timestamp':2,'audio_prefix_start':0})
        metadata[source]={'eot_kind':'response','split':'train','call_id':'call42','speaker':speaker,'call_start':0}
    actual,_=canonical_rows(rows,metadata)
    assert actual[0]['session_group']==actual[1]['session_group']=='callhome_en:call42'
    assert actual[0]['split']==actual[1]['split']


def test_bad_language_cannot_hide_behind_large_legacy_dev():
    import numpy as np
    from turn_v2.natural_v5_train import summarize
    rows=[];probabilities=[]
    for lang in ['vi','en','zh']:
        for label,p in [('NEOT',.1),('EOT',.9)]:
            rows.append({'language':lang,'label':label,'evaluation_slice':'natural'})
            probabilities.append(.6 if lang=='zh' else p)
    for _ in range(100):
        rows.append({'language':'vi','label':'EOT','evaluation_slice':'legacy_vi'})
        probabilities.append(.9)
    result=summarize(rows,np.array(probabilities))
    assert result['natural_languages']['zh']['accuracy']==.5
    assert result['minimum_dev_slice_joint_score']==.5
    assert not result['development_target_met']
