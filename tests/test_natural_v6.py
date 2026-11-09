from turn_v2.natural_v6_data import choose_peer,vi_turn_start


def test_immediate_partner_must_finish_before_current_speech():
    current={'call_id':'call1','speaker':'a'}
    previous={'id':'past','call_id':'call1','speaker':'b','call_start':1.,'duration':2.,'eot_len':.5}
    future={'id':'future','call_id':'call1','speaker':'b','call_start':2.,'duration':4.,'eot_len':.5}
    assert choose_peer(current,[previous,future],3.)['id']=='past'


def test_vi_current_turn_resets_only_after_observed_partner_handoff():
    own=[{'start':0,'end':16000},{'start':80000,'end':96000},{'start':104000,'end':112000}]
    other=[{'start':19200,'end':76800}]
    assert vi_turn_start(own,other,2)==(5.,1.)


def test_future_partner_response_cannot_change_current_turn_start():
    own=[{'start':0,'end':16000},{'start':80000,'end':96000},{'start':104000,'end':112000}]
    other=[{'start':19200,'end':76800}]
    future={'start':115200,'end':160000}
    assert vi_turn_start(own,other,2)==vi_turn_start(own,other+[future],2)
