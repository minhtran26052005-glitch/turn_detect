from turn_v2.natural_v6_finetune import encode_packet


class Tokenizer:
    cls_token_id=1
    sep_token_id=2

    def encode(self,text,add_special_tokens=False):
        return [int(value) for value in text.split()]


def test_both_immediate_partner_and_current_tail_survive_long_input():
    current=' '.join(str(i) for i in range(1000,1800))
    context=' '.join(str(i) for i in range(2000,2800))
    values=encode_packet(Tokenizer(),{'text':current,'context':context})
    cut=values.index(2)
    assert len(values)==512
    assert values[cut+1:-1]==list(range(1546,1800))
    assert values[1:cut]==list(range(2545,2800))


def test_target_labels_and_future_reference_fields_do_not_change_model_input():
    row={'text':'20 21','context':'10 11'}
    injected={**row,'label':'EOT','source_id':'future','other_speech_first3seconds':2.,'hidden_suffix':'22 23'}
    assert encode_packet(Tokenizer(),row)==encode_packet(Tokenizer(),injected)==[1,10,11,2,20,21,2]
