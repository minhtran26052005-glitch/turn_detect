import json
from pathlib import Path
import pytest
from turn_v2.monolingual_data import partition_language,encode_tail,group
from turn_v2.monolingual_runtime import SeparateLanguageModels
from turn_v2.common import file_hash,input_id


def row(lang,label,source,text):
    return {'language':lang,'label':label,'source_id':source,'source':'proxy','text':text}


def test_language_training_partition_and_protected_inputs():
    protected=row('en','EOT','held','held words')
    natural={'train':[row('en','EOT','call_a','one'),row('en','NEOT','call_b','two'),row('zh','EOT','call_c','three'),protected],
        'dev':[row('en','EOT','call_d','four')]}
    replay={'train':[row('en','EOT','call_d','different input same dev call'),row('en','NEOT','call_e','five')],'dev':[]}
    data,_=partition_language('en',natural,replay,{'en:'+input_id(protected)})
    assert all(r['language']=='en' for rows in data.values() for r in rows)
    assert all(r['text']!='held words' for r in data['train'])
    assert not {group(r) for r in data['train']}&{group(r) for r in data['dev']}
    with pytest.raises(ValueError):partition_language('vi',natural,replay,set())


def test_current_tail_has_priority_over_partner_history():
    class Tokenizer:
        cls_token_id=101;sep_token_id=102
        def encode(self,text,**kwargs):return list(map(int,text.split()))
    assert encode_tail(Tokenizer(),'1 2','3 4 5 6 7',6)['input_ids']==[101,102,5,6,7,102]
    assert encode_tail(Tokenizer(),'1 2 3 4','5 6',7)['input_ids']==[101,3,4,102,5,6,102]


def test_explicit_routing_uses_own_weights_and_has_no_fallback(tmp_path):
    models={}
    for language in ['vi','en','zh']:
        folder=tmp_path/language;folder.mkdir();(folder/'model.onnx').write_bytes(language.encode())
        models[language]={'path':language,'status':'RESEARCH_ONLY','model_sha256':file_hash(folder/'model.onnx')}
    registry=tmp_path/'registry.json';registry.write_text(json.dumps({'models':models,'project_root_relative':'.'}))
    calls=[]
    class Detector:
        def __init__(self,path):calls.append(path.name);self.language=path.name
        def predict(self,text,context):return {'prediction':self.language}
    router=SeparateLanguageModels(registry,Detector)
    assert router.predict('words',language='en')['prediction']=='en'
    assert router.predict('words',language='zh')['prediction']=='zh'
    assert calls==['en','zh']
    with pytest.raises(ValueError):router.predict('words')
    router.entries['vi']['status']='PENDING_TRAINING'
    with pytest.raises(RuntimeError):router.predict('words',language='vi')
