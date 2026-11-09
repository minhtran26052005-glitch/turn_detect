"""Publish bounded reader report files; no model execution or large transfer."""
import base64
import hashlib
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell

J=Path('outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1')
SOURCE=Path('turn_v2/translation_segmenter_diagnostic_publication_v6.py')


def main():
    names=['bao_cao_ket_qua.html','review_complete_v3.html','vi_compression_evidence_v1.json','plain_report_receipt_v3.json',
           'old_dev_c75_vi_report.json','old_dev_c75_vi_protocol.json','old_dev_c75_vi_drive_manifest.json',
           'old_dev_c75_vi_samples.jsonl','report_qa_old_dev/qa_receipt.json','README.md']
    files=[(n,J/n) for n in names]+[
        ('source/build_translation_segmenter_plain_report.py',Path('tools/build_translation_segmenter_plain_report.py')),
        ('source/translation_segmenter_old_dev_c75_vi_v1.py',Path('turn_v2/translation_segmenter_old_dev_c75_vi_v1.py')),
        ('source/vi_small_model_manifest.json',Path('models/turn_v2_distilled_qat4_20261024/model_manifest.json')),
        ('source/vi_small_model_config.json',Path('models/turn_v2_distilled_qat4_20261024/onnx/config.json'))]
    source=SOURCE.read_bytes();checksum=hashlib.sha256(source).hexdigest()
    r=ColabRuntime(0);connection=r.connect();assert connection['execution_state']=='idle'
    preflight="""import hashlib,json
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/'outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1'
assert not (job/'review_publication_v6').exists(),'Publisher exists; inspect rather than relaunch'
(root/'colab').mkdir(exist_ok=True);job.mkdir(parents=True,exist_ok=True)
c=json.loads(Path('/content/turn_detect_old_dev_c75_vi_v1/config.json').read_text());dest=Path(c['drive_path'])/job.name
m=json.loads((dest/'latest_manifest.json').read_text());assert m['complete'] and m['destination_identity']==c['identity']
raw=(dest/m['files']['report.json']['object']).read_bytes();assert hashlib.sha256(raw).hexdigest()==c['parent_report_sha256']
p=job/'report.json'
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
p=job/'drive_manifest.json'
if p.exists() and json.loads(p.read_text())!=m:(job/'drive_manifest_before_v6.json').write_bytes(p.read_bytes())
p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+chr(10))
assert json.loads((job/'status.json').read_text())['stage']=='COMPLETE'
print(json.dumps({'ready':True}))"""
    compile(preflight,'publication_preflight','exec');cell(r,preflight);uploads=[]
    for i,(logical,path) in enumerate(files):
        raw=path.read_bytes();assert len(raw)<5*1024**2;key='reader_legacy_v6_'+str(i)+path.suffix
        r.request('PUT','api/contents/content/turn_detect_v2/colab/'+key,json={'type':'file','format':'base64','content':base64.b64encode(raw).decode()})
        uploads.append((key,logical,hashlib.sha256(raw).hexdigest()))
    r.request('PUT','api/contents/content/turn_detect_v2/colab/'+SOURCE.name,json={'type':'file','format':'base64','content':base64.b64encode(source).decode()})
    code=f'''import json,hashlib,subprocess,sys,os
from pathlib import Path
root=Path('/content/turn_detect_v2');job=root/{str(J).replace(chr(92),'/')!r};control=job/'review_publication_v6'
assert not control.exists();files={{}}
for staging,logical,expected in {uploads!r}:
 raw=(root/'colab'/staging).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
 p=job/logical;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and logical not in ('bao_cao_ket_qua.html','review_complete_v3.html','README.md','source/build_translation_segmenter_plain_report.py'):assert p.read_bytes()==raw,'Conflicting completed evidence'
 p.write_bytes(raw);files[logical]=expected
raw=(root/'colab'/{SOURCE.name!r}).read_bytes();assert hashlib.sha256(raw).hexdigest()=={checksum!r}
p=root/'turn_v2'/{SOURCE.name!r}
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
(job/'source'/{SOURCE.name!r}).write_bytes(raw);files['source/'+{SOURCE.name!r}]={checksum!r}
control.mkdir();(control/'request.json').write_text(json.dumps({{'raw_report_sha256':'fe4555399de148fe34c2e96f7d95d4edb2aa453caa618f0d69a333ebbb238650','files_sha256':files}},indent=2))
command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_diagnostic_publication_v6']
with (control/'worker.log').open('a') as log:p=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,'OMP_NUM_THREADS':'1'}})
a={{'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],'command':command,'source_sha256':{checksum!r}}}
(control/'launcher.json').write_text(json.dumps(a,indent=2));print(json.dumps(a))'''
    compile(code,'publication_launch','exec');result=cell(r,code);r.close()
    (J/'reader_publication_launch_v6.json').write_text(json.dumps({'connection':connection,'launcher':result},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
