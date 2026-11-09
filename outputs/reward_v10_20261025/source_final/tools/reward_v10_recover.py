"""Preserve completed Qwen; repair fresh-runtime ONNX dependency before train."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/content/turn_detect_v2');sys.path.insert(0,str(root));from turn_v2.common import write_json
    output=root/'outputs/reward_v10_20261025';start=time.monotonic();receipts=[]
    old=json.loads((root/'jobs/reward_v10_vi_20261025/status.json').read_text());assert old['status']=='FAILED'
    assert not (output/'vi').exists(),'Inspect partial training before any resume'
    qwen=json.loads((root/'jobs/reward_v10_qwen_20261025/status.json').read_text());assert qwen['status']=='COMPLETED' and qwen['exit_code']==0
    with (output/'dependency_repair.log').open('w') as log:
        installed=subprocess.run([sys.executable,'-m','pip','install','onnx==1.19.1','onnxruntime==1.23.2'],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    assert installed.returncode==0,'Inspect dependency repair log'
    for language in ['vi','en','zh']:
        name='reward_v10_'+language+'_20261025_r2'
        write_json(output/'recovery_progress.json',{'stage':'TRAINING','language':language,'job':name,'receipts':receipts,'elapsed_seconds':time.monotonic()-start})
        command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','2400','--',sys.executable,'-u','-m','turn_v2.reward_v10_train','--language',language]
        result=subprocess.run(command,cwd=root,timeout=2460);state=json.loads((root/'jobs'/name/'status.json').read_text());receipts.append({'job':name,'status':state['status'],'exit_code':state.get('exit_code')})
        if result.returncode or state['status']!='COMPLETED':raise RuntimeError('Inspect '+name+'; previous phases preserved')
    write_json(output/'recovery_progress.json',{'stage':'COMPLETE_THREE_LANGUAGE_DEV_ABLATIONS','receipts':receipts,'elapsed_seconds':time.monotonic()-start,'quality_target_certified':False})


if __name__=='__main__':main()
