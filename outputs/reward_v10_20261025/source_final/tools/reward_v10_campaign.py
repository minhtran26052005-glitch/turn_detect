"""Bounded GPU labeling and independent QAT/reward experiments."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/content/turn_detect_v2');sys.path.insert(0,str(root));from turn_v2.common import write_json
    out=root/'outputs/reward_v10_20261025';out.mkdir(parents=True,exist_ok=True);start=time.monotonic();receipts=[]
    def status(stage,**kwargs):write_json(out/'progress.json',{'stage':stage,'receipts':receipts,'elapsed_seconds':time.monotonic()-start,**kwargs})
    for name,module,extra,timeout in [('reward_v10_qwen_20261025','reward_v10_qwen',[],1500)]+[('reward_v10_'+lang+'_20261025','reward_v10_train',['--language',lang],2400) for lang in ['vi','en','zh']]:
        status('RUNNING',job=name)
        command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout',str(timeout),'--',sys.executable,'-u','-m','turn_v2.'+module,*extra]
        result=subprocess.run(command,cwd=root,timeout=timeout+60)
        state=json.loads((root/'jobs'/name/'status.json').read_text());receipts.append({'job':name,'status':state['status'],'exit_code':state.get('exit_code')})
        if result.returncode or state['status']!='COMPLETED':status('FAILED',failed_job=name);raise RuntimeError('Inspect '+name+'; preserved outputs, no silent relaunch')
    status('QWEN_AND_THREE_LANGUAGE_REWARD_ABLATIONS_COMPLETE',quality_target_certified=False)


if __name__=='__main__':main()
