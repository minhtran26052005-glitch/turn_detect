"""Run a bounded experiment with durable heartbeat, output, and true exit code."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--timeout',type=int,default=3600);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('Need a command')
    a.job.mkdir(parents=True,exist_ok=False)
    start=time.time();state={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),'command':command,'timeout_seconds':a.timeout}
    def save():
        temporary=a.job/'status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(a.job/'status.json')
    with (a.job/'stdout.log').open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'})
        state['pid']=process.pid;save()
        while process.poll() is None:
            state['elapsed_seconds']=time.time()-start;state['heartbeat_unix']=time.time();save()
            if state['elapsed_seconds']>a.timeout:
                state['status']='TIMEOUT';os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(5)
        state['exit_code']=process.wait()
    if state['status']!='TIMEOUT':state['status']='COMPLETED' if state['exit_code']==0 else 'FAILED'
    state['elapsed_seconds']=time.time()-start;state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
