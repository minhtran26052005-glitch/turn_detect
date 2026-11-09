"""Bounded search: smaller GRU students first; select using frozen dev only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    out=root/'runs/distill_campaign_20261024';out.mkdir(exist_ok=False)
    started=time.time();reports=[]
    def save(stage,**kwargs):(out/'status.json').write_text(json.dumps({'stage':stage,'elapsed_seconds':time.time()-started,'reports':reports,**kwargs},indent=2)+'\n')
    save('WAITING_FOR_TEACHER_TRAIN_TARGETS')
    for _ in range(120):
        cache=root/'runs/distill_cache_20261024/cache_report.json'
        if cache.exists():break
        job=root/'jobs/distill_teacher_20261024/status.json'
        if job.exists() and json.loads(job.read_text())['status'] in ('FAILED','TIMEOUT'):raise RuntimeError('Teacher cache job failed; inspect its receipt')
        time.sleep(5)
    else:raise TimeoutError('Teacher cache was not ready in 10 minutes')
    assert json.loads(cache.read_text())['status']=='TRAIN_TARGETS_COMPLETE'
    candidate=(64,64,1)
    for attempt in range(3):
        hidden,embedding,layers=candidate;name=f'distill_gru_h{hidden}_e{embedding}_l{layers}_20261024'
        save('TRAINING',candidate=name)
        command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','2400','--',sys.executable,'-u','-m','turn_v2.distill','--output','runs/'+name,'--hidden-size',str(hidden),'--embedding-size',str(embedding),'--layers',str(layers),'--epochs','8','--batch-size','256']
        result=subprocess.run(command,cwd=root,timeout=2450)
        status=json.loads((root/'jobs'/name/'status.json').read_text())
        if result.returncode or status['status']!='COMPLETED':save('FAILED',candidate=name,job=status);raise RuntimeError('Student training did not complete')
        report=json.loads((root/'runs'/name/'training_report.json').read_text())
        reports.append({'run':name,**report});save('CANDIDATE_COMPLETE',candidate=name)
        met=report['dev_selected_threshold']['joint_score']>=.9
        if hidden==64:candidate=(32,32,1) if met else (128,64,1)
        elif hidden==32:
            if not met:break
            candidate=(16,16,1)
        elif hidden==128:
            if met:break
            candidate=(256,96,2)
        else:break
    passed=[r for r in reports if r['dev_selected_threshold']['joint_score']>=.9]
    save('TRAINING_SEARCH_COMPLETE',quality_candidates=[r['run'] for r in passed],holdout_evaluated=False)


if __name__=='__main__':main()
