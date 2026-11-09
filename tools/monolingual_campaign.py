"""Two independent teacher/student/QAT jobs; Vietnamese remains its own model."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/content/turn_detect_v2');out=root/'outputs/monolingual_20261025';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from turn_v2.common import file_hash,write_json
    started=time.monotonic();receipts=[]
    def status(stage,**kwargs):write_json(out/'progress.json',{'stage':stage,'receipts':receipts,'elapsed_seconds':time.monotonic()-started,**kwargs})
    # A new Colab VM may lack the already-tested packed-ONNX dependencies.
    versions=json.loads((out/'dependency_versions.json').read_text())
    packages=[name+'=='+version for name,version in versions.items()]
    status('INSTALLING_EXPORT_DEPENDENCIES')
    with (out/'dependencies.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pip','install',*packages],stdout=log,stderr=subprocess.STDOUT,timeout=600)
    assert result.returncode==0,'Inspect dependencies.log'
    for language in ['en','zh']:
        data='data_v9/monolingual_20261025/'+language;teacher='runs/mono_'+language+'_teacher_20261025';student='runs/mono_'+language+'_student_20261025'
        for phase,module,extra in [('teacher','monolingual_teacher',['--output',teacher,'--steps','400']),
            ('student','monolingual_student',['--teacher',teacher,'--output',student])]:
            name='mono_'+language+'_'+phase+'_20261025';status('RUNNING',language=language,phase=phase,job=name)
            command=[sys.executable,'tools/distill_job.py','--job','jobs/'+name,'--timeout','2400','--',sys.executable,'-u','-m','turn_v2.'+module,'--language',language,'--data',data,*extra]
            result=subprocess.run(command,cwd=root,timeout=2450)
            job=json.loads((root/'jobs'/name/'status.json').read_text());receipts.append({'job':name,'status':job['status'],'exit_code':job.get('exit_code')})
            if result.returncode or job['status']!='COMPLETED':status('FAILED',language=language,phase=phase);raise RuntimeError('Inspect '+name+' logs; do not restart completed languages')
        report=json.loads((root/student/'monolingual_report.json').read_text())
        write_json(out/(language+'_result.json'),report)
    status('INDEPENDENT_EN_ZH_TRAINING_COMPLETE',quality_target_certified=False,held_evaluated=False)


if __name__=='__main__':main()
