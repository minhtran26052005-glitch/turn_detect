"""Preserve a CPU-only cache prefix, then enable overlapping CPU/GPU work.

Refuses an active model-training child. The old CPU writer is frozen while its
published prefix is flushed/hashed, then retired only after adoption succeeds.
No kernel interrupt/reset, optimizer restart, test scoring or data relabeling.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .translation_segmenter_v15c import atomic_json
from . import translation_segmenter_cache_v1 as cached

ROOT=cached.data.ROOT
JOB=cached.data.JOB
CONTROL=JOB/'cache_handoff_v1'


def state(stage,**extra):
    atomic_json(CONTROL/'status.json',{'stage':stage,'at_utc':cached.data.now(),**extra})


def live(a):
    try:
        p=Path('/proc')/str(a['pid'])
        return (p/'stat').read_text().split()[21]==a['start_ticks'] and [x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]==a['command']
    except (OSError,KeyError):return False


def spawn(module,extra,log):
    command=[sys.executable,'-u','-m',module]+extra
    with log.open('a') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    return {'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],
      'command':command,'module':module,'log':str(log),
      'source_sha256':hashlib.sha256((Path('turn_v2')/(module.split('.')[-1]+'.py')).read_bytes()).hexdigest()}


def main():
    config=json.loads((CONTROL/'config.json').read_text())
    parent=json.loads((ROOT/'launcher.json').read_text());child=json.loads((ROOT/'child.json').read_text())
    assert live(parent) and live(child)
    assert parent['module']=='turn_v2.translation_segmenter_full_campaign_v7'
    assert child['module']=='turn_v2.translation_segmenter_full_train_v4'
    assert child['command'][child['command'].index('--language')+1]=='zh'
    progress=json.loads((JOB/'training_status.json').read_text())
    assert progress['pid']==child['pid'] and progress['stage']=='LOADING_FULL_EPOCH_DATA_CACHE'
    m=json.loads((JOB/'current_dataset.json').read_text());assert m['complete']
    folder=JOB/'datasets'/m['generation'];verification=json.loads((folder/'verification_receipt.json').read_text())
    assert verification['complete'] and verification['database_sha256']==m['database_sha256']
    before={lang:{'stat':tuple((ROOT/'teachers'/lang/'latest_resume.pt').stat()[i] for i in (1,6,8)),
      'receipt':json.loads((ROOT/'teachers'/lang/'resume_receipt.json').read_text())} for lang in cached.data.LANGUAGES}
    assert before['zh']['receipt']['total_steps']==1400 and not before['zh']['receipt'].get('full_epoch_generation')
    assert all(before[l]['receipt']['full_epoch_cursor']==32000 for l in ('vi','en'))
    assert cached.ready(m,'vi') and cached.ready(m,'en') and not cached.ready(m,'zh')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in gpu.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if parts and parts[0]==str(child['pid']):assert int(parts[1])<64,'Active CUDA model must finish first'
    state('FREEZING_ONLY_OWN_CPU_CACHE_WRITER',child=child,parent=parent)
    os.kill(child['pid'],signal.SIGSTOP);retired=False
    try:
        for _ in range(20):
            if (Path('/proc')/str(child['pid'])/'stat').read_text().split()[2] in ('T','t'):break
            time.sleep(.1)
        assert (Path('/proc')/str(child['pid'])/'stat').read_text().split()[2] in ('T','t')
        current=json.loads((JOB/'data_status.json').read_text())
        assert current['stage']=='ENCODING_VERIFIED_DATASET_CACHE' and current['generation']==m['generation']
        assert current['language']=='zh' and current['split']=='train'
        # Old status publishes index i after finishing that row. Keep only
        # [0,i), conservatively discarding at most the last status interval.
        cursor=int(current['rows']);assert 0<cursor<m['fingerprints']['zh']['train']['rows']
        state('HASHING_AND_ADOPTING_PUBLISHED_CPU_CACHE_PREFIX',rows=cursor)
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Cache adoption exceeded120s')))
        signal.alarm(120)
        proof=cached.adopt_prefix(m,'zh','train',cursor)
        signal.alarm(0)
        atomic_json(CONTROL/'cache_prefix_adoption.json',proof)
        state('PREFIX_DURABLE_RETIRING_CACHE_WRITER',preserved_rows=cursor)
        os.kill(parent['pid'],signal.SIGTERM)
        os.kill(child['pid'],signal.SIGKILL);retired=True
        for _ in range(40):
            if not live(parent) and not live(child):break
            time.sleep(.25)
        assert not live(parent) and not live(child),'Old coordinator has not exited'
        after={lang:{'stat':tuple((ROOT/'teachers'/lang/'latest_resume.pt').stat()[i] for i in (1,6,8)),
          'receipt':json.loads((ROOT/'teachers'/lang/'resume_receipt.json').read_text())} for lang in cached.data.LANGUAGES}
        assert before==after,'Cache transition touched a model checkpoint'
        error=JOB/'campaign_error.json'
        if error.exists():error.replace(CONTROL/'retired_coordinator_error.json')
        state('WAITING_FOR_BACKUP_PASS_BOUNDARY')
        backup=ROOT/'drive_backup_v1';b=json.loads((backup/'launcher.json').read_text())
        deadline=time.monotonic()+300
        while json.loads((backup/'status.json').read_text())['stage']!='SYNCED':
            assert live(b) and time.monotonic()<deadline,'Backup must finish before replacing its writer'
            time.sleep(2)
        assert live(b);os.kill(b['pid'],signal.SIGTERM)
        for _ in range(20):
            if not live(b):break
            time.sleep(.1)
        assert not live(b),'Do not duplicate Drive backup writers'
        for name,wanted in config['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==wanted
        prefetch=JOB/'prefetch_v1';prefetch.mkdir(exist_ok=True)
        cpu=spawn('turn_v2.translation_segmenter_cache_v1',['--generation',m['generation'],'--language','zh','--split','train'],prefetch/'zh_train.log')
        cpu.update(generation=m['generation'],language='zh',split='train',CPU_only=True)
        atomic_json(prefetch/'zh_train.json',cpu)
        coordinator=spawn('turn_v2.translation_segmenter_full_campaign_v8',[],JOB/'campaign_worker_v8.log')
        atomic_json(ROOT/'launcher.json',coordinator)
        writer=spawn('turn_v2.translation_segmenter_drive_backup_v7',[],backup/'worker_cache_handoff_v1.log')
        writer['drive_path']=b['drive_path'];atomic_json(backup/'launcher.json',writer)
        result={'complete':True,'preserved_zh_cache_rows':cursor,'remaining_zh_cache_rows':m['fingerprints']['zh']['train']['rows']-cursor,
          'all_three_checkpoint_files_and_receipts_unchanged':True,'VI_EN_resume_cursors':{l:before[l]['receipt']['full_epoch_cursor'] for l in ('vi','en')},
          'optimizer_or_sampler_reinitialization':False,'kernel_reset_or_interrupt':False,
          'prefetch':cpu,'coordinator':coordinator,'backup':writer,'effective_batch_unchanged':64,
          'source_sha256':config['source_sha256'],'data_generation':m['generation']}
        atomic_json(CONTROL/'receipt.json',result);state('COMPLETE_CPU_PREFETCH_AND_GPU_READY_LANGUAGE_SCHEDULING',**result)
    except BaseException:
        signal.alarm(0)
        if not retired and live(child):os.kill(child['pid'],signal.SIGCONT)
        raise


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic_json(CONTROL/'error.json',{'error':repr(exc),'traceback':traceback.format_exc(),'state_preserved':True})
        state('FAILED_PRESERVED',error=repr(exc));raise
