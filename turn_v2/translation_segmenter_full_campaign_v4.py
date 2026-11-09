"""Continuous full-epoch vi/en/zh campaign, with frozen data generations.

One round covers every eligible training row once for each language. Languages
rotate in 500-step chunks. Several full rounds continue until DEV saturation;
the crawler keeps its saved cursor and adds data between completed rounds.
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

from . import translation_segmenter_full_data_v3 as data
from . import translation_segmenter_full_train_v4 as train
from .translation_segmenter_v15c import atomic_json,file_sha

ROOT=data.ROOT
JOB=data.JOB
STOP=False


def stop(*_):
    global STOP
    STOP=True


def stopping():
    return STOP or (ROOT/'STOP_REQUESTED').exists()


def live(record):
    try:
        p=Path('/proc')/str(record['pid'])
        return (p/'stat').read_text().split()[21]==record['start_ticks'] and [x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]==record['command']
    except (OSError,KeyError):return False


def state(stage,**fields):
    value={'campaign':'translation_segmenter_full_epoch_v4','status':'STOPPED_BY_REQUEST' if stage.startswith('STOPPED') else 'RUNNING','stage':stage,
      'heartbeat_utc':data.now(),'pid':os.getpid(),'qwen_judge_enabled':False,'quality_target_certified':False,**fields}
    atomic_json(JOB/'campaign_status.json',value);atomic_json(ROOT/'status.json',value)


def wait_for_backup(manifest):
    logical='full_epoch_v1/datasets/'+manifest['generation']+'/dataset.zip'
    while not stopping():
        p=ROOT/'drive_backup_v1/latest_verified_manifest.json'
        m=json.loads(p.read_text()) if p.exists() else {}
        entry=m.get('files',{}).get(logical,{})
        if m.get('pass_complete') and m.get('full_recovery_available') and entry.get('sha256')==manifest['archive_sha256']:
            atomic_json(JOB/'dataset_backup_ready.json',{'at_utc':data.now(),'generation':manifest['generation'],
              'dataset_archive_sha256':manifest['archive_sha256'],'committed_manifest_sha256':file_sha(p)})
            return
        state('WAITING_FOR_VERIFIED_FULL_EPOCH_DATA_ON_DRIVE',generation=manifest['generation'])
        time.sleep(10)


def run_chunk(language,manifest,epoch):
    command=[sys.executable,'-u','-m','turn_v2.translation_segmenter_full_train_v4',
      '--language',language,'--generation',manifest['generation'],'--epoch',str(epoch),'--budget',str(train.CHUNK_STEPS)]
    log=JOB/'runs'/manifest['generation']/language/f'epoch_{epoch:04d}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        a={'pid':p.pid,'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().split()[21],
           'command':command,'module':'turn_v2.translation_segmenter_full_train_v4','log':str(log)}
        atomic_json(ROOT/'child.json',a)
        while p.poll() is None:
            # User stop/transition is graceful at a saved chunk boundary.
            time.sleep(3)
    if p.returncode:raise RuntimeError(f'Full-epoch child failed {language}, rc={p.returncode}; preserve cursor/log')


def crawl_three_saved_cycles():
    from . import translation_segmenter_v15d3 as crawl
    corpus=crawl.Corpus();jieba=crawl.trainer.ensure_jieba();streams=crawl.HFStreams(corpus)
    try:
        for _ in range(3):
            if stopping():return
            cycle=int(corpus.meta_get('next_cycle','1'))
            state('CRAWLING_NEXT_DATA_FOR_FUTURE_GENERATION',cycle=cycle)
            stats={spec['key']:streams.take(spec,cycle,200,jieba) for spec in crawl.SOURCES}
            wiki=crawl.crawl_wikipedia(corpus,cycle,jieba);corpus.db.commit()
            crawl.write_summary(corpus,cycle,stats,wiki)
            corpus.meta_set('next_cycle',str(cycle+1));corpus.db.commit();crawl.make_data_snapshot(corpus,cycle)
    finally:corpus.close()


def run():
    import fcntl
    JOB.mkdir(parents=True,exist_ok=True)
    with (JOB/'CAMPAIGN_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not stopping():
            cp=JOB/'current_dataset.json'
            manifest=json.loads(cp.read_text()) if cp.exists() else None
            if manifest is None:
                state('WAITING_FOR_COMPLETE_THREE_LANGUAGE_DATA_AUDIT')
                status_path=JOB/'data_status.json'
                if status_path.exists() and json.loads(status_path.read_text()).get('stage')=='FAILED_PRESERVED':
                    raise RuntimeError('Current data audit failed; preserve versioned error/log and frozen source snapshot')
                time.sleep(10);continue
            assert manifest['complete'] and manifest['all_train_dev_sources_examined']
            wait_for_backup(manifest)
            while not stopping():
                histories={lang:train.history_for(manifest,lang) for lang in data.LANGUAGES}
                active=[lang for lang in data.LANGUAGES if not train.convergence(histories[lang])['converged']]
                if not active:
                    state('THREE_LANGUAGE_DEV_SATURATION_CRAWL_CONTINUES',generation=manifest['generation'])
                    crawl_three_saved_cycles()
                    manifest=data.build();wait_for_backup(manifest);continue
                # Finish the same global round for every active language first.
                epoch=min(len(histories[lang]) for lang in active)+1
                for language in data.LANGUAGES:
                    if language not in active or len(histories[language])>=epoch:continue
                    run_chunk(language,manifest,epoch)
                    if stopping():break
                if stopping():break
                round_done=all(len(train.history_for(manifest,lang))>=epoch for lang in active)
                if round_done:
                    atomic_json(JOB/'latest_round_receipt.json',{'at_utc':data.now(),'generation':manifest['generation'],
                      'language_epochs':{lang:epoch for lang in active},'all_active_languages_full_coverage':True,
                      'test_scored':False,'campaign_continues':True})
                    crawl_three_saved_cycles()
    state('STOPPED_BY_USER_WITH_CHECKPOINTS_PRESERVED')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:run()
    except Exception as exc:
        atomic_json(JOB/'campaign_error.json',{'at_utc':data.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('CAMPAIGN_ERROR_PRESERVED',error=repr(exc));raise
