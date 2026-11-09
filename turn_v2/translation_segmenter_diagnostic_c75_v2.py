"""Technical continuation of v1 frozen inputs; never redo scored languages.

Fix: snapshot mutable metadata before changing the publisher's own status.
The v1 error, frozen manifest, checkpoint pins and all outputs are preserved.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as base
from . import translation_segmenter_v15d_test_v2 as ev

JOB=base.JOB


def pin_metadata(source, staging):
    """Atomic-replace writers cannot change these already captured bytes."""
    raw=Path(source).read_bytes()
    checksum=hashlib.sha256(raw).hexdigest()
    staging.mkdir(exist_ok=True)
    target=staging/(checksum+Path(source).suffix)
    if not target.exists():target.write_bytes(raw)
    if target.read_bytes()!=raw:raise RuntimeError('Pinned metadata digest collision or corruption')
    return target


def drive_publish(complete=False):
    config=ev.read_json(base.ROOT/'drive_backup_v1/config.json')
    root=Path(config['drive_path'])
    if not os.path.ismount('/content/drive') or ev.read_json(root/'destination_identity.json')!=config['identity']:
        raise RuntimeError('Mounted Drive identity unavailable')
    destination=root/JOB.name;destination.mkdir(exist_ok=True)
    objects=destination/'objects';objects.mkdir(exist_ok=True)
    entries=ev.read_json(JOB/'drive_manifest.json').get('files',{}) if (JOB/'drive_manifest.json').exists() else {}
    db_zip=JOB/'frozen_database.zip'
    if not db_zip.exists():raise RuntimeError('Preserved frozen database archive missing')
    paths=[p for p in (JOB/'frozen').rglob('*') if p.is_file() and p.suffix not in ('.sqlite','.pt')]
    paths += [db_zip,Path(__file__),Path(base.__file__),Path(ev.__file__)]
    paths += [p for p in JOB.iterdir() if p.is_file() and p.name not in ('drive_manifest.json','LOCK') and p.suffix in ('.json','.jsonl','.gz','.html','.md')]
    # Capture *all* mutable metadata before the first status() write below.
    pinned=[]
    for original in paths:
        relative=str(original.relative_to(JOB)) if original.is_relative_to(JOB) else 'source/'+original.name
        source=pin_metadata(original,JOB/'publisher_metadata_pins_v2') if original.suffix in ('.json','.jsonl','.md','.html') else original
        pinned.append((relative,source))
    for relative,source in pinned:
        checksum=ev.file_sha(source);target=objects/(checksum+'.bin')
        base.state('BACKING_UP_DIAGNOSTIC_DIRECT_TO_DRIVE_V2',logical_path=relative,bytes=source.stat().st_size)
        if not target.exists():
            temp=objects/(checksum+'.v2.uploading')
            with source.open('rb') as src,temp.open('wb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2);dst.flush();os.fsync(dst.fileno())
            if ev.file_sha(temp)!=checksum:raise RuntimeError('Drive SHA readback mismatch: '+relative)
            temp.replace(target)
        if target.stat().st_size!=source.stat().st_size or ev.file_sha(target)!=checksum:
            raise RuntimeError('Existing Drive object is corrupt: '+relative)
        entries[relative]={'sha256':checksum,'bytes':source.stat().st_size,'object':str(target.relative_to(destination)),'readback_verified':True}
    value={'at_utc':base.now(),'complete':complete,'destination_identity':config['identity'],'files':entries,
        'scope':'Frozen diagnostic weights/tokenizers/data/protocol, traces, reports; redundant optimizer pins excluded',
        'publisher':'immutable_metadata_pins_v2','reserve_scored':False}
    ev.atomic_json(JOB/'drive_manifest.json',value)
    raw=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode();temp=destination/'latest_manifest.v2.tmp'
    with temp.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    if temp.read_bytes()!=raw:raise RuntimeError('Drive manifest readback mismatch')
    temp.replace(destination/'latest_manifest.json')


def run():
    import fcntl
    ev.JOB=JOB
    with (JOB/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'report.json').exists():raise RuntimeError('Completed test must not be rescored')
        manifest=ev.read_json(JOB/'manifest.json')
        base.state('VERIFYING_PRESERVED_C75_FROZEN_INPUTS_V2')
        if ev.file_sha(JOB/'frozen/test.sqlite')!=manifest['database_sha256']:
            raise RuntimeError('Preserved frozen SQLite hash changed')
        for lang,record in manifest['checkpoints'].items():
            if record['checkpoint_cycle']!=75 or record['cycle_step']!=100:raise RuntimeError('Unexpected frozen cycle')
            for name,checksum in record['files_sha256'].items():
                if ev.file_sha(JOB/'frozen'/lang/name)!=checksum:raise RuntimeError('Preserved model file changed: '+lang+'/'+name)
        partial=ev.read_json(JOB/'partial_report.json') if (JOB/'partial_report.json').exists() else None
        done=partial.get('languages',{}) if partial else {}
        for lang in ev.LANGUAGES:
            trace=JOB/f'test_predictions_{lang}.jsonl.gz'
            if lang in done:
                if not trace.exists() or ev.file_sha(trace)!=done[lang]['full_trace_sha256']:raise RuntimeError('Completed scored evidence changed')
            elif trace.exists():raise RuntimeError('Uncommitted scored trace requires specific recovery; refusing to redo it')
        ev.atomic_json(JOB/'continuation_v2.json',{'at_utc':base.now(),'source_sha256':ev.file_sha(Path(__file__)),
            'original_error_sha256':ev.file_sha(JOB/'error.json'),'frozen_manifest_sha256':ev.file_sha(JOB/'manifest.json'),
            'verified_original_frozen_inputs':True,'completed_languages_not_rescored':list(done),
            'fix':'pin mutable metadata before publisher status changes'})
        drive_publish()
        import torch
        import transformers.utils as utils
        import transformers.utils.import_utils as imports
        utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
        utils.is_librosa_available=imports.is_librosa_available=lambda:False
        torch.set_num_threads(1)
        report=partial or {'created_at_utc':base.now(),'manifest':manifest,'complete':False,
            'scope':'FROZEN_C75_NEW_SOURCE_PROSE_DIAGNOSTIC_TEST_NOT_HUMAN_AUDIO_GOLD',
            'test_used_for_selection':False,'future_train_crawl_design_may_use_diagnostic_errors':True,
            'unscored_reserve_preserved':True,'quality_target_certified':False,'languages':{}}
        report['technical_continuation']='translation_segmenter_diagnostic_c75_v2; identical frozen protocol/weights/data'
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        try:
            for lang in ev.LANGUAGES:
                if lang in report['languages']:continue
                selected,excluded,partition=base.select_sources(db,lang)
                ev.audit_test=lambda handle,language,rows,encoded,tokenizer:base.diagnostic_audit(handle,language,rows,encoded,tokenizer,excluded)
                result=ev.score_language(base.SelectedTestDB(db,lang,selected),lang,manifest)
                result['partition']={k:v for k,v in partition.items() if k!='unscored_reserve_source_ids'}
                result['error_analysis']=base.error_summary(db,lang)
                report['languages'][lang]=result
                ev.atomic_json(JOB/'partial_report.json',report)
                base.enhance_review(db);drive_publish()
                del selected,excluded;gc.collect()
        finally:db.close()
        report.update(complete=True,completed_at_utc=base.now())
        ev.atomic_json(JOB/'report.json',report);drive_publish(complete=True)
        base.state('COMPLETE',report_sha256=ev.file_sha(JOB/'report.json'),reserve_scored=False,drive_backup_complete=True)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        ev.JOB=JOB
        ev.atomic_json(JOB/'continuation_v2_error.json',{'at_utc':base.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        ev.status('FAILED_PRESERVED_V2',error=repr(exc));raise
