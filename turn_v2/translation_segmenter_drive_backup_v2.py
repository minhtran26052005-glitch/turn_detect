"""Extend existing Drive backup with immutable full-epoch data and sampler state.

Uses the same destination identity, manifest and process lock as v1. The
coordinator replaces v1 only after a SYNCED pass; it never runs both writers.
"""
from pathlib import Path
import json
import time
import traceback
from . import translation_segmenter_drive_backup_v1 as old


class Backup(old.Backup):
    def save_dataset(self,generation):
        folder=old.ROOT/'full_epoch_v1/datasets'/generation
        manifest=old.read_json(folder/'manifest.json')
        assert manifest['complete'] and manifest['generation']==generation
        prefix='full_epoch_v1/datasets/'+generation+'/'
        self.put(prefix+'dataset.zip',folder/'dataset.zip',expected=manifest['archive_sha256'],source_receipt=manifest)
        self.put(prefix+'manifest.json',folder/'manifest.json',expected=old.digest_file(folder/'manifest.json'))

    def save_teacher(self,language):
        folder=old.ROOT/'teachers'/language
        p=folder/'resume_receipt.json'
        if p.exists():
            r=old.read_json(p)
            if r.get('full_epoch_generation'):self.save_dataset(r['full_epoch_generation'])
        archived=folder/'previous_recipe_v15d'
        if archived.exists():
            for p in sorted(archived.rglob('*')):
                if p.is_file():self.put('teachers/'+language+'/previous_recipe_v15d/'+str(p.relative_to(archived)),p)
        super().save_teacher(language)

    def save_reports(self):
        p=old.ROOT/'full_epoch_v1/current_dataset.json'
        if p.exists():
            m=old.read_json(p);self.save_dataset(m['generation'])
            self.put('full_epoch_v1/current_dataset.json',p)
        super().save_reports()

    def manifest(self,complete):
        result=super().manifest(complete)
        result['full_epoch_recipe_supported']=True
        result['required_restore_for_full_epoch']='turn_v2.translation_segmenter_full_campaign_v1'
        for language in ('vi','en','zh'):
            receipt=result['files'].get('teachers/'+language+'/latest_resume.pt',{}).get('source_receipt',{})
            generation=receipt.get('full_epoch_generation')
            if generation:
                prefix='full_epoch_v1/datasets/'+generation+'/'
                if any(prefix+n not in result['files'] for n in ('dataset.zip','manifest.json')):
                    result['full_recovery_available']=False
                    result['missing_recovery_files'].append(prefix+'dataset.zip')
        return result


def main():
    import fcntl
    old.JOB.mkdir(parents=True,exist_ok=True)
    with (old.JOB/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=old.read_json(old.JOB/'config.json');backup=Backup(cfg);backup.load_manifest()
        while True:
            try:
                backup.sync();time.sleep(cfg['interval_seconds'])
            except Exception as exc:
                old.atomic_json(old.JOB/'latest_error.json',{'at_utc':old.now(),'error':repr(exc),'traceback':traceback.format_exc(),'retry_after_seconds':60})
                backup.state('RETRYING_AFTER_ERROR',error=repr(exc),retry_after_seconds=60);time.sleep(60)


if __name__=='__main__':main()
