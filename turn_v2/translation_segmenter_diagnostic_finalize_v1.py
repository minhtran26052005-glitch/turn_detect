"""Finish reports and Drive publication after all one-shot controls complete."""
import importlib.util
from pathlib import Path
import time
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_diagnostic_c75_v2 as publish
from . import translation_segmenter_v15d_test_v2 as ev

JOB=diag.JOB
CONTROL=JOB/'finalizer_v1'


def state(stage,**fields):
    ev.atomic_json(CONTROL/'status.json',{'at_utc':diag.now(),'stage':stage,**fields})


def run():
    import fcntl
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'finalization_receipt_v1.json').exists():raise RuntimeError('Finalized result already exists; preserve it')
        while True:
            for path in (JOB/'continuation_v2_error.json',JOB/'analysis_v2/error.json'):
                if path.exists():raise RuntimeError('An upstream stage failed; preserved at '+str(path))
            path=JOB/'analysis_v2/status.json'
            done=ev.read_json(path).get('stage')=='COMPLETE' if path.exists() else False
            if done and all((JOB/name).exists() for name in ('report.json','error_analysis_v1.json','parser_audit_v1.json','crawl_training_balance_v1.json')):break
            state('WAITING_FOR_SCORED_TEST_TRAIN_CONTROL_AND_SOURCE_AUDIT');time.sleep(20)
        ev.JOB=CONTROL
        state('RENDERING_COMPACT_ANALYSIS_AND_CRAWL_RECIPE_ON_COLAB')
        script=Path('/content/turn_detect_v2/colab/report_translation_segmenter_diagnostic.py')
        spec=importlib.util.spec_from_file_location('diagnostic_report_renderer_v1',script)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.JOB=JOB;module.main()
        files={name:ev.file_sha(JOB/name) for name in ('report.json','error_analysis_v1.json','parser_audit_v1.json','error_analysis.md','crawl_plan_v1.json','review.html')}
        ev.atomic_json(JOB/'finalization_receipt_v1.json',{'at_utc':diag.now(),'complete':True,'files_sha256':files,
            'report_renderer_sha256':ev.file_sha(script),'renderer_source_utf8':script.read_text(encoding='utf-8'),
            'scope':'One frozen cycle75 diagnostic plus fixed train-pool controls and source alignment; future TRAIN-only recipe not activated',
            'reserve_scored':False,'test_used_for_model_or_threshold_selection':False})
        publish.drive_publish(complete=True)
        manifest=ev.read_json(JOB/'drive_manifest.json')
        for name,checksum in files.items():
            if manifest['files'].get(name,{}).get('sha256')!=checksum:raise RuntimeError('Final Drive manifest missing '+name)
        state('COMPLETE',readback_verified=True,report='review.html',crawl_recipe='crawl_plan_v1.json')


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True)
        ev.atomic_json(CONTROL/'error.json',{'at_utc':diag.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('FAILED_PRESERVED',error=repr(exc));raise
