"""Continue unscored fit/source controls after the metadata-publisher repair."""
import gc
from pathlib import Path
import sqlite3
import time
import traceback

from . import translation_segmenter_diagnostic_c75_v1 as diag
from . import translation_segmenter_diagnostic_c75_v2 as publish
from . import translation_segmenter_diagnostic_analysis_v1 as fit
from . import translation_segmenter_parser_audit_v1 as parser
from . import translation_segmenter_v15d_test_v2 as ev

JOB=diag.JOB
CONTROL=JOB/'analysis_v2'


def state(stage,**fields):
    diag.disk_guard();ev.atomic_json(CONTROL/'status.json',{'at_utc':diag.now(),'stage':stage,**fields})


def run():
    import fcntl
    CONTROL.mkdir(exist_ok=True);fit.CONTROL=CONTROL;parser.CONTROL=CONTROL;ev.JOB=CONTROL
    with (CONTROL/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (JOB/'error_analysis_v1.json').exists():raise RuntimeError('Analysis already scored; preserve it')
        manifest=ev.read_json(JOB/'manifest.json')
        counts=ev.read_json(JOB/'training_coverage_v1.json')
        db=sqlite3.connect('file:'+str((JOB/'frozen/test.sqlite').resolve())+'?mode=ro',uri=True)
        eligible={lang+':'+str(page) for lang,page in db.execute('SELECT language,page_id FROM sources WHERE added_cycle<=75')}
        parser_report={'created_at_utc':diag.now(),'scope':'CPU_SOURCE_ALIGNMENT_OF_CLEAN_SCORED_DIAGNOSTIC_ONLY',
            'reserve_read_or_scored':False,'languages':{},'complete':False,
            'audit_source_sha256':ev.file_sha(Path(parser.__file__)),
            'corrected_extractor_source_sha256':ev.file_sha(Path('turn_v2/translation_segmenter_causal_extraction_v1.py'))}
        try:
            import jieba
            for lang in ev.LANGUAGES:
                while True:
                    partial=ev.read_json(JOB/'partial_report.json') if (JOB/'partial_report.json').exists() else {}
                    if lang in partial.get('languages',{}):break
                    if (JOB/'continuation_v2_error.json').exists():raise RuntimeError('Diagnostic continuation failed; preserve controls')
                    state('WAITING_FOR_DIAGNOSTIC_LANGUAGE',language=lang);time.sleep(15)
                trace=JOB/f'test_predictions_{lang}.jsonl.gz'
                if ev.file_sha(trace)!=partial['languages'][lang]['full_trace_sha256']:raise RuntimeError('Scored trace changed')
                parser_report['languages'][lang]=parser.audit_language(db,lang,trace,jieba)
                ev.atomic_json(CONTROL/'parser_partial_report.json',parser_report)
            parser_report.update(complete=True,completed_at_utc=diag.now())
            ev.atomic_json(JOB/'parser_audit_v1.json',parser_report)
            while not (JOB/'report.json').exists():
                if (JOB/'continuation_v2_error.json').exists():raise RuntimeError('Diagnostic continuation failed')
                state('WAITING_FOR_FINAL_DIAGNOSTIC_REPORT');time.sleep(10)
            import torch
            import transformers.utils as utils
            import transformers.utils.import_utils as imports
            utils.is_torchvision_available=imports.is_torchvision_available=lambda:False
            utils.is_librosa_available=imports.is_librosa_available=lambda:False
            torch.set_num_threads(1)
            report={'created_at_utc':diag.now(),'scope':'EXPLORATORY_ERROR_ANALYSIS_PLUS_FIXED_TRAIN_POOL_CONTROL',
                'diagnostic_report_sha256':ev.file_sha(JOB/'report.json'),'training_coverage':counts,
                'train_fit_controls':{},'quality_target_certified':False}
            for lang in ev.LANGUAGES:
                report['train_fit_controls'][lang]=fit.train_control(db,lang,eligible,manifest)
                ev.atomic_json(CONTROL/'fit_partial_report.json',report)
        finally:db.close()
        report.update(complete=True,completed_at_utc=diag.now());ev.atomic_json(JOB/'error_analysis_v1.json',report)
        names=[Path(__file__),Path(parser.__file__),Path(fit.__file__),Path('turn_v2/translation_segmenter_causal_extraction_v1.py')]
        ev.atomic_json(JOB/'analysis_source_receipts_v2.json',{p.name:{'sha256':ev.file_sha(p),'source_utf8':p.read_text(encoding='utf-8')} for p in names})
        publish.drive_publish(complete=True);state('COMPLETE',report='error_analysis_v1.json',source_audit_complete=True)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        CONTROL.mkdir(parents=True,exist_ok=True)
        ev.atomic_json(CONTROL/'error.json',{'at_utc':diag.now(),'error':repr(exc),'traceback':traceback.format_exc()})
        state('FAILED_PRESERVED',error=repr(exc));raise
