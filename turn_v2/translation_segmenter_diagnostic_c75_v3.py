"""Before first inference, tighten v2 to never infer removed/reserve overlaps."""
from pathlib import Path
import traceback
from . import translation_segmenter_diagnostic_c75_v2 as continuation
from . import translation_segmenter_clean_scoring_v1 as clean
from . import translation_segmenter_v15d_test_v2 as ev

JOB=continuation.JOB


if __name__=='__main__':
    try:
        ev.atomic_json(JOB/'clean_scoring_revision_v3.json',{'source_sha256':ev.file_sha(Path(__file__)),
            'clean_scoring_sha256':ev.file_sha(Path(clean.__file__)),
            'clean_scoring_source_utf8':Path(clean.__file__).read_text(encoding='utf-8'),
            'excluded_rows_never_inferred':True,'frozen_weights_data_thresholds_unchanged':True,
            'scored_languages_before_revision':[]})
        ev.score_language=clean.score_language
        continuation.run()
    except Exception as exc:
        ev.JOB=JOB
        ev.atomic_json(JOB/'continuation_v3_error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        # Existing control/finalizer waiters subscribe to this compatibility path.
        if not (JOB/'continuation_v2_error.json').exists():
            ev.atomic_json(JOB/'continuation_v2_error.json',{'error':repr(exc),'origin':'continuation_v3_error.json','compatibility_failure_signal':True})
        ev.status('FAILED_PRESERVED_V3',error=repr(exc));raise
