"""Acceptance rules for teacher-first training; automatic labels are not gold."""
from collections import Counter
from .common import digest, input_id

LANGUAGES = ('vi', 'en', 'zh')


def source_identity(row):
    return row.get('source_id') or row.get('group_id') or 'legacy:' + input_id(row)


def independent_qualification(report, checkpoint_sha256):
    reasons = []
    if not report:
        return {'passed': False, 'reasons': ['INDEPENDENT_EOT_GOLD_NOT_AVAILABLE']}
    required = {'label_scope': 'SEMANTIC_EOT', 'label_origin': 'INDEPENDENT_HUMAN_OR_PUBLISHER_EOT_GOLD',
                'split': 'SEALED_UNSEEN_TEST', 'used_for_training_or_selection': False,
                'checkpoint_sha256': checkpoint_sha256}
    for key, expected in required.items():
        if report.get(key) != expected:
            reasons.append('INVALID_' + key.upper())
    metrics = report.get('metrics', {})
    if metrics.get('samples', 0) < 200:
        reasons.append('INSUFFICIENT_INDEPENDENT_TEST_CASES')
    if min(report.get('label_counts', {}).get(k, 0) for k in ['EOT', 'NEOT']) < 40:
        reasons.append('INSUFFICIENT_CLASS_COVERAGE')
    if min(metrics.get('accuracy', 0), metrics.get('eot_f1', 0)) < .9:
        reasons.append('INDEPENDENT_ACCURACY_OR_F1_BELOW_90')
    if not report.get('predictions_sha256') or not report.get('dataset_sha256'):
        reasons.append('MISSING_INDEPENDENT_EVIDENCE_FINGERPRINTS')
    return {'passed': not reasons, 'reasons': reasons}


def distillation_gate(champions, qualifications=None):
    qualifications = qualifications or {}
    languages = {}
    for language in LANGUAGES:
        champion = champions.get(language, {})
        dev = champion.get('semantic_dev', {})
        qualification = independent_qualification(qualifications.get(language), champion.get('weights_sha256'))
        silver_pass = min(dev.get('accuracy', 0), dev.get('eot_f1', 0)) >= .9
        languages[language] = {'teacher_silver_dev_pass': silver_pass,
                               'independent_qualification': qualification,
                               'distillation_allowed': silver_pass and qualification['passed']}
    return {'distillation_allowed': all(x['distillation_allowed'] for x in languages.values()),
            'languages': languages, 'quality_target_certified': all(x['independent_qualification']['passed'] for x in languages.values()),
            'rule': 'All three large teachers first; silver90 alone never enables distillation'}


def training_fingerprint(rows):
    return digest('\n'.join(sorted(input_id(r) + ':' + r['label'] for r in rows)))


def require_train_classes(rows):
    counts = Counter(row['label'] for row in rows)
    if min(counts.get(label, 0) for label in ['NEOT', 'EOT']) < 5:
        raise ValueError('Need at least five accepted examples per reward class')
    return dict(counts)
