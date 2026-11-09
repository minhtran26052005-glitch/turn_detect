"""Build a bounded literature/evidence bundle from saved reports; never run models."""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime, timezone
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/turn_failure_literature_20261108'


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), {
        'path': relative, 'sha256_bytes': hashlib.sha256(raw).hexdigest(),
        'sha256_lf_normalized': hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
    }


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def slim(selected):
    return {
        'threshold': selected['threshold'],
        'overall': {lang: {k: x[k] for k in ('eot_f1', 'eot_precision', 'eot_recall', 'tp', 'fp', 'fn')}
                    for lang, x in selected['by_language'].items()},
        'domains': [{k: x[k] for k in ('domain', 'language', 'eot_f1', 'eot_precision', 'eot_recall', 'tp', 'fp', 'fn')}
                    for x in selected['by_domain']],
    }


def main():
    old, old_source = read('outputs/asr_only_adaptation_20261107_v1/dev_report.json')
    parent, parent_source = read('outputs/malbert_calibration_audit_20261107_v1/report.json')
    controlled, controlled_source = read('outputs/asr_effective_batch16_20261107_v1/report.json')
    assert all(x['complete'] for x in (old, parent, controlled))
    rows = [
        ('parent_v7_expanded_calibration', parent['selected'], parent_source),
        ('old_asr_old_grid', old['selected'], old_source),
        ('same_old_asr_expanded_calibration', controlled['arms']['control']['selected'], controlled_source),
        ('asr_effective16_expanded_calibration', controlled['arms']['effective16']['selected'], controlled_source),
    ]
    evidence = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Saved punctuation-silver DEV reports only; no new inference or training.',
        'units': 'F1, precision and recall are fractions in JSON, percent in README.',
        'rows': [{'model_policy': name, 'source': source, **slim(selected)} for name, selected, source in rows],
        'limitations': [
            'Threshold selected on calibration before monitor; arm thresholds differ.',
            'Predicted resets change subsequent input contexts.',
            'Effective16 also changes update count and dropout grouping; not normalization-only attribution.',
            'No independent human/audio gold or new translation quality measurement.',
            'Rejected ASR weights intentionally removed by user; reports and traces retained.',
        ],
        'proposed_experiment_launched': False,
    }
    write('local_evidence.json', evidence)
    md = (OUT / 'README.md').read_text(encoding='utf-8')
    sections = re.split(r'(?m)^### ', md)[1:]
    refs = []
    for section in sections:
        if not re.match(r'P\d+ ', section):
            continue
        heading = section.splitlines()[0]
        match = re.search(r'\[([^\]]+)\]\((https://[^)]+)\)', section)
        assert match, heading
        refs.append({'id': heading.split(' ')[0], 'citation': heading,
                     'title': match.group(1), 'primary_url': match.group(2),
                     'access_scope_and_limits': section.split('\n\n')[1]})
    assert len(refs) == 10
    write('references.json', {'review_type': 'targeted narrative review', 'references': refs})
    write('search_log.json', {
        'search_date': '2026-11-08', 'primary_sources_only': True,
        'queries': [
            'dynamic sentence boundary detection simultaneous translation lookahead',
            'online punctuation restoration ELECTRA small streaming ASR right context',
            'punctuation restoration noisy speech transcripts domain adaptation',
            'fine tuning out of distribution feature distortion calibration',
            'segmentation free streaming machine translation',
        ],
        'scope': 'Targeted source discovery and relevant methods/results review, not exhaustive systematic screening.',
        'included_primary_documents': len(refs),
        'excluded_or_limited': [
            {'url': 'https://aclanthology.org/2024.tacl-1.58/', 'reason': 'Different summarization paper; excluded.'},
            {'url': 'https://www.sciencedirect.com/science/article/pii/S0167639325000846',
             'reason': 'Publisher full text returned 403; no table-level quantitative claim used.'},
            {'url': 'https://openreview.net/forum?id=UYneFzXSJWh',
             'reason': 'Challenge blocked access; corresponding author arXiv paper used instead.'},
        ],
        'not_claimed': ['Exhaustive 2026 literature coverage', 'PRISMA counts', 'Meta-analysis', 'Exact replication'],
    })
    body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'sane_lists'])
    html = '''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nghiên cứu lỗi detector — 08/11/2026</title>
<style>body{font:17px/1.65 system-ui,sans-serif;color:#172235;background:#f3f5f8;margin:0}
main{max-width:1020px;margin:30px auto;padding:32px;background:white;border-radius:12px}
h1{font-size:30px;line-height:1.3}h2{margin-top:38px}h3{margin-top:28px}a{color:#175bb2}
table{border-collapse:collapse;width:100%;font-size:15px;display:block;overflow:auto}
th,td{border:1px solid #d3dae4;padding:8px 12px}th{background:#eaf0f8}
code{font-size:14px;overflow-wrap:anywhere}li{margin:10px 0}
@media(max-width:600px){main{margin:0;padding:20px;border-radius:0}body{font-size:16px}}
</style><main>''' + body + '</main></html>'
    (OUT / 'review.html').write_text(html, encoding='utf-8')
    receipt, _ = read('outputs/failed_checkpoint_cleanup_20261108/deletion_receipt.json')
    verification, _ = read('outputs/failed_checkpoint_cleanup_20261108/verification.json')
    assert receipt['complete'] and verification['main_full_checkpoint_sha_unchanged']
    assert len(receipt['remote']['removed']) == 77
    checks = {'complete': True, 'source_reports_complete': True, 'references': len(refs),
              'no_training_or_inference': True, 'cleanup_verified': True, 'files': []}
    for path in OUT.iterdir():
        if path.is_file() and path.name != 'bundle_verification.json':
            raw = path.read_bytes()
            checks['files'].append({'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
    assert sum(x['bytes'] for x in checks['files']) < 5 * 1024**2
    for name in ('local_evidence.json', 'references.json', 'search_log.json'):
        json.loads((OUT / name).read_text(encoding='utf-8'))
    write('bundle_verification.json', checks)
    print(json.dumps({'complete': True, 'files': len(checks['files']),
                      'bytes': sum(x['bytes'] for x in checks['files']),
                      'references': len(refs), 'no_new_training': True}))


if __name__ == '__main__':
    main()
