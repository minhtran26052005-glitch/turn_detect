"""Read saved DEV/TEST text structure only; never load a model or score it."""
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path('/home/edabk/hoangbpm/turn_detect/outputs')
JOB = ROOT / 'multilingual_malbert_vi_en_sentence_mix_20261107_v7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(streams):
    groups = defaultdict(lambda: dict(hist=Counter(), closed_hist=Counter(), words=[],
                                     streams=set(), eot=0, eligible_eot_k2=0,
                                     eligible_neot_k2=0, no_fullstop=0))
    examples = {}
    stream_hist = Counter()
    for si, s in enumerate(streams):
        eots = sorted(set(s['eots']))
        unknown = set(s['unknown'])
        total_units = 0
        for seg in s['segments']:
            a, b = seg['start'], seg['end']
            if not seg.get('punctuation_present', True) or b-a < 3:
                continue
            ends = eots[bisect.bisect_right(eots, a):bisect.bisect_right(eots, b)]
            # Fullstop-delimited spans, including one residual span if the turn
            # ends without an unambiguous fullstop. This is not clause parsing.
            units = len(ends) + int(not ends or ends[-1] < b)
            assert units >= 1
            total_units += units
            n_eot = sum(x in set(ends) for x in range(a+1, b-1) if x not in unknown)
            eligible = sum(x not in unknown for x in range(a+1, b-1))
            for key in ['ALL', 'language:'+seg['language'], 'domain:'+s['domain']+'|'+seg['language']]:
                g = groups[key]
                g['hist'][str(units)] += 1
                g['closed_hist'][str(len(ends))] += 1
                g['words'].append(b-a)
                g['streams'].add(si)
                g['eot'] += len(ends)
                g['eligible_eot_k2'] += n_eot
                g['eligible_neot_k2'] += eligible-n_eot
                g['no_fullstop'] += int(not ends)
            bucket = '1' if units == 1 else '2' if units == 2 else '3-5' if units <= 5 else '6+'
            ek = seg['language']+'|'+bucket
            if ek not in examples and 8 <= b-a <= 65 and s['domain'] in ('daily','factory') and len(ends)==units:
                text = seg.get('display')
                reconstructed = text is None
                if reconstructed:
                    text = ' '.join(w+('.' if i+1 in ends else '') for i,w in enumerate(s['words'][a:b],a))
                examples[ek] = dict(stream_key=s['key'], domain=s['domain'], units=units,
                                    fullstops=len(ends), words=b-a, text=text,
                                    reconstructed_from_normalized_words_and_fullstop_labels=reconstructed)
        if total_units:
            stream_hist[str(total_units)] += 1
    result = {}
    for key,g in groups.items():
        n = len(g['words'])
        hist = g['hist']
        buckets = {str(i):hist[str(i)] for i in range(1,6)}
        buckets['6-10'] = sum(v for k,v in hist.items() if 6<=int(k)<=10)
        buckets['>10'] = sum(v for k,v in hist.items() if int(k)>10)
        result[key] = dict(turns=n, streams=len(g['streams']), sentence_span_buckets=buckets,
                           sentence_span_percent={k:100*v/n for k,v in buckets.items()},
                           exact_sentence_span_histogram=dict(hist),
                           completed_fullstop_histogram=dict(g['closed_hist']),
                           turns_without_unambiguous_fullstop=g['no_fullstop'],
                           median_words=statistics.median(g['words']), max_words=max(g['words']),
                           fullstops=g['eot'], eligible_eot_k2=g['eligible_eot_k2'],
                           eligible_neot_k2=g['eligible_neot_k2'])
    return dict(groups=result, stream_total_sentence_span_histogram=dict(stream_hist), examples=examples)


def main():
    manifest=json.loads((JOB/'dataset_manifest.json').read_text())
    files = {'DEV_monitor':JOB/'dev_monitor.json.gz',
             'DEV_calibration':JOB/'dev_calibration.json.gz',
             'DEV_full':JOB/'dev_full.json.gz',
             'TEST_last_evaluated':ROOT/'malbert_uploaded_test_20261105_v1/prepared_streams.json.gz'}
    report = dict(complete=True, at_utc=datetime.now(timezone.utc).isoformat(),
                  definition='Nonempty actual source turn with punctuation and >=3 words (potential k2 input). Count unambiguous fullstops plus a nonempty residual span. Not grammatical simple clauses; questions/exclamations/abbreviations retain original unscored policy.',
                  model_inference_performed=False, data_or_training_changed=False, datasets={})
    for name,path in files.items():
        before=sha(path)
        if name.startswith('DEV_'):
            assert before==manifest['files'][path.name]
        else:
            expected=json.loads((path.parent/'input_manifest.json').read_text())['prepared_streams_sha256']
            assert before==expected
        streams=json.load(gzip.open(path,'rt',encoding='utf-8'))
        out=summarize(streams)
        assert sha(path)==before
        out.update(file=str(path),sha256=before,saved_streams=len(streams))
        if name=='TEST_last_evaluated':
            out['primary_domains_only']=summarize([s for s in streams if s['domain'] in ('asr','daily','factory','numeric/dialogue')])
        report['datasets'][name]=out
    # The monitor's scored population must reproduce the actual evaluator.
    latest=json.loads((JOB/'latest_dev.json').read_text())
    for lang in ('vi','en'):
        g=report['datasets']['DEV_monitor']['groups']['language:'+lang]
        assert g['eligible_eot_k2']==latest['dev']['by_language'][lang]['actual_eot']
        assert g['eligible_eot_k2']+g['eligible_neot_k2']==latest['dev']['by_language'][lang]['samples']
    test_report=json.loads((files['TEST_last_evaluated'].parent/'report.json').read_text())
    report['test_report_schema_keys']=list(test_report)
    path=JOB/'dev_test_sentence_structure_audit_20261107.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
