"""Independent saved-trace census for the user-requested end-only denominators."""
from collections import Counter, defaultdict
from datetime import datetime,timezone
import bisect
import gzip
import hashlib
import json
from pathlib import Path

P=Path('/home/edabk/hoangbpm/turn_detect/outputs/sentence_mix_end_only_test_20261107_v1')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    report=json.loads((P/'report.json').read_text());assert report['complete']
    assert sha(P/'predictions.jsonl.gz')==report['predictions_sha256']
    analysis=json.loads((P/'end_only_analysis.json').read_text())
    source=json.load(gzip.open(P/'prepared_streams.json.gz','rt',encoding='utf-8'))
    predicted=defaultdict(list)
    with gzip.open(P/'predictions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['predicted_eot']:predicted[r['stream'],r['lookahead_words']].append(r['boundary'])
    checks=defaultdict(Counter);short=Counter();source_turns=Counter()
    for i,s in enumerate(source):
        for seg in s['segments']:
            a,b=seg['start'],seg['end']
            if not seg.get('punctuation_present',True):source_turns['no_punctuation']+=1;continue
            if a==b:source_turns['empty']+=1;continue
            f=[x for x in s['eots'] if a<x<=b]
            # Count internal period breaks plus the last span, independently
            # of whether the final span itself has a fullstop.
            spans=1+sum(x<b for x in f)
            source_turns['all_nonempty_punctuated']+=1
            if b-a<3:
                short['multi' if spans>=2 else 'single']+=1;continue
            for k in (1,2):
                cs=predicted[i,k];left=bisect.bisect_right(cs,a);right=bisect.bisect_right(cs,b-k)
                internal=cs[left:right]
                no_cut=left==right
                cohort='multi_sentence' if spans>=2 else 'single_sentence_excluded'
                refs={x for x in f if x<=b-k and x not in s['unknown']}
                matched=len(refs.intersection(internal))
                keys=[('ALL',k,cohort),('language:'+seg['language'],k,cohort)]
                if len(f)>=2:keys.append(('ALL',k,'at_least_two_completed_fullstops'))
                if spans>=2 and not s['known_train_source_overlap']:keys.append(('ALL',k,'multi_without_known_legacy_overlap'))
                for key in keys:
                    c=checks[key];c['turns']+=1;c['no_internal_model_cut']+=int(no_cut);c['has_internal_model_cut']+=int(not no_cut)
                    c['words']+=b-a;c['words_in_no_cut_turns']+=(b-a)*int(no_cut)
                    c['model_internal_cuts']+=len(internal);c['correct_fullstop_cuts']+=matched;c['other_cuts']+=len(internal)-matched
                    c['eligible_internal_fullstops']+=len(refs);c['missed_internal_fullstops']+=len(refs)-matched
                    c['has_cuts_but_no_correct_fullstop_cut']+=int(bool(internal) and not matched)
                    c['all_eligible_fullstops_missed']+=int(bool(refs) and not matched)
                    c['no_evaluable_fullstop']+=int(not refs)
    indexed={(r['group'],r['lookahead_words'],r['cohort']):r for r in analysis['rows']}
    for key,counts in checks.items():
        for name,n in counts.items():assert indexed[key][name]==n,(key,name,indexed[key][name],n)
    mt=json.loads((P/'translation_input_end_only_report.json').read_text())
    assert sha(P/'translation_input_segmentation.json')==mt['segmentation_sha256']
    segmentation=json.loads((P/'translation_input_segmentation.json').read_text())
    decisions=defaultdict(list)
    with gzip.open(P/'translation_input_decisions.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            d=json.loads(line)
            if d['cut']:decisions[d['message']].append(d['gap'])
    cc=defaultdict(Counter)
    for i,r in enumerate(segmentation):
        words=r['asr_source'].split();bounds=[0]+sorted(decisions[i])+[len(words)]
        chunks=[' '.join(words[a:b]) for a,b in zip(bounds,bounds[1:])]
        assert chunks==r['chunks'] and len(chunks)==len(decisions[i])+1
        assert r['end_only']==(len(chunks)==1)
        cohort='multi_sentence' if r['sentence_spans']>=2 else 'single_sentence_excluded'
        keys=[('ALL',cohort),('language:'+r['language'],cohort)]
        if r['completed_fullstops']>=2:keys.append(('ALL','at_least_two_completed_fullstops'))
        for key in keys:
            c=cc[key];c['messages']+=1;c['end_only']+=int(not decisions[i]);c['with_internal_cut']+=int(bool(decisions[i]))
            c['chunks']+=len(chunks);c['words']+=len(words);c['words_in_end_only']+=len(words)*int(not decisions[i])
    mtindexed={(r['group'],r['cohort']):r for r in mt['counts']}
    for key,counts in cc.items():
        for name,n in counts.items():assert mtindexed[key][name]==n,(key,name)
    mainp=P.parent/'multilingual_malbert_vi_en_sentence_mix_20261107_v7'
    cp=report['checkpoint'];assert sha(mainp/'runs/finetune/latest_resume.pt')==cp['full_checkpoint_sha256']
    launch=json.loads((mainp/'launcher.json').read_text());assert not Path('/proc',str(launch['pid'])).exists()
    assert json.loads((mainp/'status.json').read_text())['stage']=='STOPPED_WITH_CHECKPOINT'
    out=dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(P/'report.json'),
        independent_test_turn_census_verified=True,independent_translation_message_census_verified=True,
        turn_groups_checked=len(checks),messages_checked=len(segmentation),
        source_turns=dict(source_turns),turns_under_three_words_excluded_separately=dict(short),
        no_new_model_inference=True,trainer_still_paused=True,full_checkpoint_unchanged=True)
    (P/'independent_end_only_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
