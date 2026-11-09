"""Read-only structural/provenance audit. Does not run any model or relabel data."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from turn_v2.common import normalize,input_id

J=ROOT/'outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1'


def rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def main():
    frozen=ROOT/'data_v2/frozen';manifest=json.loads((frozen/'manifest.json').read_text(encoding='utf-8'))
    dev=list(rows(frozen/'dev.jsonl'));assert len(dev)==17833 and sha(frozen/'dev.jsonl')==manifest['split_sha256']['dev']
    assert sha(ROOT/'train.jsonl')==manifest['input_sha256']['train']
    origin_wanted={x['origin_row']:x for x in dev};origin_mismatches=[]
    with (ROOT/'train.jsonl').open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            if number not in origin_wanted:continue
            source=json.loads(line);target=origin_wanted[number]
            if source['label']!=target['label'] or normalize(source['text'])!=target['text'] or normalize(source.get('context',''))!=target.get('context',''):
                origin_mismatches.append(number)
    keys=defaultdict(list);groups=defaultdict(list)
    for x in dev:keys[input_id(x)].append(x);groups[x['group_id']].append(x)
    train_keys=set();train_groups=set();train_count=0
    for row in rows(frozen/'train.jsonl'):
        train_keys.add(input_id(row));train_groups.add(row['group_id']);train_count+=1
    prefix_matches=[];unmatched=[];brief=[];ambiguous_continuations=[]
    suspicious_endings={'của','với','để','vì','vào','từ','tại','trong','và','hoặc','nhưng','một','những','các','đang','sẽ','rất','là','bằng','cho','trên','dưới'}
    candidate_eot=[];tail_counts=Counter();label_lengths=defaultdict(Counter)
    for x in dev:
        words=x['text'].split();label_lengths[x['label']][len(words)]+=1
        if x['label']=='EOT':
            tail_counts[words[-1]]+=1
            if len(words)>=4 and words[-1] in suspicious_endings:candidate_eot.append(x)
            continue
        parents=[p for p in groups[x['group_id']] if p['label']=='EOT' and len(p['text'].split())>len(words) and p['text'].split()[:len(words)]==words]
        if parents:
            parent=min(parents,key=lambda p:(abs(p['origin_row']-x['origin_row']),len(p['text'])))
            prefix_matches.append(x)
            if len(words)<=4:brief.append({'prefix':x['text'],'label':x['label'],'parent_eot_text':parent['text'],'origin_row':x['origin_row']})
            future=parent['text'].split()[len(words):]
            if len(words)>=5 and future and future[0] in {'và','nhưng','hay','hoặc','rồi'}:
                ambiguous_continuations.append({'prefix':x['text'],'label':x['label'],'parent_eot_text':parent['text'],
                    'inferred_omitted_next_words_from_same_group':future[:2],'origin_row':x['origin_row'],
                    'interpretation':'Potentially complete shorter utterance; not adjudicated wrong. These are not independently recorded future words.'})
        else:unmatched.append(x)
    end_examples=[{k:x[k] for k in ('text','label','origin_row','group_id')} for x in sorted(candidate_eot,key=lambda x:(len(x['text'].split()),x['origin_row']))[:24]]
    brief=sorted(brief,key=lambda x:(len(x['prefix'].split()),x['origin_row']))
    # These are ambiguity flags, never automatic label corrections.
    out={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'READ_ONLY_LEGACY_DEV_STRUCTURE_AND_SOURCE_AUDIT_NO_MODEL_NO_RELABEL',
      'dev_sha256':sha(frozen/'dev.jsonl'),'raw_source_sha256':manifest['input_sha256']['train'],
      'samples':len(dev),'labels':dict(Counter(x['label'] for x in dev)),
      'all_origin_rows_match_original_source':not origin_mismatches,'origin_mismatch_count':len(origin_mismatches),
      'nonempty_context_rows':sum(bool(x.get('context','')) for x in dev),
      'rows_with_explicit_lookahead_metadata':sum('lookahead_words' in x for x in dev),
      'rows_with_source_id':sum(bool(x.get('source_id')) for x in dev),
      'rows_with_audio_or_conversation_provenance':sum(any(x.get(k) for k in ('audio','audio_path','session_id','conversation_id')) for x in dev),
      'right_slot_word_count_gt2':sum(len(x['text'].split())>2 for x in dev),
      'within_dev_duplicate_input_keys':sum(len(v)>1 for v in keys.values()),
      'within_dev_conflicting_input_labels':sum(len({x['label'] for x in v})>1 for v in keys.values()),
      'exact_input_overlap_with_old_train_keys':len(set(keys)&train_keys),
      'group_overlap_with_old_train':len(set(groups)&train_groups),'old_train_rows_scanned':train_count,
      'groups':len(groups),'group_ids_are_heuristic_not_verified_dialogue':True,
      'neot_matching_strict_prefix_of_same_group_eot':len(prefix_matches),'neot_without_same_group_prefix_parent':len(unmatched),
      'eot_suspicious_ending_candidates_not_adjudicated':len(candidate_eot),'eot_suspicious_ending_examples':end_examples,
      'short_neot_prefix_examples_not_adjudicated':brief[:32],
      'neot_prefix_before_conjunction_candidates':len(ambiguous_continuations),
      'ambiguous_continuation_examples_not_adjudicated':ambiguous_continuations[:12],
      'eot_contains_question_phrase_heuristic':sum(any(t in ' '+x['text']+' ' for t in (' gì ',' nào ',' ai ',' đâu ',' bao nhiêu ',' tại sao ',' vì sao ',' thế nào ')) for x in dev if x['label']=='EOT'),
      'label_word_length_histograms':{k:dict(sorted(v.items())) for k,v in label_lengths.items()},
      'eot_final_words_most_common':tail_counts.most_common(20),
      'no_estimated_true_label_noise_rate':True,'labels_not_changed':True,'holdout_not_opened_or_scored':True,
      'conclusions':{'verified_protocol_mismatch':True,'verified_data_corruption':bool(origin_mismatches),
        'missing_future_cannot_be_reconstructed_for_eot_from_this_file':True,
        'prefix_membership_does_not_prove_semantic_incompleteness':True,
        'provided_gold_is_a_copied_tag_not_human_annotation_proof':True}}
    (J/'legacy_protocol_data_audit_v1.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('eot_suspicious_ending_examples','short_neot_prefix_examples_not_adjudicated','label_word_length_histograms','eot_final_words_most_common')},ensure_ascii=True))


if __name__=='__main__':main()
