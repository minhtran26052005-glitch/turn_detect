"""Canonical call IDs and causal, text-only history before any V5 training.

The provider's clip ID includes channel and ending type. Neither identifies a
whole call: use the authoritative sidecar call_id to prevent train/dev leakage.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

from .common import digest, file_hash, input_id, read_jsonl, write_json, write_jsonl
SCICOM='Scicom-intl/semantic-vad-eot'
SCICOM_REV='98d1445eab71e4084ccb838dbd87a205f6f2e293'


def canonical_rows(rows, metadata):
    kept=[];excluded=Counter()
    for original in rows:
        row=dict(original)
        row['precanonical_session_group']=row['session_group']
        if row['language']!='vi':
            meta=metadata[row['source_id']]
            # Actual other-party response, rather than a long silence alone.
            if meta['eot_kind']!='response':
                excluded[row['language']+':'+meta['eot_kind']]+=1
                continue
            assert meta['split'] in ['train','validation']
            row['session_group']='callhome_'+row['language']+':'+meta['call_id']
            row['speaker']=meta['speaker']
            row['publisher_official_split']=meta['split']
            row['split']='sealed' if meta['split']=='validation' else ('dev' if int(digest(row['session_group'])[:8],16)%100<15 else 'train')
            row['absolute_decision']=float(meta['call_start'])+row['score_timestamp']
            row['absolute_window_start']=float(meta['call_start'])+row['audio_prefix_start']
            row['eot_kind']=meta['eot_kind']
        else:
            row['speaker']=str(row['channel'])
            row['absolute_decision']=row['score_timestamp']
            row['absolute_window_start']=row['audio_prefix_start']
        kept.append(row)
    return kept,dict(excluded)


def add_causal_history(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['session_group']].append(row)
    for group in groups.values():
        for row in group:
            candidates=[past for past in group if past['text'].strip()
                and past['absolute_decision']<=row['absolute_window_start']-1e-4]
            candidates.sort(key=lambda past:past['absolute_decision'],reverse=True)
            selected=[];seen=set()
            for past in candidates:
                identity=(past['source_id'],past['speaker'])
                if identity in seen:continue
                seen.add(identity);selected.append(past)
                if len(selected)==2:break
            selected.reverse()
            row['context']=' '.join('speaker'+past['speaker']+': '+past['text'] for past in selected)
            row['context_observations']=[{'case_id':past['case_id'],'observed_through':past['absolute_decision']} for past in selected]
            row['context_rule']='last2 distinct earlier source/speaker ASR windows ending before current audio window starts; no labels used'
    return rows


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    metadata={};sidecars=[]
    for lang in ['en','zh']:
        filename=f'meta/dialogue/callhome_{lang}.parquet'
        path=Path(hf_hub_download(SCICOM,filename,repo_type='dataset',revision=SCICOM_REV,token=False))
        for row in pq.read_table(path,columns=['id','call_id','speaker','split','eot_kind','call_start']).to_pylist():metadata[row['id']]=row
        sidecars.append({'filename':filename,'revision':SCICOM_REV,'sha256':file_hash(path),'bytes':path.stat().st_size})
    rows=[]
    for name in ['train','dev','sealed']:rows.extend(read_jsonl(args.source/(name+'.jsonl')))
    rows,excluded=canonical_rows(rows,metadata)
    rows=add_causal_history(rows)
    protected=set()
    for row in read_jsonl(Path('data_v4/en_zh_natural_pause_20261025/cases.jsonl')):
        protected.add('callhome_'+row['language']+':'+metadata[row['source_id']]['call_id'])
    assert not protected.intersection({row['session_group'] for row in rows})
    split_groups={split:{r['session_group'] for r in rows if r['split']==split} for split in ['train','dev','sealed']}
    for left,right in [('train','dev'),('train','sealed'),('dev','sealed')]:assert not split_groups[left].intersection(split_groups[right])
    # Deterministic test-first exact input deduplication, independent of score.
    seen=set();files={};duplicates=Counter()
    for split in ['sealed','dev','train']:
        values=[]
        for row in sorted((r for r in rows if r['split']==split),key=lambda r:r['case_id']):
            identity=row['language']+input_id(row)
            if identity in seen:duplicates[split]+=1;continue
            seen.add(identity);values.append(row)
        write_jsonl(args.output/(split+'.jsonl'),values)
        counts=Counter(r['language']+':'+r['label'] for r in values)
        assert all(counts[lang+':'+label]>0 for lang in ['vi','en','zh'] for label in ['EOT','NEOT']),counts
        files[split]={'cases':len(values),'counts':dict(counts),'sha256':file_hash(args.output/(split+'.jsonl')),
            'calls':{lang:len({r['session_group'] for r in values if r['language']==lang}) for lang in ['vi','en','zh']},
            'with_text_history':sum(bool(r['context']) for r in values),'empty_text_cases_kept':sum(not r['text'] for r in values)}
    write_json(args.output/'manifest.json',{'status':'CANONICAL_PUBLIC_NATURAL_TEXT_READY_NOT_HUMAN_GOLD','files':files,
        'source_manifest_sha256':file_hash(args.source/'manifest.json'),'sidecars':sidecars,'source_sha256':file_hash(Path(__file__)),
        'excluded_non_response_cases':excluded,'exact_input_duplicates_removed':dict(duplicates),
        'context_labels_used':False,'context_uses_future_audio_or_text':False,
        'grouping':'Authoritative publisher call_id; both channels and ending types stay in one split',
        'prior_exposed_call_groups_excluded':sorted(protected),'old_v4_pilot_scores_used_for_selection':False})
    print(__import__('json').dumps({'stage':'CANONICAL_READY','files':files,'excluded':excluded}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('data_v5/natural_vi_expanded_20261025'))
    parser.add_argument('--output',type=Path,default=Path('data_v5/canonical_20261025'))
    run(parser.parse_args())
