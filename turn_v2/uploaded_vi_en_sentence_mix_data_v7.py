"""Balanced synthetic TRAIN packs of one to ten source fullstop sentences."""
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
import shutil

import numpy as np

from .uploaded_vi_en_data_v5 import ROOT, LANGS, StreamOrder, initialize, counts, clean, gaps, key, sha
from .uploaded_vi_en_curriculum_data_v6 import make_order, check_order

PARENT=ROOT/'outputs/multilingual_malbert_vi_en_curriculum_20261106_v6'
OUT=ROOT/'outputs/multilingual_malbert_vi_en_sentence_mix_20261107_v7'
MODULES=['uploaded_vi_en_data_v4.py','uploaded_vi_en_data_v5.py','uploaded_vi_en_curriculum_data_v6.py',
         'uploaded_vi_en_sentence_mix_data_v7.py','uploaded_vi_en_sentence_mix_evaluation_v7.py',
         'multilingual_malbert_vi_en_sentence_mix_train_v7.py','malbert_turn_student_v1.py',
         'tiny_compression_v1/common.py']


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_name(p.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(p)


def status(stage,file='status.json',**values):
    raw=Path('/proc/self/stat').read_text()
    v=dict(stage=stage,pid=os.getpid(),start_ticks=raw[raw.rfind(')')+2:].split()[19],
           boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),at_utc=datetime.now(timezone.utc).isoformat(),
           active_languages=list(LANGS),**values)
    save(file,v);print(json.dumps(v),flush=True)


def reserve():
    mount=next(row.split() for row in Path('/proc/mounts').read_text().splitlines() if row.split()[1]=='/mnt/hdd500')
    assert mount[0]=='/dev/sda1' and 'rw' in mount[3].split(',')
    if min(shutil.disk_usage(OUT).free,shutil.disk_usage('/mnt/hdd500').free)<13*2**30:
        raise InterruptedError('Reserve guard, preserve checkpoints')


def write_gzip(name,value):
    p=OUT/name;t=p.with_name(p.name+'.tmp')
    with gzip.open(t,'wt',encoding='utf-8',compresslevel=5) as stream:
        json.dump(value,stream,ensure_ascii=False,separators=(',',':'))
    with t.open('rb') as stream:os.fsync(stream.fileno())
    t.replace(p)


def signature(words):
    return hashlib.sha256(' '.join(words).encode()).digest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);reserve()
    if (OUT/'dataset_manifest.json').exists() and (OUT/'independent_preparation_verification.json').exists():
        return json.loads((OUT/'dataset_manifest.json').read_text())
    status('PREPARING_BALANCED_SENTENCE_PACKS',file='preparation_status.json')
    parent=json.loads((PARENT/'dataset_manifest.json').read_text())
    assert sha(PARENT/'train_streams.json.gz')==parent['files']['train_streams.json.gz']
    originals=initialize(json.load(gzip.open(PARENT/'train_streams.json.gz','rt',encoding='utf-8')))
    held=json.load(gzip.open(ROOT/'outputs/malbert_eval_zip_20261104_v2/prepared_streams.json.gz','rt',encoding='utf-8'))
    protected=set();shingles=set()
    for source in held:
        assert source['split'] in ('dev','test')
        for segment in source['segments']:
            a,b=segment['start'],segment['end'];words=source['words'][a:b]
            if len(words)>=8:protected.add(signature(words))
            for offset in range(max(0,len(words)-31)):
                shingles.add(signature(words[offset:offset+32]))
            start=a
            for end in sorted(e for e in source['eots'] if a<e<=b):
                if end-start>=8:protected.add(signature(source['words'][start:end]))
                start=end
    del held
    def collision(words):
        return (len(words)>=8 and signature(words) in protected) or any(signature(words[i:i+32]) in shingles for i in range(max(0,len(words)-31)))
    units=[];dedup={};audit=Counter()
    for index,source in enumerate(originals):
        assert source['source_split']=='train';stops=sorted(source['_eots'])
        for j,segment in enumerate(source['segments']):
            a,b=segment['start'],segment['end'];start=a
            for end in (e for e in stops if a<e<=b):
                words=source['words'][start:end];audit['source_complete_sentence_occurrences']+=1
                if collision(words):
                    audit['held_source_sentence_occurrences_excluded']+=1;start=end;continue
                digest=(segment['language'],signature(words))
                reference=dict(stream=index,segment=j,start=start,end=end)
                unknown={e-start for e in source['_unknown'] if start<e<end}
                if digest in dedup:
                    unit=units[dedup[digest]];unit['unknown']=sorted(set(unit['unknown'])|unknown);unit['references'].append(reference)
                    audit['exact_sentence_occurrences_deduplicated']+=1
                else:
                    dedup[digest]=len(units)
                    units.append(dict(words=words,unknown=sorted(unknown),language=segment['language'],
                                      origin=source['domain'],references=[reference]))
                start=end
            if start<b:
                audit['non_fullstop_tail_turns_retained_in_parent']+=1
                audit['non_fullstop_tail_words_retained_in_parent']+=b-start
    assert units
    buckets=defaultdict(list)
    for index,unit in enumerate(units):buckets[(unit['origin'],unit['language'])].append(index)
    rng=np.random.default_rng(20261107);packs=[];used=Counter();seen_packs=set();histogram=Counter();block_members=defaultdict(list)
    for (origin,language),indices in sorted(buckets.items()):
        blocks=math.ceil(len(indices)/85)
        singles=[int(i) for i in rng.permutation([i for i in indices if len(units[i]['words'])>=3])]
        assert len(singles)>=3*blocks,'Insufficient unique nontrivial single sentences'
        singles=singles[:3*blocks];reserved=set(singles)
        remaining=[int(i) for i in rng.permutation([i for i in indices if i not in reserved])]
        assert remaining
        single_cursor=0;cursor=0
        for block in range(blocks):
            sizes=[1,2,3,4,5]*3+[6,7,8,9,10]
            family=f'packed:{origin}:{language}:{block}'
            for number,size in enumerate(sizes):
                if size==1:
                    chosen=[singles[single_cursor]];single_cursor+=1
                else:
                    chosen=[remaining[(cursor+j)%len(remaining)] for j in range(size)];cursor+=size
                for attempt in range(100):
                    if attempt:chosen=[chosen[int(i)] for i in rng.permutation(len(chosen))]
                    words=[word for i in chosen for word in units[i]['words']]
                    identity=(language,signature(words))
                    if identity not in seen_packs and not collision(words):break
                else:raise ValueError('Could not form a unique held-source-disjoint pack')
                seen_packs.add(identity);ends=[];unknown=[];offset=0
                for i in chosen:
                    unit=units[i];unknown.extend(offset+e for e in unit['unknown']);offset+=len(unit['words']);ends.append(offset);used[i]+=1
                source=dict(key=f'{family}:pack{number}',family=family,domain=origin,source_split='train',
                    words=words,segments=[dict(start=0,end=len(words),language=language)],eots=ends,unknown=unknown,
                    conflict_mask={'1':[],'2':[]},source_rows=0,source_unit_count=size,sentence_count=size,
                    unit_ids=chosen,unit_ends=ends,synthetic_train_pack=True,
                    joins_may_cross_original_turns_and_speakers=True)
                block_members[family].append(len(packs));packs.append(source);histogram[language+':'+str(size)]+=1
    assert set(used)==set(range(len(units)))
    status('FILTERING_NEW_PACK_CONTEXT_CONFLICTS',file='preparation_status.json',unique_sentences=len(units),packs=len(packs))
    initialize(packs);positive=set();conflicts=set()
    for source in packs:
        for b,k,y,lang in gaps(source):
            if y:positive.add(key(source,b,k))
    for source in packs:
        for b,k,y,lang in gaps(source):
            if not y and key(source,b,k) in positive:conflicts.add(key(source,b,k))
    del positive
    masked=Counter();join_points=Counter()
    for source in packs:
        for b,k,y,lang in gaps(source):
            if key(source,b,k) in conflicts:
                source['conflict_mask'][str(k)].append(b);masked[lang+':'+str(y)]+=1
            elif y and b<len(source['words']):join_points[lang+':k'+str(k)]+=1
    # Validate every source unit and every concatenation, without trusting the packing counters.
    actual_histogram=Counter();actual_used=Counter()
    for source in packs:
        assert len(source['unit_ids'])==source['sentence_count']==len(source['eots'])
        reconstructed=[];offset=0
        for i,end in zip(source['unit_ids'],source['unit_ends']):
            unit=units[i];assert unit['language']==source['segments'][0]['language'];reconstructed.extend(unit['words']);offset+=len(unit['words']);assert offset==end
            for reference in unit['references']:
                original=originals[reference['stream']]
                assert original['words'][reference['start']:reference['end']]==unit['words']
                assert reference['end'] in original['_eots']
            actual_used[i]+=1
        assert reconstructed==source['words'] and not collision(source['words'])
        actual_histogram[source['segments'][0]['language']+':'+str(source['sentence_count'])]+=1
    assert actual_used==used and actual_histogram==histogram
    for language in LANGS:
        n=sum(v for k,v in histogram.items() if k.startswith(language+':'))
        for count in range(1,11):assert histogram[language+':'+str(count)]*100==n*(15 if count<=5 else 5)
    write_gzip('sentence_units.json.gz',units);write_gzip('train_streams.json.gz',clean(packs))
    for name in ['dev_full.json.gz','dev_calibration.json.gz','dev_monitor.json.gz','dev_diagnostic.json.gz','teacher_new_dev.json']:
        if not (OUT/name).exists():os.link(PARENT/name,OUT/name)
        assert sha(OUT/name)==sha(PARENT/name)
    stages=[]
    families=sorted(block_members,key=lambda f:hashlib.sha256(f.encode()).digest())
    strata=defaultdict(list)
    for f in families:
        first=packs[block_members[f][0]];strata[(first['domain'],first['segments'][0]['language'])].append(f)
    for index,fraction in enumerate([.1,.25,.5,1.]):
        selected=[f for members in strata.values() for f in members[:max(1,math.ceil(len(members)*fraction))]]
        indices=sorted(i for f in selected for i in block_members[f]);c=counts([packs[i] for i in indices])
        stages.append(dict(index=index,fraction_of_families=fraction,family_count=len(selected),stream_indices=indices,
                           update_budget=[10000,20000,30000,None][index],train=c,
                           eligible_points_per_pass=sum(v for k,v in c.items() if ':k' in k)))
    plan=dict(version=7,stages=stages,initial_stage=3,full_stage_minimum_epochs=6,
              schedule='Continue from the parent full stage on all balanced packs; earlier nested pools retained as metadata',
              sentence_count_distribution={str(i):(.15 if i<=5 else .05) for i in range(1,11)},
              distribution_basis='Pack count per language, not boundary points or tokens',whole_families=True,
              family_semantics='Synthetic balanced block of twenty packs, original source families recorded in unit references')
    save('curriculum_plan.json',plan)
    manifest=dict(parent);manifest.update(version=7,parent_manifest_sha256=sha(PARENT/'dataset_manifest.json'),
        train=stages[3]['train'],original_uploaded_source_rows=parent.get('original_uploaded_source_rows',parent['train']['source_rows']),
        source_rows_semantics='Zero direct raw rows in synthetic packs; exact source unit references in sentence_units.json.gz',
        files={name:sha(OUT/name) for name in parent['files']},curriculum_plan_sha256=sha(OUT/'curriculum_plan.json'),
        sentence_units_sha256=sha(OUT/'sentence_units.json.gz'),code_sha256={name:sha(ROOT/'turn_v2'/name) for name in MODULES},
        training_future_contract='One or two future words within the synthetic TRAIN pack, including across its original-source sentence joins',
        dev_future_contract='Original actual-turn future contract, byte-identical fixed DEV',
        label_source='Existing source fullstop silver labels, no added semantic judgments',
        all_eligible_uploaded_train_gaps_in_every_epoch=False,all_eligible_uploaded_train_gaps_in_full_stage_epoch=True)
    save('dataset_manifest.json',manifest)
    audit=dict(complete=True,original_source_statistics=dict(audit),unique_complete_sentence_units=len(units),
        fullstop_sentence_occurrences_represented=sum(len(u['references']) for u in units),
        all_eligible_unique_units_used=True,total_unit_uses=sum(used.values()),extra_unit_uses_for_final_balanced_blocks=sum(used.values())-len(units),
        pack_count=len(packs),histogram=dict(histogram),exact_global_pack_duplicates=0,
        exact_held_turn_sentence_min8_and_32word_shingle_checks_passed=True,
        conflicts=len(conflicts),masked_points=dict(masked),eligible_internal_eot_points=dict(join_points),
        all_pack_words_and_fullstop_labels_verified_against_original_units=True,
        synthetic_cross_turn_joins_explicit=True,last_pack_endpoint_without_future_still_omitted=True,
        test_metrics_used=False,not_human_audio_gold=True,
        limitations='Short common held strings below eight words, WordPiece-only and paraphrase collisions are not exhausted. Synthetic joins do not establish natural same-speaker continuity.')
    save('sentence_pack_audit.json',audit)
    save('independent_preparation_verification.json',dict(complete=True,manifest_sha256=sha(OUT/'dataset_manifest.json'),
        curriculum_plan_sha256=sha(OUT/'curriculum_plan.json'),pack_count=len(packs),unique_sentence_units=len(units),
        balanced_histogram_verified=True,all_source_provenance_words_labels_and_pack_joins_verified=True,
        protected_source_checks_passed=True,dev_bytes_identical=True,initial_stage=3))
    status('BALANCED_SENTENCE_PACKS_READY',file='preparation_status.json',packs=len(packs),unique_units=len(units),
           histogram=dict(histogram),eligible_points=stages[3]['eligible_points_per_pass'])
    return manifest


if __name__=='__main__':prepare()
