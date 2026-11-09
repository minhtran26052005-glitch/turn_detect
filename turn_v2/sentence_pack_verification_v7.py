"""Independent provenance, packing and conflict-mask verification before cutover."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json

from .uploaded_vi_en_sentence_mix_data_v7 import ROOT, OUT, PARENT, key


def run():
    assert not (OUT/'independent_pack_verification.json').exists()
    sources=json.load(gzip.open(PARENT/'train_streams.json.gz','rt',encoding='utf-8'))
    units=json.load(gzip.open(OUT/'sentence_units.json.gz','rt',encoding='utf-8'))
    packs=json.load(gzip.open(OUT/'train_streams.json.gz','rt',encoding='utf-8'))
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    audit=json.loads((OUT/'sentence_pack_audit.json').read_text())
    assert hashlib.sha256((OUT/'train_streams.json.gz').read_bytes()).hexdigest()==manifest['files']['train_streams.json.gz']
    for unit in units:
        unknown=set()
        for ref in unit['references']:
            source=sources[ref['stream']];segment=source['segments'][ref['segment']];a,b=ref['start'],ref['end']
            assert segment['start']<=a<b<=segment['end'] and (a==segment['start'] or a in source['eots']) and b in source['eots']
            assert source['words'][a:b]==unit['words'] and segment['language']==unit['language']
            assert not any(a<end<b for end in source['eots'])
            unknown.update(end-a for end in source['unknown'] if a<end<b)
        assert set(unit['unknown'])==unknown
    positive=set();negative=set();histogram=Counter();used=Counter();seen=set();joins=0;natural=0
    for source in packs:
        assert source['synthetic_train_pack'] and len(source['segments'])==1
        segment=source['segments'][0];assert segment['start']==0 and segment['end']==len(source['words'])
        words=[];ends=[];unknown=[];last=None
        for index in source['unit_ids']:
            unit=units[index];offset=len(words);words.extend(unit['words']);ends.append(len(words))
            unknown.extend(offset+x for x in unit['unknown']);used[index]+=1
            if last is not None:
                joins+=1;left=units[last]['references'][0];right=unit['references'][0]
                natural+=int(left['stream']==right['stream'] and left['segment']==right['segment'] and left['end']==right['start'])
            last=index
        assert words==source['words'] and ends==source['eots']==source['unit_ends'] and unknown==source['unknown']
        assert len(ends)==source['sentence_count']
        assert all(segment['language']==units[index]['language'] for index in source['unit_ids'])
        digest=(segment['language'],hashlib.sha256(' '.join(words).encode()).hexdigest())
        assert digest not in seen;seen.add(digest)
        histogram[segment['language']+':'+str(len(ends))]+=1
        eots=set(ends);unknown=set(unknown)
        for k in (1,2):
            for boundary in range(1,len(words)-k+1):
                if boundary not in unknown:
                    (positive if boundary in eots else negative).add(key(source,boundary,k))
    conflicts=positive&negative
    assert len(conflicts)==audit['conflicts']
    del positive,negative
    counts=Counter();masked=Counter()
    for source in packs:
        eots=set(source['eots']);unknown=set(source['unknown']);masks={k:set(v) for k,v in source['conflict_mask'].items()}
        language=source['segments'][0]['language']
        for k in (1,2):
            for boundary in range(1,len(source['words'])-k+1):
                if boundary in unknown:continue
                excluded=key(source,boundary,k) in conflicts
                assert (boundary in masks[str(k)])==excluded
                if excluded:masked[language+':'+str(int(boundary in eots))]+=1
                else:counts[language+':k'+str(k)+':'+('EOT' if boundary in eots else 'NEOT')]+=1
    assert dict(histogram)==audit['histogram'] and dict(masked)==audit['masked_points'] and len(used)==len(units)
    for name,value in counts.items():assert value==manifest['train'][name]
    for language in ('vi','en'):
        total=sum(value for name,value in histogram.items() if name.startswith(language+':'))
        for size in range(1,11):assert histogram[language+':'+str(size)]*100==total*(15 if size<=5 else 5)
    for name in ['dev_calibration.json.gz','dev_monitor.json.gz']:
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==hashlib.sha256((PARENT/name).read_bytes()).hexdigest()
    proof=dict(complete=True,all_units_source_words_fullstops_unknown_masks_verified=True,
        all_packs_reconstructed=True,all_pack_hashes_unique=True,histogram=dict(histogram),
        all_conflict_memberships_both_classes_recomputed=True,masked=dict(masked),
        all_eligible_label_counts_match_manifest=True,eligible_points=sum(counts.values()),
        eligible_eot_points=sum(value for name,value in counts.items() if name.endswith(':EOT')),
        internal_joins=joins,canonical_originally_adjacent_same_turn_joins=natural,
        synthetic_reordered_or_cross_source_joins=joins-natural,unique_units_used=len(used),
        total_unit_uses=sum(used.values()),final_endpoint_omission_fraction=len(packs)/sum(used.values()),dev_bytes_identical=True)
    (OUT/'independent_pack_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':run()
