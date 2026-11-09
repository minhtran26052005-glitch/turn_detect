"""Read ZIP members without extracting or modifying training/evaluation state."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_v2.common import normalize

ROOT=Path.home()/'Downloads'
OUT=Path('outputs/data_role_audit_20261104')
ARCHIVES={'asr_crawl':'asr_crawl-20261103T173849Z-1-001.zip','eval':'eval-20261103T173842Z-1-001.zip',
    'factory':'factory-20261103T173856Z-1-001.zip','daily_conversation':'daily_conversation-20261103T173852Z-1-001.zip'}
IMPORTED={'01a8df34923011d038a2a30cce60774a69f07794756f41f44e8924341e29c9f8':'factory_checked.jsonl',
    '96f999ac6feeaad29fadc23d0bda87f44d8ade337f088418e9fb6c6332e7c6e4':'public_checked.jsonl'}


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);entries={};manifests={};cohorts={};payload=[]
    for archive,name in ARCHIVES.items():
        with zipfile.ZipFile(ROOT/name) as z:
            for info in z.infolist():
                if info.is_dir():continue
                assert not info.filename.startswith(('/','\\')) and '..' not in Path(info.filename.replace('\\','/')).parts
                if info.filename.endswith('eval_set.json'):
                    manifests[info.filename]=json.loads(z.read(info));continue
                if not info.filename.endswith('labels_checked.jsonl'):continue
                counts={k:Counter() for k in ['split','src_lang','keep','run']};groups=defaultdict(list);rows=0;h=hashlib.sha256()
                with z.open(info) as f:
                    for line in f:
                        h.update(line)
                        if not line.strip():continue
                        r=json.loads(line);rows+=1
                        for k,c in counts.items():c[str(r.get(k,'unspecified'))]+=1
                        groups[r['dialogue_id']].append({k:r.get(k) for k in ['chunk_idx','turn_idx','chunk_in_turn','speaker','src_lang','source_raw','split','run','keep','starts_mid_sentence','ends_mid_sentence']})
                file_sha=h.hexdigest();records={};text_sigs=set()
                for identity,parts in groups.items():
                    parts.sort(key=lambda r:r['chunk_idx']);stream=[];turn_groups=defaultdict(list)
                    for r in parts:
                        words=normalize(r['source_raw']).split();stream.extend(r['src_lang']+':'+w for w in words)
                        if len(words)>=8:text_sigs.add(digest(r['src_lang']+'\0'+' '.join(words)))
                        key=(r.get('turn_idx') if r.get('turn_idx') is not None else r['chunk_idx'],r.get('speaker'),r['src_lang'])
                        turn_groups[key].append(r)
                    paragraphs=[]
                    for (turn,speaker,language),chunks in turn_groups.items():
                        chunks.sort(key=lambda r:r['chunk_in_turn'] if r.get('chunk_in_turn') is not None else r['chunk_idx'])
                        words=normalize(' '.join(r['source_raw'] for r in chunks)).split()
                        if len(words)>=8:paragraphs.append({'sha256':digest(language+'\0'+' '.join(words)),'language':language,'words':len(words),'turn_idx':turn})
                    splits={r['split'] or 'unspecified' for r in parts};runs={r['run'] or 'unspecified' for r in parts}
                    assert len(splits)==1 and len(runs)==1
                    records[identity]={'dialogue_sha256':digest(' '.join(stream)),'split':next(iter(splits)),'run':next(iter(runs)),
                        'family_id':re.sub(r'\.s\d+\.(?:vi|en)$','',identity),'paragraphs':paragraphs,'rows':len(parts)}
                entries[info.filename]={'archive':name,'sha256':file_sha,'bytes':info.file_size,'rows':rows,'dialogues':len(groups),
                    'split_rows':dict(counts['split']),'language_rows':dict(counts['src_lang']),'keep_rows':dict(counts['keep']),
                    'run_rows':dict(counts['run']),'identical_to_already_imported_raw':IMPORTED.get(file_sha)}
                cohorts[info.filename]={'records':records,'chunk_hashes':text_sigs}
                if archive=='eval':
                    payload.extend({'entry':info.filename,'dialogue_id':identity,**r} for identity,r in records.items())
    train_names=[n for n in entries if not n.startswith('eval/')];comparisons=[]
    for name,group in cohorts.items():
        if not name.startswith('eval/'):continue
        for other in train_names:
            a,b=group['records'],cohorts[other]['records'];other_hashes={r['dialogue_sha256'] for r in b.values()}
            matches=[identity for identity,r in a.items() if r['dialogue_sha256'] in other_hashes]
            shared_chunks=len(group['chunk_hashes']&cohorts[other]['chunk_hashes'])
            if matches or shared_chunks:
                comparisons.append({'eval_entry':name,'source_entry':other,'exact_full_dialogue_matches':len(matches),
                    'shared_normalized_source_chunks_at_least_8words':shared_chunks,'matched_dialogue_examples':matches[:5]})
    daily=cohorts['daily_conversation/full/labels_checked.jsonl']['records'];split_by_family=defaultdict(set)
    for r in daily.values():split_by_family[r['family_id']].add(r['split'])
    daily_eval=cohorts['eval/daily/labels_checked.jsonl']['records'];cross=Counter();conflicts=[]
    for identity,r in daily_eval.items():
        orig=split_by_family.get(r['family_id'],set());cross[str((r['split'],tuple(sorted(orig))))]+=1
        if 'train' in orig:conflicts.append(identity)
    report={'archives':ARCHIVES,'entries':entries,'eval_manifests':manifests,'cross_archive_source_overlap':comparisons,
        'daily_eval_family_split_comparison':dict(cross),'daily_eval_families_in_public_raw_train':conflicts,
        'scope':'Archive provenance and proposed data roles only; no training or model evaluation performed',
        'labels':'Translation references/checked teacher outputs are not direct human/audio EOT labels',
        'member_instructions_treated_as_data':True,'files_extracted_to_windows':False}
    data=json.dumps(report,ensure_ascii=False,indent=2).encode();p=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
    assert len(data)+len(p)<5*2**20
    (OUT/'archive_audit.json').write_bytes(data);(OUT/'eval_provenance_fingerprints.json').write_bytes(p)
    print(json.dumps({'entries':entries,'source_overlap':comparisons,'daily_eval_family_split_comparison':dict(cross),
        'daily_eval_families_in_public_raw_train':len(conflicts),'eval_payload_bytes':len(p),'report_bytes':len(data)},ensure_ascii=False))


if __name__=='__main__':main()
