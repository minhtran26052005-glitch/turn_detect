"""Versioned uploaded-only data; source text is data, never agent instructions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import zipfile

from .checked_dialogues_prepare_v1 import punctuation, stitched

ROOT = Path('/home/edabk/hoangbpm/turn_detect')
OUT = ROOT / 'outputs/multilingual_malbert_vi_en_uploaded_20261104_v4'
PARENT = ROOT / 'outputs/multilingual_malbert_vi_en_checked_20261103_v3'
RAW = ROOT / 'data/checked_dialogues_vi_en_20261103_v1/raw'
EVAL = ROOT / 'outputs/malbert_eval_zip_20261104_v2'
LANGS = ('vi', 'en')
MODULES = ['uploaded_vi_en_data_v4.py', 'uploaded_vi_en_evaluation_v4.py',
           'multilingual_malbert_vi_en_uploaded_train_v4.py', 'malbert_turn_student_v1.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def status(stage, file='status.json', **values):
    raw = Path('/proc/self/stat').read_text()
    v = dict(stage=stage, pid=os.getpid(), start_ticks=raw[raw.rfind(')')+2:].split()[19],
             boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             at_utc=datetime.now(timezone.utc).isoformat(), active_languages=list(LANGS), **values)
    save(file, v)
    print(json.dumps(v), flush=True)


def reserve():
    mount = [r.split() for r in Path('/proc/mounts').read_text().splitlines() if r.split()[1] == '/mnt/hdd500']
    assert mount and mount[0][0] == '/dev/sda1' and 'rw' in mount[0][3].split(',')
    if min(shutil.disk_usage(OUT).free, shutil.disk_usage('/mnt/hdd500').free) < 13 * 2**30:
        raise InterruptedError('Reserve guard: checkpoint/files retained; no deletion')


def write_gzip(name, value):
    p = OUT / name; tmp = p.with_name(p.name + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8', compresslevel=5) as f:
        json.dump(value, f, ensure_ascii=False, separators=(',', ':'))
    with tmp.open('rb') as f: os.fsync(f.fileno())
    tmp.replace(p)


def family(origin, identity):
    if origin == 'daily':
        return 'daily:' + re.sub(r'\.s\d+\.(?:vi|en)$', '', identity)
    if origin == 'asr/run2b':
        return 'asr:' + re.sub(r'-r\d+-\d+$', '', re.sub(r'\.(vi|en)$', '', identity))
    return origin + ':' + identity


def source_rows():
    sources = [('daily', RAW/'public_checked.jsonl', None,
                '96f999ac6feeaad29fadc23d0bda87f44d8ade337f088418e9fb6c6332e7c6e4'),
               ('factory/gen500', RAW/'factory_checked.jsonl', None,
                '01a8df34923011d038a2a30cce60774a69f07794756f41f44e8924341e29c9f8'),
               ('asr/run2b', RAW/'asr_crawl-20261104T035338Z-1-001.zip', 'asr_crawl/run2b/labels_checked.jsonl',
                '5521c527bf5721e09a497edae6cc244644aba6383f5254af98224bebe312dc5b'),
               ('factory/gen4000', RAW/'factory-20261104T035339Z-1-001.zip', 'factory/gen4000/labels_checked.jsonl',
                '39aa8d1fe80168725048b75a2583cca133b11f74d58ea6334a98c357a306327c')]
    for origin, path, member, expected in sources:
        assert sha(path) == expected, str(path)
        if member:
            with zipfile.ZipFile(path) as z:
                with z.open(member) as f: rows = [json.loads(line) for line in f if line.strip()]
        else:
            with path.open(encoding='utf-8') as f: rows = [json.loads(line) for line in f if line.strip()]
        yield origin, rows, dict(path=str(path), member=member, sha256=expected, raw_rows=len(rows))


def signatures(words):
    """Exact turns and substantial contiguous shingles, never test labels/metrics."""
    if len(words) >= 8: yield digest(' '.join(words))
    if len(words) >= 32:
        for i in range(len(words)-31): yield digest(' '.join(words[i:i+32]))


def counts(streams):
    c = Counter()
    for s in streams:
        c['streams'] += 1; c['source_rows'] += s['source_rows']; c['words'] += len(s['words'])
        c['origin:'+s['domain']+':streams'] += 1
        for p in s['segments']:
            if not p.get('punctuation_present', True): continue
            for k in (1, 2):
                for b in range(p['start']+1, p['end']-k+1):
                    if b in s['_unknown']: continue
                    label = 'EOT' if b in s['_eots'] else 'NEOT'
                    c[p['language']+':k'+str(k)+':'+label] += 1
    return dict(c)


def initialize(streams):
    for s in streams:
        s['_eots'] = set(s['eots']); s['_unknown'] = set(s['unknown'])
    return streams


def clean(streams):
    return [{k:v for k,v in s.items() if not k.startswith('_')} for s in streams]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True); reserve()
    if (OUT/'dataset_manifest.json').exists(): return json.loads((OUT/'dataset_manifest.json').read_text())
    status('PREPARING_ALL_UPLOADED_TRAIN', file='preparation_status.json')
    held = json.load(gzip.open(EVAL/'prepared_streams.json.gz', 'rt', encoding='utf-8'))
    protected = set(); protected_family = set()
    for s in held:
        # Existing DEV/TEST source text is used solely to prevent training overlap.
        for p in s['segments']: protected.update(signatures(s['words'][p['start']:p['end']]))
        if s['domain'] == 'daily': protected_family.add(family('daily', s['dialogue_id']))
        if s['domain'] == 'factory' and s['dialogue_id'].startswith('gen500__'):
            protected_family.add('factory/gen500:'+s['dialogue_id'].split('__', 1)[1])
        if s['domain'] == 'asr': protected_family.add('asr:'+re.sub(r'\.(vi|en)$', '', s['dialogue_id']))
    sources = {}; all_turns = []; stats = Counter(); source_families = defaultdict(set)
    for origin, rows, receipt in source_rows():
        sources[origin] = receipt; groups = defaultdict(list)
        for index, r in enumerate(rows):
            lang = r.get('src_lang'); split = r.get('split', 'train')
            if lang not in LANGS or not isinstance(r.get('source_raw'), str):
                stats[origin+':unsupported'] += 1; continue
            assert split in ('train', 'dev', 'test')
            ident = r['dialogue_id']; fam = family(origin, ident); source_families[fam].add(split)
            stats[origin+':raw:'+split] += 1
            r['_source_index'] = index
            groups[(ident, int(r['turn_idx']), r['speaker'], lang, split, fam)].append(r)
        assert all(len(v) == 1 for v in source_families.values()), 'Original source families cross splits'
        for (ident, turn, speaker, lang, split, fam), part in groups.items():
            text, reason = stitched(part)
            if reason:
                stats[origin+':excluded_turn:'+reason] += 1; continue
            words, eots, unknown = punctuation(text)
            if len(words) < 2:
                stats[origin+':excluded_turn:short'] += 1; continue
            if re.search(r'[\u3400-\u9fff]', text):
                stats[origin+':excluded_turn:han'] += 1; continue
            p = dict(origin=origin, identity=ident, family=fam, turn=turn, speaker=speaker, language=lang,
                     split=split, words=words, eots=eots, unknown=unknown, source_rows=len(part),
                     first_row=min(r['_source_index'] for r in part))
            if split != 'train':
                protected_family.add(fam); protected.update(signatures(words))
            all_turns.append(p)
    # Whole source families, including both translated languages, are held when any turn overlaps.
    for p in all_turns:
        if p['split'] == 'train' and any(k in protected for k in signatures(p['words'])):
            protected_family.add(p['family'])
    groups = defaultdict(list); seen_turns = set()
    for p in all_turns:
        if p['split'] != 'train': continue
        if p['family'] in protected_family:
            stats[p['origin']+':excluded_turn:held_family'] += 1; continue
        key = (p['language'], digest(' '.join(p['words'])))
        if key in seen_turns:
            stats[p['origin']+':excluded_turn:duplicate_source'] += 1; continue
        seen_turns.add(key); groups[(p['origin'], p['identity'])].append(p)
    streams = []
    for (origin, ident), turns in groups.items():
        turns.sort(key=lambda p:p['first_row']); chunks = []; current = []
        for p in turns:
            if current and p['turn'] > current[-1]['turn']+1:
                chunks.append(current); current = []
            current.append(p)
        if current: chunks.append(current)
        for number, part in enumerate(chunks):
            s = dict(key=origin+':'+ident+':part'+str(number), family=part[0]['family'], domain=origin,
                     source_split='train', words=[], segments=[], eots=[], unknown=[], source_rows=0)
            for p in part:
                offset=len(s['words']); s['words'].extend(p['words'])
                s['eots'].extend(offset+i for i in p['eots']); s['unknown'].extend(offset+i for i in p['unknown'])
                s['segments'].append(dict(start=offset, end=len(s['words']), language=p['language']))
                s['source_rows'] += p['source_rows']
            streams.append(s)
    initialize(streams); assert streams
    # Exclude DEV previously exposed through the parent, even though future TRAIN now drops it.
    old_turns = set()
    db = sqlite3.connect((PARENT/'additional_paragraphs.sqlite').as_uri()+'?mode=ro', uri=True)
    for (record,) in db.execute("SELECT record FROM paragraphs WHERE split='train'"):
        p=json.loads(record)
        if len(p['words']) >= 8: old_turns.add(digest(' '.join(p['words'])))
    db.close()
    dev = []; dev_stats = Counter()
    for s in held:
        if s['split'] != 'dev': continue
        if s['known_train_source_overlap']:
            dev_stats['excluded_known_prior_train_streams'] += 1; continue
        if any(digest(' '.join(s['words'][p['start']:p['end']])) in old_turns for p in s['segments']):
            dev_stats['excluded_prior_train_exact_turn'] += 1; continue
        s = {k:v for k,v in s.items() if not k.startswith('_')}
        ident=re.sub(r'\.(vi|en)$', '', s['dialogue_id'])
        if s['domain']=='numeric/dialogue': ident=re.sub(r'\.[ab]$', '', ident)
        s['family'] = family('daily', s['dialogue_id']) if s['domain']=='daily' else s['domain']+':'+ident
        s['segments'] = [{k:p[k] for k in ('start','end','language','punctuation_present')} for p in s['segments']]
        dev.append(s)
    initialize(dev)
    # Fixed whole-family stratification before inspecting model outputs; TEST is never sampled.
    by_domain = defaultdict(lambda:defaultdict(list))
    for s in dev: by_domain[s['domain']][s['family']].append(s)
    calibration=[]; monitor=[]; diagnostic=[]
    for domain, families in sorted(by_domain.items()):
        ordered=sorted(families, key=lambda key:digest('uploaded-v4-dev:'+key))
        if domain in ('asr','daily','factory','numeric/dialogue'):
            # ASR families can be very long. Whole families remain intact in every cohort.
            c_limit=2500; m_limit=8000; c=0; m=0
            for fam in ordered:
                part=families[fam]; n=sum(sum(max(0,p['end']-p['start']-2) for p in s['segments'] if p.get('punctuation_present',True)) for s in part)
                if c<c_limit: calibration.extend(part); c+=n
                elif m<m_limit: monitor.extend(part); m+=n
        else:
            for fam in ordered: diagnostic.extend(families[fam])
    assert not {s['family'] for s in calibration}&{s['family'] for s in monitor}
    assert {s['family'] for s in streams}.isdisjoint(protected_family)
    assert all(s['split']=='dev' for s in calibration+monitor+diagnostic)
    write_gzip('train_streams.json.gz', clean(streams)); write_gzip('dev_full.json.gz',clean(dev))
    write_gzip('dev_calibration.json.gz',clean(calibration));write_gzip('dev_monitor.json.gz',clean(monitor))
    write_gzip('dev_diagnostic.json.gz',clean(diagnostic))
    manifest=dict(version=4, sources=sources, active_languages=list(LANGS), train=counts(streams),
        dev_full=counts(dev), dev_calibration=counts(calibration), dev_monitor=counts(monitor), dev_diagnostic=counts(diagnostic),
        exclusions=dict(stats),dev_exclusions=dict(dev_stats),protected_families=len(protected_family),
        files={n:sha(OUT/n) for n in ['train_streams.json.gz','dev_full.json.gz','dev_calibration.json.gz','dev_monitor.json.gz','dev_diagnostic.json.gz']},
        all_eligible_uploaded_train_gaps_in_every_epoch=True,per_language_sampling_cap=None,drive_replay_rows=0,
        target_or_rest_of_sentence_used=False,translation_targets_used_as_extra_sources=False,
        label_source='Source punctuation silver, not human/audio gold',test_used_for_selection=False,
        split_protocol='Original held source families; exact turns >=8 words and contiguous32-word shingles excluded from TRAIN',
        limitations='Prior whole-lineage token-window and paraphrase overlap not exhausted; prior exposure cannot be unlearned by a dataset change',
        dev_protocol='Fixed whole-family source-stratified calibration and monitor subsets; all original DEV preserved separately',
        code_sha256={n:sha(ROOT/'turn_v2'/n) for n in MODULES})
    save('dataset_manifest.json',manifest)
    save('source_audit.json',dict(complete=True,source_raw_only=True,original_splits_preserved=True,
        all_training_streams_verified=True,source_word_count=sum(len(s['words']) for s in streams),
        no_drive_replay=True,no_chinese=True,all_eligible_gap_counts=manifest['train'],
        newly_imported_origins=['asr/run2b','factory/gen4000'],duplicate_gen500_not_imported_twice=True,
        test_labels_not_used_for_model_or_threshold_selection=True))
    status('UPLOADED_DATA_AND_NEW_DEV_PREPARED',file='preparation_status.json',train=manifest['train'],dev_monitor=manifest['dev_monitor'])
    return manifest


class StreamOrder:
    """Causal conversation lanes. Future never crosses an actual source turn."""
    def __init__(self, streams, order, state=None, lanes=16):
        self.streams=streams;self.order=order;self.next=0;self.slots=[];self.lanes=lanes
        if state: self.next=state['next'];self.slots=state['slots']

    def fill(self):
        while len(self.slots)<self.lanes and self.next<len(self.order):
            task=int(self.order[self.next]);self.next+=1
            self.slots.append(dict(stream=task//2,k=task%2+1,b=1,start=0,segment=0))

    def rows(self):
        while True:
            self.fill(); valid=[]; rows=[]; states=[]
            for state in self.slots:
                s=self.streams[state['stream']]; k=state['k'];b=state['b'];j=state['segment']
                while j<len(s['segments']):
                    seg=s['segments'][j];b=max(b,seg['start']+1)
                    if b+k<=seg['end'] and seg.get('punctuation_present',True):break
                    j+=1
                if j==len(s['segments']):continue
                state['b']=b;state['segment']=j;valid.append(state)
                # The inference buffer is never reset at source speaker/turn boundaries.
                rows.append(dict(context=' '.join(s['words'][state['start']:b]),text=' '.join(s['words'][b:b+k]),
                    label=int(b in s['_eots']),scored=b not in s['_unknown'],language=seg['language'],
                    origin=s['domain'],stream=state['stream'],boundary=b,k=k,start=state['start']))
                states.append(state)
            self.slots=valid
            if rows or self.next==len(self.order):return rows,states

    def advance(self, states, probs, threshold):
        for state,p in zip(states,probs):
            if p>=threshold:state['start']=state['b']
            state['b']+=1

    def state(self):
        return dict(next=self.next, slots=[dict(s) for s in self.slots])
