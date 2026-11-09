"""Dialogue-group splits and lazy every-word streaming training examples."""
from collections import Counter
from functools import lru_cache
import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

from .common import normalize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4*1024**2), b''): h.update(part)
    return h.hexdigest()


def paragraph(turn):
    words = []; eots = []; unknown = []
    for sentence in turn['sentences']:
        text = sentence['display']; clean = normalize(text)
        assert clean == sentence['normalized'] and clean
        words.extend(clean.split())
        if text.rstrip().endswith(('.', '。')): eots.append(len(words))
        elif text.rstrip().endswith(('?', '!', '？', '！')): unknown.append(len(words))
    assert len(words) >= 4
    return {'words': words, 'eots': eots, 'unknown': unknown,
            'display': ' '.join(s['display'] for s in turn['sentences'])}


def family(dialogue):
    task = dialogue.get('generation_task', {})
    values = [task.get(k, '') for k in ('industry', 'equipment', 'topic', 'style')]
    if not any(values): values = [dialogue['dialogue_id']]
    return hashlib.sha256(json.dumps([dialogue['language'], *values], ensure_ascii=False).encode()).hexdigest()


def build(source, output, protected):
    import numpy as np
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'manifest.json').exists(), 'Dataset already prepared; do not overwrite'
    export = json.loads((source/'export_manifest.json').read_text())
    protected_ids = {s['dialogue']['dialogue_id'] for s in json.loads(protected.read_text())['samples']}
    conn = sqlite3.connect(output/'paragraphs.sqlite')
    conn.execute('CREATE TABLE paragraphs(id INTEGER PRIMARY KEY,language TEXT,split TEXT,dialogue_id TEXT,family_id TEXT,ordinal INTEGER,record TEXT)')
    counts = Counter(); source_counts = Counter(); ids = set(); canonical = set(); quarantine = []; pid = 0
    train_codes = {'vi': [], 'en': []}; heldout_ids = {'dev': set(), 'sealed': set(), 'train': set()}
    for index in range(1, export['frozen_commit_index']+1):
        commit = json.loads((source/'commits'/f'{index:09d}.json').read_text())
        archive = source/commit['object']; assert sha(archive) == commit['object_sha256']
        payload = json.loads(gzip.decompress(archive.read_bytes())); assert len(payload['dialogues']) == commit['accepted']
        for dialogue in payload['dialogues']:
            lang = dialogue['language']; identity = dialogue['dialogue_id']; source_counts[lang] += 1
            assert lang in train_codes and lang == commit['language']
            if identity in ids: raise ValueError('Duplicate dialogue identity in committed source')
            ids.add(identity)
            try:
                assert 4 <= len(dialogue['turns']) <= 12
                paras = [paragraph(t) for t in dialogue['turns']]
                assert all(2 <= len(t['sentences']) <= 8 for t in dialogue['turns'])
            except (AssertionError, KeyError, TypeError) as error:
                quarantine.append({'dialogue_id': identity, 'reason': type(error).__name__}); continue
            key = hashlib.sha256((lang+'\0'+' '.join(' '.join(p['words']) for p in paras)).encode()).hexdigest()
            if key in canonical:
                quarantine.append({'dialogue_id': identity, 'reason': 'exact_duplicate_dialogue'}); continue
            canonical.add(key); group = family(dialogue)
            split = 'exposed_regression' if identity in protected_ids else 'dev' if int(group[:8],16)%1000 < 50 else 'sealed' if int(group[:8],16)%1000 < 100 else 'train'
            if split in heldout_ids: heldout_ids[split].add(group)
            counts[(lang, split, 'dialogues')] += 1
            for ordinal, p in enumerate(paras):
                pid += 1; p.update(dialogue_id=identity, language=lang, paragraph_id=pid, source_turn=ordinal)
                conn.execute('INSERT INTO paragraphs VALUES(?,?,?,?,?,?,?)', (pid,lang,split,identity,group,ordinal,json.dumps(p,ensure_ascii=False)))
                counts[(lang,split,'paragraphs')] += 1
                if split == 'train':
                    for k in (1,2):
                        for b in range(1,len(p['words'])-k+1):
                            if b in p['unknown']: continue
                            train_codes[lang].append((pid << 32) | (b << 2) | k)
                            counts[(lang, split, 'EOT' if b in p['eots'] else 'NEOT')] += 1
        if index%25 == 0:
            conn.commit(); (output/'preparation_status.json').write_text(json.dumps({'stage':'BUILDING_PARAGRAPHS','commit':index,'total':export['frozen_commit_index']}))
    assert dict(source_counts) == export['source_state']['total_counts']
    assert not heldout_ids['train'] & (heldout_ids['dev'] | heldout_ids['sealed'])
    assert not heldout_ids['dev'] & heldout_ids['sealed']
    conn.execute('CREATE INDEX language_split ON paragraphs(language,split,id)'); conn.commit()
    assert conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    dev = {}
    for lang in ('vi','en'):
        candidates = [(hashlib.sha256(f'{identity}:{ordinal}'.encode()).hexdigest(), json.loads(record))
                      for identity,ordinal,record in conn.execute("SELECT dialogue_id,ordinal,record FROM paragraphs WHERE language=? AND split='dev'",(lang,))]
        dev[lang] = [p for _,p in sorted(candidates)[:24]]; assert len(dev[lang]) == 24
        codes = np.asarray(train_codes[lang],dtype=np.uint64); assert len(codes)>1000
        np.save(output/f'train_index_{lang}.npy',codes)
    streams = []
    for lang in ('vi','en'):
        for p in dev[lang]:
            streams.append({**p,'language_by_word':[lang]*len(p['words']),'mode':'monolingual'})
    for i in range(12):
        for first,second in [('vi','en'),('en','vi')]:
            a,b=dev[first][i],dev[second][i];offset=len(a['words'])
            streams.append({'words':a['words']+b['words'],'eots':a['eots']+[x+offset for x in b['eots']],
                            'unknown':a['unknown']+[x+offset for x in b['unknown']],
                            'language_by_word':[first]*offset+[second]*len(b['words']),
                            'source_ids':[a['dialogue_id'],b['dialogue_id']],'mode':'synthetic_splice','splice':offset})
    conn.close(); (output/'dev_streams.json').write_text(json.dumps(streams,ensure_ascii=False),encoding='utf-8')
    (output/'quarantine.json').write_text(json.dumps(quarantine,ensure_ascii=False,indent=2),encoding='utf-8')
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in output.iterdir() if p.name in ['paragraphs.sqlite','train_index_vi.npy','train_index_en.npy','dev_streams.json','quarantine.json']}
    manifest={'complete':True,'source_export_manifest_sha256':sha(source/'export_manifest.json'),
              'source_frozen_commit_index':export['frozen_commit_index'],'all_source_dialogues':dict(source_counts),
              'counts':{':'.join(k):v for k,v in counts.items()},'quarantined_dialogues':len(quarantine),
              'protected_exposed_dialogues':len(protected_ids),'files':files,
              'split_rule':'5% dev, 5% sealed by language+industry/equipment/topic/style family; exposed review IDs excluded from train/dev; split before boundary expansion.',
              'train_targets':'All eligible word gaps: EOT only source sentence-final full stop; otherwise NEOT; ?/! endpoints unscored.',
              'context_variants':'25% rolling history,50% reference-reset TRAIN ONLY,25% random fragment starts; DEV inference never uses reference resets.',
              'cross_language_training':'25% extra synthetic VI/EN splices, only train sources, not translations.',
              'labels':'Qwen synthetic punctuation-derived silver, not human/audio gold',
              'training_authorized':'Explicit user request 2026-11-02 to transfer all generated data and train a new version; raw source flags unchanged.',
              'sealed_scored':False,'source_generator_modified':False}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'preparation_status.json').write_text(json.dumps({'stage':'COMPLETE','counts':manifest['counts']}),encoding='utf-8')
    return manifest


class Sampler:
    def __init__(self, root, seed=20261102, state=None):
        import numpy as np
        self.np=np;self.root=Path(root);self.db=sqlite3.connect((self.root/'paragraphs.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        self.index={l:np.load(self.root/f'train_index_{l}.npy',mmap_mode='r') for l in ('vi','en')}
        self.pids={l:[r[0] for r in self.db.execute("SELECT id FROM paragraphs WHERE language=? AND split='train' ORDER BY id",(l,))] for l in ('vi','en')}
        self.seed=seed;self.rng=np.random.default_rng(seed);self.positions={'vi':0,'en':0};self.epochs={'vi':0,'en':0};self.perms={}
        if state:self.load_state_dict(state)

    @lru_cache(maxsize=4096)
    def get(self,pid):
        return json.loads(self.db.execute('SELECT record FROM paragraphs WHERE id=?',(int(pid),)).fetchone()[0])

    def state_dict(self):
        import copy
        return copy.deepcopy({'positions':self.positions,'epochs':self.epochs,'rng':self.rng.bit_generator.state,'seed':self.seed})

    def load_state_dict(self,s):
        import copy
        assert s['seed']==self.seed;self.positions=copy.deepcopy(s['positions']);self.epochs=copy.deepcopy(s['epochs']);self.rng.bit_generator.state=copy.deepcopy(s['rng']);self.perms={}

    def next_point(self,lang):
        size=len(self.index[lang])
        if self.positions[lang]>=size:self.positions[lang]=0;self.epochs[lang]+=1
        epoch=self.epochs[lang]
        if lang not in self.perms or self.perms[lang][0]!=epoch:
            self.perms[lang]=(epoch,self.np.random.default_rng(self.seed+epoch*17+(0 if lang=='vi' else 1)).permutation(size))
        code=int(self.index[lang][self.perms[lang][1][self.positions[lang]]]);self.positions[lang]+=1
        return self.get(code>>32),(code&0xffffffff)>>2,code&3

    def row(self,lang):
        mixed=self.rng.random()<.25
        if mixed:
            first,second=('vi','en') if self.rng.random()<.5 else ('en','vi')
            a=self.get(self.pids[first][int(self.rng.integers(len(self.pids[first])))])
            c=self.get(self.pids[second][int(self.rng.integers(len(self.pids[second])))])
            offset=len(a['words']);words=a['words']+c['words'];eots=a['eots']+[x+offset for x in c['eots']];unknown=a['unknown']+[x+offset for x in c['unknown']]
            k=int(self.rng.integers(1,3));choice=self.rng.random()
            b=offset if choice<.5 else max(1,offset-1) if choice<.75 else int(self.rng.integers(1,len(words)-k+1))
            if b in unknown:b=next(x for x in range(1,len(words)-k+1) if x not in unknown)
            target_language=first if b<=offset else second
        else:
            p,b,k=self.next_point(lang);words=p['words'];eots=p['eots'];unknown=p['unknown'];target_language=lang
        assert b not in unknown and b+k<=len(words)
        mode=self.rng.random()
        start=0 if mode<.25 else (eots[bisect.bisect_left(eots,b)-1] if bisect.bisect_left(eots,b) else 0) if mode<.75 else int(self.rng.integers(0,b))
        return {'context':' '.join(words[start:b]),'text':' '.join(words[b:b+k]),'label':'EOT' if b in eots else 'NEOT',
                'language':target_language,'lookahead_words':k,'synthetic_splice':mixed,'context_start_mode':'rolling' if mode<.25 else 'teacher_forced_training_only' if mode<.75 else 'fragment_training_only'}

    def preview_sha(self,lang):
        state=self.state_dict();rows=[self.row(lang) for _ in range(4)];self.load_state_dict(state)
        return hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--protected',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.protected),ensure_ascii=False))
