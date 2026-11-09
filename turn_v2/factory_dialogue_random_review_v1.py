"""Read-only, reproducible sampling of committed dialogues; never uses the GPU."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import traceback

DRIVE = Path('/content/drive/MyDrive/turn_detect_backups/translation_segmenter_v15d_20261026')
NONCE = '9e618b0f2db442a6a9ee3ee18f1d7143'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    return data


def sample_campaign(campaign, count, seed, progress):
    snapshot = json.loads((campaign / 'state.json').read_bytes())
    frozen_index = snapshot['commit_index']
    manifests = {'vi': [], 'en': []}
    totals = Counter(vi=0, en=0)
    previous = None
    for index in range(1, frozen_index + 1):
        raw = (campaign / 'commits' / f'{index:09d}.json').read_bytes()
        commit = json.loads(raw)
        assert commit['index'] == index and commit['previous_commit_sha256'] == previous
        assert commit['object_readback_verified'] and commit['language'] in manifests
        previous = sha(raw)
        lang = commit['language']
        if commit['accepted']:
            first = totals[lang]
            totals[lang] += commit['accepted']
            manifests[lang].append((totals[lang], first, previous, commit))
        if index % 25 == 0:
            progress('READING_COMMIT_MANIFESTS', current=index, total=frozen_index)
    assert previous == snapshot['last_commit_sha256']
    assert dict(totals) == snapshot['total_counts']
    samples = []
    source_receipts = []
    for lang in ('vi', 'en'):
        assert totals[lang] >= count
        rng = random.Random(f'{seed}:{lang}:dialogue')
        turn_rng = random.Random(f'{seed}:{lang}:turn')
        ranks = rng.sample(range(totals[lang]), count)
        ends = [entry[0] for entry in manifests[lang]]
        wanted = defaultdict(list)
        for order, rank in enumerate(ranks, 1):
            entry = manifests[lang][bisect_right(ends, rank)]
            wanted[entry[3]['index']].append((order, rank, rank - entry[1], entry))
        selected = []
        for ordinal, (index, picks) in enumerate(sorted(wanted.items()), 1):
            entry = picks[0][3]
            commit = entry[3]
            raw = (campaign / commit['object']).read_bytes()
            assert len(raw) == commit['bytes'] and sha(raw) == commit['object_sha256']
            payload = json.loads(gzip.decompress(raw))
            assert payload['language'] == lang
            assert len(payload['dialogues']) == commit['accepted']
            source_receipts.append({'index': index, 'language': lang, 'object': commit['object'],
                                    'commit_sha256': entry[2], 'object_sha256': sha(raw),
                                    'readback_verified': True})
            for order, rank, offset, _ in picks:
                dialogue = payload['dialogues'][offset]
                assert dialogue['language'] == lang and dialogue['turns']
                selected.append({'number': order, 'language': lang, 'population_rank_zero_based': rank,
                                 'source_commit_index': index, 'source_object': commit['object'],
                                 'source_object_sha256': commit['object_sha256'], 'dialogue': dialogue})
            progress('READING_SELECTED_SHARDS', language=lang, current=ordinal, total=len(wanted))
        for sample in sorted(selected, key=lambda row: row['number']):
            sample['selected_turn_index'] = turn_rng.randrange(len(sample['dialogue']['turns']))
            sample['selected_turn'] = sample['dialogue']['turns'][sample['selected_turn_index']]
            samples.append(sample)
    assert Counter(s['language'] for s in samples) == {'vi': count, 'en': count}
    assert len({s['dialogue']['dialogue_id'] for s in samples}) == count * 2
    return {'samples': samples}, {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': str(seed),
        'per_language': count, 'population_dialogues': dict(totals),
        'frozen_commit_index': frozen_index, 'frozen_commit_sha256': previous,
        'method': 'Uniform without replacement over accepted dialogues per language, then one random turn per dialogue; complete dialogue retained for context.',
        'source': str(campaign), 'source_objects': source_receipts,
        'commit_chain_verified': True, 'sampled_objects_sha_readback_verified': True,
        'synthetic': True, 'review_state': 'UNREVIEWED', 'training_eligible': False,
        'source_data_modified': False, 'generation_interrupted': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def progress(stage, **kwargs):
        save(args.output / 'status.json', {'stage': stage, 'pid': os.getpid(), **kwargs})

    try:
        os.nice(5)
        assert os.path.ismount('/content/drive')
        identity = json.loads((DRIVE / 'destination_identity.json').read_bytes())
        assert identity['nonce'] == NONCE and identity['intended_path'] == str(DRIVE)
        payload, manifest = sample_campaign(DRIVE / 'factory_dialogue_campaign_v6', args.count,
                                            args.seed, progress)
        encoded = (json.dumps(payload, ensure_ascii=False) + '\n').encode()
        assert len(encoded) < 2 * 1024 * 1024, 'Review export exceeds 2 MiB allowance'
        with (args.output / 'samples.json').open('wb') as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        assert sha((args.output / 'samples.json').read_bytes()) == sha(encoded)
        manifest.update(samples_sha256=sha(encoded), samples_bytes=len(encoded))
        save(args.output / 'manifest.json', manifest)
        progress('COMPLETE', samples=len(payload['samples']), samples_bytes=len(encoded),
                 samples_sha256=sha(encoded))
    except Exception as error:
        save(args.output / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc()})
        progress('ERROR', error=repr(error))
        raise


if __name__ == '__main__':
    main()
