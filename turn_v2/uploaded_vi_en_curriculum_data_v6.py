"""Immutable, nested whole-family TRAIN curriculum for the uploaded VI/EN data."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import math
import os
import shutil

import numpy as np

from .uploaded_vi_en_data_v5 import ROOT, LANGS, StreamOrder, initialize, counts, clean, sha

PARENT = ROOT/'outputs/multilingual_malbert_vi_en_conflict_weighted_20261106_v5'
OUT = ROOT/'outputs/multilingual_malbert_vi_en_curriculum_20261106_v6'
MODULES = ['uploaded_vi_en_data_v4.py', 'uploaded_vi_en_data_v5.py',
           'uploaded_vi_en_curriculum_data_v6.py', 'uploaded_vi_en_curriculum_evaluation_v6.py',
           'multilingual_malbert_vi_en_curriculum_train_v6.py', 'malbert_turn_student_v1.py',
           'tiny_compression_v1/common.py']
FRACTIONS = (.10, .25, .50, 1.)
UPDATE_BUDGETS = (10000, 20000, 30000, None)


def save(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def status(stage, file='status.json', **values):
    raw = Path('/proc/self/stat').read_text()
    record = dict(stage=stage, pid=os.getpid(), start_ticks=raw[raw.rfind(')')+2:].split()[19],
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  at_utc=datetime.now(timezone.utc).isoformat(), active_languages=list(LANGS), **values)
    save(file, record)
    print(json.dumps(record), flush=True)


def reserve():
    mount = next(row.split() for row in Path('/proc/mounts').read_text().splitlines()
                 if row.split()[1] == '/mnt/hdd500')
    assert mount[0] == '/dev/sda1' and 'rw' in mount[3].split(',')
    if min(shutil.disk_usage(OUT).free, shutil.disk_usage('/mnt/hdd500').free) < 13*2**30:
        raise InterruptedError('Reserve guard requires a full checkpoint and waiting')


def make_order(stage, rng):
    indices = np.asarray(stage['stream_indices'], dtype=np.int64)
    tasks = np.stack((2*indices, 2*indices+1), axis=1).reshape(-1)
    return rng.permutation(tasks)


def check_order(stage, order):
    expected = {2*index+k for index in stage['stream_indices'] for k in (0, 1)}
    assert len(order) == len(expected) and set(map(int, order)) == expected


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    reserve()
    if (OUT/'dataset_manifest.json').exists() and (OUT/'independent_preparation_verification.json').exists():
        return json.loads((OUT/'dataset_manifest.json').read_text())
    status('PREPARING_NESTED_TRAIN_CURRICULUM', file='preparation_status.json')
    parent = json.loads((PARENT/'dataset_manifest.json').read_text())
    for name, digest in parent['files'].items():
        assert sha(PARENT/name) == digest
        if not (OUT/name).exists():
            os.link(PARENT/name, OUT/name)
    for name in ['teacher_new_dev.json', 'conflict_filter_audit.json']:
        if not (OUT/name).exists():
            os.link(PARENT/name, OUT/name)
        assert sha(OUT/name) == sha(PARENT/name)
    streams = initialize(json.load(gzip.open(OUT/'train_streams.json.gz', 'rt', encoding='utf-8')))
    families = defaultdict(list)
    for index, source in enumerate(streams):
        assert source['source_split'] == 'train'
        families[source['family']].append(index)
    records = {}
    strata = defaultdict(list)
    for family, indices in families.items():
        domains = {streams[index]['domain'] for index in indices}
        assert len(domains) == 1
        languages = sorted({segment['language'] for index in indices for segment in streams[index]['segments']})
        assert set(languages) <= set(LANGS)
        possible = 0
        excluded = 0
        maximum_turn = 0
        words = 0
        for index in indices:
            source = streams[index]
            words += len(source['words'])
            masks = {key:set(values) for key, values in source['conflict_mask'].items()}
            for segment in source['segments']:
                maximum_turn = max(maximum_turn, segment['end']-segment['start'])
                for k in (1, 2):
                    for boundary in range(segment['start']+1, segment['end']-k+1):
                        possible += 1
                        excluded += int(boundary in source['_unknown'] or boundary in masks[str(k)])
        digest = hashlib.sha256(('curriculum-v6:'+family).encode()).hexdigest()
        difficulty = (possible == excluded, excluded/max(possible, 1), maximum_turn, words, digest)
        stratum = next(iter(domains))+'|'+','.join(languages)
        records[family] = dict(stream_indices=indices, stratum=stratum, difficulty=difficulty,
                               possible_gaps=possible, excluded_gaps=excluded)
        strata[stratum].append(family)
    for members in strata.values():
        members.sort(key=lambda family:records[family]['difficulty'])
    stages = []
    previous = set()
    for number, (fraction, budget) in enumerate(zip(FRACTIONS, UPDATE_BUDGETS)):
        selected = {family for members in strata.values() for family in members[:max(1, math.ceil(len(members)*fraction))]}
        assert previous <= selected
        indices = sorted(index for family in selected for index in families[family])
        stage_counts = counts([streams[index] for index in indices])
        for language in LANGS:
            for label in ['EOT', 'NEOT']:
                assert sum(stage_counts.get(language+':k'+str(k)+':'+label, 0) for k in (1, 2)) > 0
        stage = dict(index=number, fraction_of_families=fraction, actual_family_fraction=len(selected)/len(families),
                     family_count=len(selected), stream_indices=indices, update_budget=budget,
                     train=stage_counts, eligible_points_per_pass=sum(value for key, value in stage_counts.items() if ':k' in key),
                     stratum_family_counts={key:sum(family in selected for family in members) for key, members in strata.items()})
        stages.append(stage)
        previous = selected
    assert stages[-1]['stream_indices'] == list(range(len(streams)))
    assert stages[-1]['train'] == parent['train']
    plan = dict(version=6, stages=stages, total_families=len(families), total_streams=len(streams),
                ranking='TRAIN-only excluded-gap fraction, longest actual source turn, word count, deterministic hash; stratified by source and family language composition',
                score_from_dev_test_or_model=False, whole_families=True, nested_replay=True,
                expansion='Fixed optimizer update budgets 10000,20000,30000, then all TRAIN',
                early_stage_complete_does_not_mean_all_pool_gaps_visited=True,
                full_stage_minimum_epochs=6, stop_for_plateau=False, language_distribution='Natural within every selected whole family')
    save('curriculum_plan.json', plan)
    manifest = dict(parent)
    manifest.update(version=6, parent_manifest_sha256=sha(PARENT/'dataset_manifest.json'),
                    curriculum_plan_sha256=sha(OUT/'curriculum_plan.json'),
                    all_eligible_uploaded_train_gaps_in_every_epoch=False,
                    all_eligible_uploaded_train_gaps_in_full_stage_epoch=True,
                    code_sha256={name:sha(ROOT/'turn_v2'/name) for name in MODULES})
    save('dataset_manifest.json', manifest)
    # Rebuild each nested membership from source families, independently of sorted index lists.
    family_stage = {}
    for stage in stages:
        chosen = set(stage['stream_indices'])
        selected_families = {streams[index]['family'] for index in chosen}
        assert chosen == {index for family in selected_families for index in families[family]}
        for family in selected_families:
            family_stage.setdefault(family, stage['index'])
        check_order(stage, make_order(stage, np.random.default_rng(71)))
    assert len(family_stage) == len(families)
    for name, digest in parent['files'].items():
        assert sha(OUT/name) == digest
    save('independent_preparation_verification.json', dict(complete=True, families=len(families),
         streams=len(streams), stages=[{key:stage[key] for key in ['index','family_count','eligible_points_per_pass','update_budget']} for stage in stages],
         all_nested_family_memberships_and_unique_task_orders_verified=True,
         full_stage_exact_parent_counts_and_all_indices_verified=True, source_dev_and_masks_byte_identical=True,
         train_only_ranking=True, no_new_samples_or_labels=True, manifest_sha256=sha(OUT/'dataset_manifest.json'),
         curriculum_plan_sha256=sha(OUT/'curriculum_plan.json')))
    status('CURRICULUM_PREPARATION_COMPLETE', file='preparation_status.json',
           stages=[{key:stage[key] for key in ['index','family_count','eligible_points_per_pass','update_budget']} for stage in stages])
    return manifest


if __name__ == '__main__':
    prepare()
