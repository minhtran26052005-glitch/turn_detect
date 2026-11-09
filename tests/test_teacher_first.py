from turn_v2.teacher_first_protocol import distillation_gate, independent_qualification, source_identity, training_fingerprint


def champion():
    return {'semantic_dev': {'accuracy': .99, 'eot_f1': .99}, 'weights_sha256': 'weights-A'}


def gold():
    return {'label_scope': 'SEMANTIC_EOT', 'label_origin': 'INDEPENDENT_HUMAN_OR_PUBLISHER_EOT_GOLD',
            'split': 'SEALED_UNSEEN_TEST', 'used_for_training_or_selection': False,
            'checkpoint_sha256': 'weights-A', 'metrics': {'samples': 300, 'accuracy': .94, 'eot_f1': .93},
            'label_counts': {'EOT': 140, 'NEOT': 160}, 'predictions_sha256': 'p', 'dataset_sha256': 'd'}


def test_perfect_silver_cannot_trigger_distillation():
    gate = distillation_gate({lang: champion() for lang in ['vi', 'en', 'zh']})
    assert not gate['distillation_allowed'] and not gate['quality_target_certified']
    assert all(x['teacher_silver_dev_pass'] for x in gate['languages'].values())


def test_qualification_binds_exact_teacher_and_unseen_reference():
    report = gold(); assert independent_qualification(report, 'weights-A')['passed']
    assert not independent_qualification(report, 'weights-B')['passed']
    for key, invalid in [('used_for_training_or_selection', True), ('label_origin', 'QWEN_SILVER'), ('label_scope', 'OBSERVED_FLOOR')]:
        assert not independent_qualification({**report, key: invalid}, 'weights-A')['passed']


def test_all_languages_need_both_metrics():
    champions = {lang: champion() for lang in ['vi', 'en', 'zh']}
    qualifications = {lang: gold() for lang in champions}
    assert distillation_gate(champions, qualifications)['distillation_allowed']
    qualifications['zh']['metrics']['eot_f1'] = .89
    assert not distillation_gate(champions, qualifications)['distillation_allowed']


def test_legacy_source_and_resume_fingerprint_retain_content_identity():
    rows = [{'text': 'a complete thought', 'label': 'EOT', 'group_id': 'anchor:a'},
            {'text': 'waiting for', 'label': 'NEOT', 'group_id': 'anchor:b'}]
    assert source_identity(rows[0]) == 'anchor:a'
    assert training_fingerprint(rows) == training_fingerprint(rows[::-1])
    assert training_fingerprint(rows) != training_fingerprint([{**rows[0], 'label': 'NEOT'}, rows[1]])
