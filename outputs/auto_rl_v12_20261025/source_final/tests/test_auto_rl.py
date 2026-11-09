from turn_v2.auto_rl_v11 import prefixes
from turn_v2.auto_rl_v12 import parse_decision


def test_candidate_cuts_do_not_create_labels_or_expose_future():
    source={'language':'en','source_id':'enwiki:1','title':'Sample','url':'https://example.invalid','page_revision':42,
        'text':'The library opens every morning for people to read books and meet their friends.'}
    rows=prefixes(source)
    assert len(rows)==3 and len({r['text'] for r in rows})==3
    assert all(r['label'] is None and r['is_gold'] is False and r['candidate_only'] for r in rows)
    assert all('hidden_suffix' not in r and 'full_text' not in r for r in rows)
    assert len(rows[0]['text'])<len(rows[-1]['text'])


def test_generated_judge_requires_an_unambiguous_letter():
    assert parse_decision(' B\n')=='NEOT'
    assert parse_decision('C')=='AMBIGUOUS'
    assert parse_decision('A or B') is None
    assert parse_decision('The answer is A') is None
    assert parse_decision('<think>reason</think>\nA')=='EOT'
