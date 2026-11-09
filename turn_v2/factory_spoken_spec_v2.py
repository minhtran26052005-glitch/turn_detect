"""Bilingual spoken scope, preserving the validated v9 paragraph rules."""
import factory_spoken_spec_v1 as previous

LANGS=('vi','en')
SYSTEM={l:previous.SYSTEM[l] for l in LANGS}
normalize=previous.normalize


def task_for(lang,cycle,attempt):
    if lang not in LANGS:raise ValueError('Only Vietnamese and English are authorized')
    return previous.task_for(lang,cycle,attempt)


def parse_generated(text,task):
    assert task['language'] in LANGS
    return previous.parse_generated(text,task)


def quality_check(dialogue,lang):
    assert lang in LANGS and dialogue['language']==lang
    return previous.quality_check(dialogue,lang)


def make_examples(dialogue):
    assert dialogue['language'] in LANGS
    return previous.make_examples(dialogue)


def advance(state):
    counts=state['cycle_counts']
    if all(counts.get(l,0)>=10000 for l in LANGS):
        state['cycle']+=1;state['language']='vi'
        state['cycle_counts']={**counts,'vi':0,'en':0}
    elif counts.get('vi',0)<10000:state['language']='vi'
    else:state['language']='en'
    assert state['language'] in LANGS
    return state
