import json,os,signal,sqlite3,time,sys
from pathlib import Path
root=Path('/content/turn_detect_v2');run=root/'outputs/translation_segmenter_v15d_20261026'
sys.path.insert(0,str(root));os.chdir(root)
from turn_v2.translation_segmenter_full_campaign_v1 import live
from turn_v2.translation_segmenter_v15c import atomic_json
parent=json.loads((run/'launcher.json').read_text());assert live(parent)
assert 'turn_v2.translation_segmenter_v15d3' in parent['command']
state=json.loads((run/'status.json').read_text());assert state['status']=='STOPPED_BY_REQUEST'
child=json.loads((run/'child.json').read_text());assert not live(child)
checks={}
for lang in ('vi','en','zh'):
    r=json.loads((run/'teachers'/lang/'resume_receipt.json').read_text())
    report=json.loads((run/'teachers'/lang/'last_training_report.json').read_text())
    assert r['cycle']==84 and r['cycle_step']==100 and r['total_steps']==1400
    assert report['cycle']==84 and report['stage']=='trained'
    checks[lang]=r
proc=Path('/proc')/str(parent['pid']);backup=json.loads((run/'drive_backup_v1/launcher.json').read_text())
assert live(backup)
children=(proc/'task'/str(parent['pid'])/'children').read_text().split()
assert set(children)<={str(backup['pid'])},children
db=sqlite3.connect('file:'+str(run/'corpus.sqlite')+'?mode=ro',uri=True)
next_cycle=int(db.execute("SELECT value FROM meta WHERE key='next_cycle'").fetchone()[0]);db.close();assert next_cycle==85
receipt={'reason':'Old launcher completed normal STOPPED_BY_REQUEST but remains in futex cleanup with only independent backup child',
 'parent':parent,'active_train_child_absent':True,'checkpoint_receipts':checks,
 'raw_next_cycle':next_cycle,'backup_child_untouched':backup,'signal_scope':'single verified completed launcher only'}
atomic_json(run/'full_epoch_v1/transition_v2/old_launcher_cleanup_exit_receipt.json',receipt)
assert live(parent);os.kill(parent['pid'],signal.SIGKILL)
print(json.dumps({'finished_completed_old_launcher':parent['pid'],'all3_cycle84_checkpoints_complete':True,'backup_untouched':True}))
