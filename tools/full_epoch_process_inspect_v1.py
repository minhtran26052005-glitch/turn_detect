import json
from pathlib import Path
root=Path('/content/turn_detect_v2/outputs/translation_segmenter_v15d_20261026');o={}
for label,path in [('parent',root/'launcher.json'),('child',root/'child.json'),('transition',root/'full_epoch_v1/transition_v2/launcher.json'),('audit',root/'full_epoch_v1/data_launcher_v2.json'),('backup',root/'drive_backup_v1/launcher.json')]:
    a=json.loads(path.read_text());proc=Path('/proc')/str(a['pid']);item={'record':a}
    try:
        stat=(proc/'stat').read_text().split();argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
        item.update(identity_verified=stat[21]==a['start_ticks'] and argv==a['command'],state=stat[2],wchan=(proc/'wchan').read_text(),children=(proc/'task'/str(a['pid'])/'children').read_text(),cmdline=argv)
    except OSError:item['identity_verified']=False
    o[label]=item
print(json.dumps(o))
