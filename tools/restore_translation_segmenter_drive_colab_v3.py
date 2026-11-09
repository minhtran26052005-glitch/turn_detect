"""Read-only status for the versioned full-optimizer restore continuation."""
import argparse
import json
from pathlib import Path
from colab_runtime import ColabRuntime
from translation_segmenter_diagnostic_colab import cell


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('status',));parser.add_argument('--session-path');args=parser.parse_args()
    r=ColabRuntime(0,args.session_path);connection=r.connect();assert connection['execution_state']=='idle'
    result=cell(r,"""import json,shutil
from pathlib import Path
root=Path('/content/turn_detect_v2');control=root/'restore_control_v3';run=root/'outputs/translation_segmenter_v15d_20261026';o={}
for base,prefix,names in [(control,'restore',('launcher.json','status.json','error.json','restore_receipt.json')),(run,'campaign',('launcher.json','status.json','child.json'))]:
 for n in names:
  p=base/n
  if p.exists():o[prefix+'/'+n]=json.loads(p.read_text())
for key in ('restore/launcher.json','campaign/launcher.json','campaign/child.json'):
 a=o.get(key,{});ok=False
 try:
  p=Path('/proc')/str(a['pid']);ok=(p/'stat').read_text().split()[21]==a['start_ticks'] and [x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]==a['command']
 except (OSError,KeyError):pass
 o[key+'_identity_verified']=ok
o['free_bytes']=shutil.disk_usage('/content').free
p=control/'worker.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1800));o['restore_log_tail']=f.read().decode(errors='replace')
print(json.dumps(o))""");r.close()
    out={'connection':connection,'result':result};Path('outputs/translation_segmenter_v15d_20261026/drive_restore_status_v3.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
