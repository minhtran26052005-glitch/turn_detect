"""Refresh only the campaign-owned status cell, preserving every earlier cell."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

from colab_runtime import ColabRuntime
from qat_colab import SESSION


def main(args):
    root=Path('outputs/natural_v7_20261025');path=Path('a.ipynb')
    notebook=json.loads(path.read_text(encoding='utf-8'))
    matches=[c for c in notebook['cells'] if c['cell_type']=='code' and 'natural_campaign_status' in c.get('metadata',{}).get('tags',[])]
    assert len(matches)==1;cell=matches[0];source=Path('tools/templates/natural_status_cell.py').read_text(encoding='utf-8')
    if ''.join(cell['source'])!=source:
        assert not cell['outputs'],'Do not silently discard executed content when changing its code'
        cell['source']=source.splitlines(keepends=True)
    runtime=ColabRuntime(0,SESSION);connection=runtime.connect()
    result=runtime.execute(source,timeout=90,echo=False)
    receipt={'connection':connection,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result}
    filename='notebook_colab_receipt_live.json' if args.live else 'notebook_colab_receipt.json'
    destination=root/filename
    if destination.exists():raise RuntimeError('Keep prior notebook receipt; choose a fresh phase instead of overwriting')
    destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['status']=='ok','Inspect the saved receipt; notebook remains editable'
    cell['outputs']=result['outputs'];cell['metadata']['colab_verified_at']=datetime.now(timezone.utc).isoformat()
    cell['metadata']['verification_receipt']=str(destination)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (root/'notebook_status_cell.py').write_text(source,encoding='utf-8')
    print(json.dumps({'status':result['status'],'outputs':len(result['outputs']),'phase':'live' if args.live else 'final','earlier_cells_changed':False}))
    runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');main(parser.parse_args())
