"""Append the automatic-loop receipt and preserve all earlier notebook cells."""
import hashlib
import json
from pathlib import Path
import shutil
from build_notebook import cell
from colab_runtime import ColabRuntime


def main():
    root=Path('outputs/auto_rl_v12_20261025');path=Path('a.ipynb');nb=json.loads(path.read_text(encoding='utf-8'));tag='auto_rl_status'
    assert not any(tag in c.get('metadata',{}).get('tags',[]) for c in nb['cells'])
    shutil.copy2(path,root/'a_before_auto_rl.ipynb')
    source=Path('tools/templates/auto_rl_status_cell.py').read_text(encoding='utf-8')
    nb['cells'].extend([cell('markdown','''
    ## Tự động crawl và RL — từng tiếng một model

    Qwen3-8B làm người chấm semantic EOT/NEOT/AMBIGUOUS. Crawl theo vòng → chấm reward → cập nhật policy nhỏ → giữ/bỏ theo dev. Nhãn judge không phải human gold. Đã chạy2vòng giới hạn; chưa chứng nhận90%hội thoại thực. [Kết quả và số liệu](outputs/auto_rl_v12_20261025/review.html) · [Cách chạy](README_AUTO_RL.md).
    ''',tag),cell('code',source,tag)])
    r=ColabRuntime(0);connection=r.connect();result=r.execute(source,timeout=90,echo=False);r.close();assert result['status']=='ok'
    receipt={'connection':connection,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result}
    destination=root/'notebook_receipt.json';assert not destination.exists();destination.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    nb['cells'][-1]['outputs']=result['outputs'];nb['cells'][-1]['metadata']['verification_receipt']=str(destination);path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'outputs':len(result['outputs']),'notebook_cells':len(nb['cells'])}))


if __name__=='__main__':main()
