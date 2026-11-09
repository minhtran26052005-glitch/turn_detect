"""Add an inspectable, read-only status cell to the connected notebook."""
import json
from pathlib import Path
import shutil

from build_notebook import cell


def main():
    notebook=Path('a.ipynb');nb=json.loads(notebook.read_text(encoding='utf-8'))
    root=Path('outputs/natural_v5_20261025');root.mkdir(exist_ok=True)
    code='''
    import json
    from pathlib import Path
    N5_ROOT = Path('/content/turn_detect_v2') if Path('/content/turn_detect_v2').is_dir() else Path.cwd()
    N5_NAME = 'natural_v5_train_20261025'
    def n5_read(relative):
        for base in [N5_ROOT, N5_ROOT/'outputs/natural_v5_20261025']:
            path = base/relative
            if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
        return None
    n5_status = n5_read('jobs/'+N5_NAME+'/status.json')
    n5_progress = n5_read('runs/'+N5_NAME+'/progress.json')
    n5_selection = n5_read('runs/'+N5_NAME+'/selection.json')
    n5_manifest = n5_read('data_v5/canonical_20261025/manifest.json')
    print(json.dumps({'job_status':n5_status, 'progress':n5_progress,
        'best_dev': n5_selection['best_dev'] if n5_selection else None,
        'natural_dataset_counts': n5_manifest['files'] if n5_manifest else None,
        'model_input':'TEXT_ONLY', 'sealed_test_scored':bool(n5_read('data_v5/canonical_20261025/sealed_consumed.json')),
        'note':'Dev is not held-test quality. Rerun this cell to read saved job state; it never launches/restarts training.'},ensure_ascii=False,indent=2))
    '''
    if not any('natural_v5_status' in c.get('metadata',{}).get('tags',[]) for c in nb['cells']):
        backup=root/'a_before_natural_v5.ipynb'
        if not backup.exists():shutil.copy2(notebook,backup)
        nb['cells'].extend([cell('markdown','''
        ## V5: chỉ nhận text, bổ sung hội thoại tự nhiên

        Train mới1.433mẫu từ4cuộc gọi Việt,101cuộc gọi Anh,93cuộc gọi Trung; dev293mẫu.384mẫu giữ riêng để chấm sau chọn checkpoint. Dữ liệu cũ chỉ replay từ train; pilot70mẫu V4 chỉ làm regression.

        ASR tạo chữ từ phần âm thanh đã quan sát. Mô hình nhận text + lịch sử chữ cũ; không nhận âm thanh/tương lai. Chia tập bằng call_id chính thức, giữ cả hai người nói của một cuộc gọi trong cùng tập. Nhãn khoảng dừng tự động, chưa có human gold.

        Ô dưới đọc trạng thái job đã chạy riêng trên Colab. Có thể chạy lại ô để xem tiến độ; ô này không khởi chạy hay ngắt train. Mục tiêu >90% vẫn phải đo trên tập giữ riêng sau khi chọn model.
        ''','natural_v5_status'),cell('code',code,'natural_v5_status')])
        notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    owned=next(c for c in nb['cells'] if c['cell_type']=='code' and 'natural_v5_status' in c.get('metadata',{}).get('tags',[]))
    (root/'notebook_status_cell.py').write_text(''.join(owned['source']),encoding='utf-8')
    print(json.dumps({'cells':len(nb['cells']),'tag':'natural_v5_status'}))


if __name__=='__main__':main()
