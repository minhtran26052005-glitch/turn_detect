"""Write the correction/recipe and execute an append-only notebook status cell."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
from build_notebook import cell
from colab_runtime import ColabRuntime

ROOT=Path('outputs/monolingual_20261025')


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['live','final'],default='live');a=p.parse_args()
    text='''# Mỗi ngôn ngữ một model

## Material Passport

Experiment type: CODE_TRAINING. Architecture: THREE_INDEPENDENT_MODELS. Đầu vào text-only. Bản Việt hiện có được giữ nguyên; Anh/Trung có job và trọng số riêng. Chưa chứng nhận mục tiêu hội thoại thực >90%.

Yêu cầu là mở rộng quy trình Việt sang từng tiếng. V3–V8 trước đó đã dùng một model chung hoặc student chung ba tiếng, không đúng kiến trúc riêng mà người dùng muốn. Ngưỡng theo tiếng trong một model chung không biến nó thành ba model.

Số93,55% accuracy/90,19% F1 của bản Việt QAT là trên17.833 frozen dev, nhiều NEOT là prefix của EOT. Số71,90%/41,58% được báo ở V8 là Qwen train chung trên210 điểm dừng hội thoại mới, nhãn observed-floor/VAD tự động và ASR. Đây là hai model, hai tập và hai protocol khác nhau. Không thể dùng chênh lệch đó để kết luận riêng tác động của trộn ngôn ngữ.

Đã kiểm tra đúng file Việt INT4 cũ trên cùng210 text/context hội thoại mới đã xem: accuracy32,86%, F1 EOT22,10%, ngưỡng0,47 giữ nguyên. Chỉ1cuộc gọi với nhãn tự động; không phải test độc lập mới hoặc human semantic gold. Kết quả cho thấy model riêng hiện có cũng chưa đáp ứng cách đánh giá mới. Mỗi model dùng tokenizer/codec native, nên so sánh vẫn mang tính mô tả, chưa cô lập được ảnh hưởng của ASR, loại dữ liệu hay nhãn.

## Pipeline sửa lại

- VI: giữ `models/turn_v2_distilled_qat4_20261024/onnx/`; SHA256 ONNX `9aeae3868139d5c2faa19bee34d7128a90c41ae5b0161a8f87fc039d96188412`.
- EN: `mono_en_teacher_20261025` → `mono_en_student_20261025`; chỉ dữ liệu Anh, vocabulary và QAT riêng.
- ZH: `mono_zh_teacher_20261025` → `mono_zh_student_20261025`; chỉ dữ liệu Trung, vocabulary và QAT riêng.

Hai teacher cùng khởi tạo từ mDeBERTa pretrained revision `a0484667b22365f84929a935b5e50a51f71f159d` như quy trình Việt ban đầu. Chúng có checkpoint và optimizer độc lập, không cập nhật vào nhau hoặc vào model Việt. Mỗi teacher tối đa400bước; GRU distill tối đa16epoch rồi QAT6epoch, early stopping trên dev riêng. Các giới hạn chỉ xác định ngân sách thử nghiệm, không bảo đảm90%.

EN:10.562train/1.682dev; ZH:4.380train/855dev sau lọc exact-input và bảo vệ nhóm dev. Dữ liệu gồm hội thoại thật có nhãn floor tự động và replay causal đọc/LiveKit. Ngưỡng chọn trên dev; điểm natural được báo riêng, không lấy điểm tổng làm chứng nhận hội thoại thật. Vocab fit train riêng. Giữ các token mới nhất khi vượt256token; cùng codec train/export/inference. Nhãn tự động không chuyển thành human gold.

Student tái sử dụng các hàm GRU/distill/INT4 đã kiểm tra, mỗi worker chỉ chạy trên file của một ngôn ngữ. Tên module kernel cũ có `trilingual`, nhưng không có batch, vocabulary hoặc model train chung trong campaign này; `monolingual_report.json` và `identity.json` là receipts xác định phạm vi mới.

## Chạy và kiểm tra

`python tools/monolingual_colab.py status` đọc các job, không khởi chạy lại. `python tools/monolingual_colab.py collect` chỉ thu sau khi campaign hoàn tất. Hai teacher được backup nguyên vẹn bằng `tools/backup_monolingual_teacher.py --language en|zh`.

Sau thu kết quả, `tools/finalize_monolingual.py` khóa registry/ngưỡng rồi chấm384mẫu đã xem dưới tên regression và tạo player204prefix. Không tune trên384mẫu; không gọi chúng test mới. Prefix crawl không có nhãn độc lập nên accuracy/F1=null.

`python -m turn_v2.monolingual_runtime --language vi --text "tôi muốn đặt vé đi"` chọn đúng file Việt. EN/ZH dùng đường dẫn riêng trong `outputs/monolingual_20261025/registry.json`; khi chưa có file sẽ báo pending, không tự dùng model tiếng khác.

Ô cuối notebook `monolingual_status` đọc tiến độ/kết quả kernel Colab. Các cell và kết quả cũ được giữ nguyên. [So sánh bản Việt trên hội thoại mới](outputs/monolingual_20261025/established_vi_regression.json).
'''
    report=ROOT/'report.json'
    if report.exists():
        result=json.loads(report.read_text());text+='\n## Kết quả regression sau khi khóa riêng từng model\n\n| Tiếng | Mẫu | Accuracy | F1 EOT |\n|---|---:|---:|---:|\n'
        for lang,v in result['regression_metrics'].items():text+=f"| {lang} | {v['samples']} | {v['accuracy']:.2%} | {v['eot_f1']:.2%} |\n"
        text+='\nNhãn tự động; cohort đã xem; chưa có chứng nhận90% hội thoại thực. [Bảng model riêng](outputs/monolingual_20261025/review.html) · [Player từng prefix](outputs/monolingual_20261025/crawl_review.html).\n'
    Path('README_MONOLINGUAL.md').write_text(text,encoding='utf-8')
    notebook=Path('a.ipynb');nb=json.loads(notebook.read_text(encoding='utf-8'));tag='monolingual_status'
    source=Path('tools/templates/monolingual_status_cell.py').read_text(encoding='utf-8')
    owned=[c for c in nb['cells'] if c['cell_type']=='code' and tag in c.get('metadata',{}).get('tags',[])]
    if not owned:
        backup=ROOT/'a_before_monolingual.ipynb';assert not backup.exists();shutil.copy2(notebook,backup)
        nb['cells'].extend([cell('markdown','''
        ## Ba tiếng — ba model độc lập

        Giữ bản Việt QAT đã có, train teacher → distill → QAT riêng cho Anh và Trung. Mỗi tiếng có model/tokenizer/ngưỡng riêng. Điểm dev tiền tố cũ không phải chất lượng trên hội thoại mới. [Recipe và giải thích](README_MONOLINGUAL.md).
        ''',tag),cell('code',source,tag)]);owned=[nb['cells'][-1]]
    current=owned[0];assert ''.join(current['source'])==source
    r=ColabRuntime(0);connection=r.connect()
    if a.phase=='final':
        data=report.read_bytes();encoded=base64.b64encode(data).decode('ascii')
        upload=r.execute("from pathlib import Path\nimport base64\np=Path('/content/turn_detect_v2/outputs/monolingual_20261025/report.json')\ndata=base64.b64decode("+repr(encoded)+")\nif p.exists():assert p.read_bytes()==data\nelse:p.write_bytes(data)",timeout=30,echo=False)
        assert upload['status']=='ok'
    result=r.execute(source,timeout=90,echo=False);r.close()
    receipt=ROOT/('notebook_'+a.phase+'_receipt.json');assert not receipt.exists(),'Preserve the prior phase receipt'
    receipt.write_text(json.dumps({'connection':connection,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result},ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['status']=='ok';current['outputs']=result['outputs'];current['metadata']['verification_receipt']=str(receipt)
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(json.dumps({'phase':a.phase,'kernel_execution_status':result['status'],'outputs':len(result['outputs']),'notebook_cells':len(nb['cells'])}))


if __name__=='__main__':main()
