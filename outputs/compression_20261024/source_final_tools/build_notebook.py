"""Build a.ipynb with embedded source, Colab checks, and two controlled training routes."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, tag=None):
    source = textwrap.dedent(source).strip() + "\n"
    value = {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12],
             "metadata": {"tags": [tag]} if tag else {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def source_bundle():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*sorted((ROOT / "turn_v2").glob("*.py")),
                     *sorted((ROOT / "tests").glob("*.py")), ROOT / "requirements-colab.txt"]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    package = base64.b64encode(buffer.getvalue()).decode("ascii")
    checksum = hashlib.sha256(buffer.getvalue()).hexdigest()
    return package, checksum


def crawl_predict_cells():
    return [cell("markdown", """
        ## Crawl và dự đoán ngay, rồi Qwen xét một số mẫu

        Luồng này ghi từng dự đoán khi crawl được mẫu, không tự cập nhật trọng số và không phải RL.
        Qwen xét cả mẫu gần ngưỡng lẫn mẫu EOT có xác suất cao trên đoạn cắt; nhãn chấp nhận vẫn là silver.
        Đặt `PREDICT_MODEL_DIR` trỏ model train đầy đủ khi có. Nếu chỉ có checkpoint smoke, output được đánh dấu chạy thử.
        Muốn RL sau này cần thêm state/action/reward cho việc chờ tiếp hay kết thúc lượt, với tham chiếu thời điểm kết thúc thật.
    """), cell("code", """
        from datetime import datetime, timedelta, timezone
        PREDICT_MODEL_DIR = None  # Ví dụ: ROOT/'runs/gold_v1/onnx'
        if PREDICT_MODEL_DIR is None:
            if globals().get('completed_runs'):
                chosen = max(completed_runs,key=lambda item:item[1]['joint_score'])[0]
                PREDICT_MODEL_DIR = chosen/'onnx'
            elif globals().get('SMOKE_DIR') and (SMOKE_DIR/'onnx').exists():
                PREDICT_MODEL_DIR = SMOKE_DIR/'onnx'
            else:
                raise ValueError('Đặt PREDICT_MODEL_DIR trỏ một checkpoint đã có.')
        CRAWL_TAG = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
        CRAWL_OUTPUT = ROOT/'outputs'/f'crawl_predict_{CRAWL_TAG}'
        contract = json.loads((Path(PREDICT_MODEL_DIR)/'turn_detector_config.json').read_text())
        arguments = ['--model',PREDICT_MODEL_DIR,'--output',CRAWL_OUTPUT,'--limit',12,
                     '--protected-data',DATA,'--legacy-test',ROOT/'test.jsonl','--review-budget',4,
                     '--qwen-review','--teacher-model',QWEN_MODEL]
        if contract.get('smoke_only'):
            print('Checkpoint SMOKE: chỉ kiểm tra pipeline, chưa phải model đạt 90%.')
            arguments += ['--allow-smoke']
        run_module('turn_v2.crawl_predict',*arguments)
        from IPython.display import HTML, display
        display(HTML((CRAWL_OUTPUT/'results.html').read_text()))
        print('Dự đoán và audit:',CRAWL_OUTPUT)
    """, "crawl_predict")]


def build():
    package, checksum = source_bundle()
    cells = [cell("markdown", """
        # Turn detector V2 — train trên Colab L4

        Mục tiêu: **accuracy ≥90% và F1 EOT ≥90%** trên holdout đã khóa, với ngưỡng chọn từ dev.
        Đây là mục tiêu cần đo sau khi train đầy đủ, không phải điểm đã đạt.

        - Hướng train: mDeBERTa đa ngôn ngữ, AdamW, warmup/cosine, early stopping, gradient checkpointing, FP32/TF32 mặc định.
        - Hướng data: crawl có nguồn → Qwen 4B chạy trên GPU → loại mơ hồ/bất đồng → silver chỉ thêm vào train với weight 0,5.
        - Cùng chuẩn tiền xử lý khi train, dự đoán và export. Test cũ đã được xem ở baseline, không dùng để chốt mốc mới.

        Trong VS Code chọn kernel **Colab GPU L4**. Chạy các ô từ trên xuống.
        Mặc định notebook chạy smoke 3 bước để kiểm tra hệ thống; bật các cờ ở ô cấu hình để train đầy đủ.
    """), cell("code", """
        import json, os, pathlib, platform, subprocess, sys
        import torch
        assert pathlib.Path('/content').is_dir() and sys.platform.startswith('linux'), 'Chọn kernel Colab, không chọn Python Windows.'
        assert torch.cuda.is_available(), 'Runtime chưa có CUDA GPU.'
        print(json.dumps({'python':platform.python_version(), 'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(0), 'cuda':torch.cuda.is_available(),
            'vram_gib':round(torch.cuda.get_device_properties(0).total_memory/2**30,2)}, indent=2))
    """, "connection"), cell("code", """
        from datetime import datetime, timedelta, timezone
        from pathlib import Path
        ROOT = Path('/content/turn_detect_v2')
        DATA = ROOT / 'data_v2/frozen'
        # Đổi RUNS_ROOT sang thư mục Drive đã mount nếu cần giữ checkpoint qua lần ngắt runtime.
        RUNS_ROOT = ROOT / 'runs'
        RUN_TAG = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S')
        BASE_MODEL = 'microsoft/mdeberta-v3-base'
        QWEN_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
        RUN_SMOKE = True
        RUN_DATA_ROUTE = True
        RUN_FULL_TRAIN = False
        RUN_FINAL_HOLDOUT = False
        EXPERIMENTS = ['gold']  # So sánh hai hướng: ['gold', 'gold_silver'], cùng split/seed.
        EXISTING_RUN_DIRS = []  # Có thể đặt các thư mục run đã train để chọn/evaluate sau khi reconnect.
        QWEN_LIMIT = 4  # Smoke; tăng lên khi chuẩn bị augmentation cho train đầy đủ.
        print({'root':str(ROOT), 'run_tag':RUN_TAG, 'full_train':RUN_FULL_TRAIN, 'experiments':EXPERIMENTS})
    """, "config")]
    bootstrap = """
        import base64, hashlib, io, zipfile
        package = base64.b64decode(PACKAGE_LITERAL)
        assert hashlib.sha256(package).hexdigest() == CHECKSUM_LITERAL
        ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            for name in archive.namelist():
                assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())
            archive.extractall(ROOT)
        sys.path.insert(0, str(ROOT))
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'  # Public models; no Colab Secrets popup in VS Code.
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        def run_module(module, *arguments):
            command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env={**os.environ, 'OMP_NUM_THREADS':'2'}) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        print('Mã nguồn đã giải nén và kiểm tra SHA-256:', ROOT)
    """.replace("PACKAGE_LITERAL", repr(package)).replace("CHECKSUM_LITERAL", repr(checksum))
    cells += [cell("code", bootstrap, "bootstrap"), cell("code", """
        import importlib.metadata
        required = {'transformers':'5.20.1','datasets':'4.8.5','accelerate':'1.15.0',
                    'sentencepiece':'0.2.2','safetensors':'0.8.0','onnx':'1.19.1','onnxruntime':'1.23.2'}
        installed = {}
        for name in required:
            try: installed[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: installed[name] = None
        if any(installed[name] != version for name,version in required.items()):
            subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-colab.txt')],check=True)
        print({name:importlib.metadata.version(name) for name in required})
    """, "dependencies"), cell("markdown", """
        ## Dữ liệu và split cố định

        Code và dữ liệu đã được đồng bộ tới runtime hiện tại. Khi dùng runtime mới, chạy `python tools/sync_colab.py`
        ở Windows sau khi kết nối lại kernel; hoặc upload `colab/turn_detect_v2_bundle.zip` vào `/content`.
        ZIP đầy đủ chứa cả `train.jsonl`, `test.jsonl` và split đã khóa.

        Không chia ngẫu nhiên từng đoạn cắt của cùng câu nguồn. Mã ưu tiên `source_id`; với dữ liệu cũ,
        nhóm được suy từ câu EOT và prefix theo thứ tự, rồi nối các prefix dài. Đây là suy đoán vì dữ liệu cũ thiếu ID hội thoại.
        Các nhóm trùng test cũ và nhãn mâu thuẫn được cách ly; không đổi nhãn dev/holdout bằng Qwen.
    """), cell("code", """
        if not (ROOT/'train.jsonl').exists() or not (ROOT/'test.jsonl').exists():
            bundle = Path('/content/turn_detect_v2_bundle.zip')
            if bundle.exists():
                with zipfile.ZipFile(bundle) as archive:
                    for name in archive.namelist():
                        assert (ROOT/name).resolve().is_relative_to(ROOT.resolve())
                    archive.extractall(ROOT)
        assert (ROOT/'train.jsonl').exists() and (ROOT/'test.jsonl').exists(), 'Upload ZIP đầy đủ vào /content hoặc chạy tools/sync_colab.py từ Windows.'
        run_module('turn_v2.prepare', '--train', ROOT/'train.jsonl', '--legacy-test', ROOT/'test.jsonl', '--output', DATA)
        manifest = json.loads((DATA/'manifest.json').read_text())
        print(json.dumps(manifest['splits'], ensure_ascii=False, indent=2))
    """, "prepare"), cell("code", """
        tested = subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=ROOT,
                                check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(tested.stdout)
    """, "tests"), cell("markdown", """
        ## Hướng dữ liệu: crawl + Qwen

        Bật `RUN_DATA_ROUTE=True` ở ô cấu hình. Qwen 4B dùng GPU nên chạy riêng trước encoder.
        Câu gốc và đoạn cắt đều là ứng viên chưa có nhãn; Qwen được phép chọn `AMBIGUOUS`.
        Hai góc hỏi cùng model vẫn có sai số tương quan, nên nhãn được giữ là silver và không thay thế gold người duyệt.
        `audit.jsonl` giữ cả hai câu trả lời; source URL được giữ để truy nguồn.
        Wikipedia là nguồn văn xuôi ban đầu, chưa đại diện đầy đủ cho hội thoại thật.
    """), cell("code", """
        SILVER = None
        if RUN_DATA_ROUTE:
            candidates = ROOT/'data_v2'/f'candidates_{RUN_TAG}.jsonl'
            run_module('turn_v2.crawl','--output',candidates,'--max-candidates',QWEN_LIMIT)
            teacher_dir = ROOT/'data_v2'/f'qwen_{RUN_TAG}'
            run_module('turn_v2.label_qwen','--input',candidates,'--output',teacher_dir,
                       '--model',QWEN_MODEL,'--limit',QWEN_LIMIT)
            labeling = json.loads((teacher_dir/'labeling_report.json').read_text())
            assert labeling['accepted'] > 0, 'Qwen chưa tạo nhãn đủ tiêu chí; xem audit trước khi chạy hướng silver.'
            SILVER = teacher_dir/'silver.jsonl'
        else:
            print('Bỏ qua crawl/label trong lượt này; để bật: RUN_DATA_ROUTE=True.')
    """, "data_route"), cell("markdown", """
        ## Smoke test: train → save/reload → ONNX parity

        Smoke có 32 mẫu train nhãn sẵn (thêm tối đa 8 silver khi bật hướng data), 16 mẫu dev và 3 bước optimizer, không mở holdout.
        Các điểm số smoke không dùng để đánh giá chất lượng hoặc xác nhận đạt 90%.
        Export FP32 kiểm tra chênh xác suất PyTorch/ONNX; INT8 cần một lượt kiểm chứng riêng.
    """), cell("code", """
        SMOKE_DIR = RUNS_ROOT/f'smoke_{RUN_TAG}'
        if RUN_SMOKE:
            smoke_arguments = ['--data',DATA,'--model',BASE_MODEL,'--output',SMOKE_DIR,
                               '--smoke','--batch-size',4,'--max-steps',3,'--export']
            if SILVER is not None:
                smoke_arguments += ['--silver',SILVER,'--legacy-test',ROOT/'test.jsonl']
            run_module('turn_v2.train',*smoke_arguments)
            smoke_report = json.loads((SMOKE_DIR/'training_report.json').read_text())
            assert smoke_report['optimizer_steps'] == 3 and not smoke_report['holdout_evaluated']
            assert smoke_report['export']['passed']
            print('SMOKE PASSED: GPU train, checkpoint reload và ONNX parity.')
    """, "smoke"), cell("markdown", """
        ## Train đầy đủ và so sánh hai hướng

        Bật `RUN_FULL_TRAIN=True`. Mặc định train 3 epoch, learning rate encoder `2e-5`, head `1e-4`,
        batch 16 × tích lũy 2, FP32/TF32, warmup 6%, cosine, gradient clipping và early stopping.
        FP32 là cấu hình đã được kiểm tra; BF16 từng tạo đầu ra không hữu hạn trong smoke trên runtime này.

        Chạy `EXPERIMENTS=['gold','gold_silver']` cùng `RUN_DATA_ROUTE=True` để so sánh cùng split/seed.
        Chọn checkpoint và ngưỡng bằng dev; chưa mở holdout ở bước này. Khi GPU hết bộ nhớ, giảm batch và tăng accumulation để giữ batch hiệu dụng 32.
        Resume dùng cùng thư mục run: thêm `--resume <run>/checkpoints/checkpoint-N` khi chạy CLI; code kiểm tra data/model/seed/settings/source trước khi resume.
    """), cell("code", """
        completed_runs = []
        if RUN_FULL_TRAIN:
            from huggingface_hub import model_info
            base_revision = model_info(BASE_MODEL,token=False).sha
            assert EXPERIMENTS and set(EXPERIMENTS) <= {'gold','gold_silver'}
            assert 'gold_silver' not in EXPERIMENTS or SILVER is not None, 'Bật RUN_DATA_ROUTE hoặc đặt SILVER trỏ file silver đã gán nhãn.'
            for variant in EXPERIMENTS:
                output = RUNS_ROOT/f'{variant}_{RUN_TAG}'
                arguments = ['--data',DATA,'--model',BASE_MODEL,'--output',output,
                             '--revision',base_revision,'--batch-size',16,'--accumulation',2,'--epochs',3,'--seed',42,'--export']
                if variant == 'gold_silver':
                    arguments += ['--silver',SILVER,'--legacy-test',ROOT/'test.jsonl']
                run_module('turn_v2.train',*arguments)
                report = json.loads((output/'training_report.json').read_text())
                completed_runs.append((output, report['dev_selected_threshold']))
            print(json.dumps([{'run':str(path), **metrics} for path,metrics in completed_runs],indent=2))
        else:
            for existing in EXISTING_RUN_DIRS:
                output = Path(existing)
                report = json.loads((output/'training_report.json').read_text())
                assert report['mode'] == 'FULL_DEV_SELECTION'
                completed_runs.append((output,report['dev_selected_threshold']))
            print('Để train đầy đủ: RUN_FULL_TRAIN=True. Chưa có kết quả 90% từ smoke.')
    """, "full_train"), cell("markdown", """
        ## Chốt một model và đánh giá holdout một lần

        Bật `RUN_FINAL_HOLDOUT=True` sau khi chạy train đầy đủ. Chọn model có `min(accuracy,F1 EOT)` cao nhất trên dev.
        Nếu dev chưa đạt cả hai mốc 90%, holdout được giữ kín. Nếu đã mở holdout, không dùng điểm đó để chọn model/threshold tiếp theo.
        Kết quả chỉ xác nhận trên nhãn được cung cấp và cách nhóm heuristic; muốn khẳng định trên hội thoại thật cần gold mới do người duyệt.
    """), cell("code", """
        if RUN_FINAL_HOLDOUT:
            assert completed_runs, 'Train đầy đủ trước khi mở holdout.'
            chosen, dev_metrics = max(completed_runs,key=lambda item:item[1]['joint_score'])
            run_module('turn_v2.evaluate','--model',chosen/'best','--data',DATA,'--output',chosen/'final')
            final = json.loads((chosen/'final/holdout_report.json').read_text())
            print(final['status'],final['metrics'])
        else:
            print('Holdout chưa được mở.')
    """, "holdout"), cell("code", """
        if RUN_SMOKE and (SMOKE_DIR/'onnx/parity_report.json').exists():
            run_module('turn_v2.predict','--model',SMOKE_DIR/'onnx','--text','tôi muốn đặt vé đi hà nội')
        print('Checkpoint/report:', RUNS_ROOT)
        print('Tải model best và report qua bảng Colab Contents của VS Code. /content sẽ mất khi runtime bị xóa.')
    """, "inference"), cell("markdown", """
        Nguồn chính: [mDeBERTa V3](https://huggingface.co/microsoft/mdeberta-v3-base),
        [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
        [Transformers Trainer](https://huggingface.co/docs/transformers/v5.20.0/main_classes/trainer),
        [Colab extension](https://github.com/googlecolab/colab-vscode).
    """)]
    cells[-1:-1] = crawl_predict_cells()
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)",
        "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13.15"},
        "colab": {"name": "a.ipynb", "provenance": []}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5}
    destination = ROOT / "a.ipynb"
    backup = ROOT / "backups" / "a.before_v2.ipynb"
    if destination.exists() and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
    for index, entry in enumerate(cells):
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), f"a.ipynb:cell{index}", "exec")
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {destination}: {len(cells)} cells, embedded source SHA-256 {checksum}")


if __name__ == "__main__":
    build()
