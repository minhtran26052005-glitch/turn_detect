"""A reader-facing report from fixed results, without model execution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from turn_v2.common import normalize

JOB = ROOT / "outputs/translation_segmenter_v15d_20261026/diagnostic_test_c75_v1"
NAMES = {"vi": "Tiếng Việt", "en": "Tiếng Anh", "zh": "Tiếng Trung"}
DECISIONS = {"EOT": "Ngắt câu", "NEOT": "Chờ thêm"}


def read(name):
    return json.loads((JOB / name).read_text(encoding="utf-8"))


def number(value):
    return f"{value:,}".replace(",", ".")


def pct(value):
    return f"{100 * value:.2f}".replace(".", ",") + "%"


def esc(value):
    return html.escape(str(value), quote=True)


def visible(row):
    return '<div class="input"><span>' + esc(normalize(row["context"])) + '</span><b class="boundary" aria-label="vị trí cần xét">▏</b><mark>' + esc(normalize(row["text"])) + '</mark></div>'


def origin(row):
    url = row.get("source_url", "")
    label = esc(row.get("source_title", "Nguồn văn bản"))
    if url.startswith(("https://", "http://")) and "Wikipedia API" not in url:
        return '<details class="origin"><summary>Nguồn ví dụ</summary><a target="_blank" rel="noreferrer" href="' + esc(url) + '">' + label + "</a></details>"
    return '<details class="origin"><summary>Nguồn ví dụ</summary><p>' + label + "</p></details>"


def main():
    report = read("report.json")
    assert report["complete"] and not report["test_used_for_selection"]
    report_sha = hashlib.sha256((JOB / "report.json").read_bytes()).hexdigest()
    coverage = read("training_coverage_v1.json")
    compressed = read("vi_compression_evidence_v1.json")
    assert all(compressed["model_artifacts_verified_by_size_and_sha256"].values())
    assert compressed["new_cycle75_VI_test_scored"] is False
    train = read("train_examples_for_report_v1.json")["vi"]
    config = read("train_examples_for_report_v2.json")["config"]
    assert all(x["num_hidden_layers"] == 12 and x["hidden_size"] == 768 for x in config.values())
    rows = [json.loads(x) for x in (JOB / "review_samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    by_id = {x["example_id"]: x for x in rows}
    # These are saved outputs, including mistakes; no new scoring is performed.
    examples = [
        ("0033aa01f4dc0806ee221c8525cda5f8b446ae845d41fb989d51815f9c0a5f36", "Ngắt đúng ở cuối câu"),
        ("00707eeaf09ca6dc65122518c3df467ddc673986ee37047f2a7d30d2517b33a3", "Chờ đúng khi câu còn tiếp"),
        ("00bdf2031720ec3b0df65451c6f84a895d7e098aec468986ca139ffac2ca7c52", "Một lần ngắt quá sớm"),
        ("00848502da7ae35197e08ad9c603b71ec9d4f95e9447fb5d0ffaab3509bd2037", "Ngắt đúng ở cuối câu"),
        ("001612f527551e2fc69c6f0adcb5ebaf4ee1568aa954eb1691ab1774353afb1e", "Một lần bỏ sót chỗ ngắt"),
        ("001fc2e14f16eec4ac2bbb082f7be907887930270543afd352e1cef72e9707de", "Chờ đúng khi câu còn tiếp"),
        ("01768d1f9a8f076a3215bfb6c6f408ca9e2462654d3835a208ab8c9dea0b9cbf", "Một lần bỏ sót chỗ ngắt"),
    ]
    chosen = []
    for example_id, title in examples:
        row = by_id[example_id]
        assert row["included_in_clean_test"]
        assert row["predicted"] == ("EOT" if row["p_eot"] >= row["threshold_from_dev"] else "NEOT")
        chosen.append((row, title))

    result_rows, horizon_rows, training_rows, source_rows, model_cards, outcome_bars = [], [], [], [], [], []
    sources = {
        "vi": "FineWeb-2 tiếng Việt và Wikipedia tiếng Việt",
        "en": "FineWeb tiếng Anh và Wikipedia tiếng Anh",
        "zh": "FineWeb-2 tiếng Trung, BAAI tin tức và Wikipedia tiếng Trung",
    }
    for lang in ("vi", "en", "zh"):
        result = report["languages"][lang]
        m = result["metrics"]
        result_rows.append(f'<tr><th scope="row">{NAMES[lang]}</th><td>{number(m["samples"])}</td><td><strong>{pct(m["accuracy"])}</strong></td><td>{pct(m["eot_f1"])}</td></tr>')
        k = result["by_lookahead_words"]
        gain = f'{100 * (k["2"]["accuracy"] - k["1"]["accuracy"]):.2f}'.replace(".", ",")
        horizon_rows.append(f'<tr><th scope="row">{NAMES[lang]}</th><td>{pct(k["1"]["accuracy"])}</td><td>{pct(k["2"]["accuracy"])}</td><td>+{gain} điểm %</td></tr>')
        training_rows.append(f'<tr><th scope="row">{NAMES[lang]}</th><td>{number(coverage[lang]["rows"])}</td><td>{sources[lang]}</td></tr>')
        source_rows.append(f'<tr><th scope="row">{NAMES[lang]}</th><td>{number(m["samples"])}</td><td>{sources[lang]}</td></tr>')
        model_cards.append(f'<div class="model-card"><span class="lang-tag">{lang.upper()}</span><h3>Model {NAMES[lang].lower()}</h3><p>Học riêng dữ liệu {NAMES[lang].lower()} và có bộ trọng số riêng.</p></div>')
        outcome_bars.append(f'<div class="bar-row"><span>{NAMES[lang]}</span><div class="bar-track"><div style="width:{m["accuracy"]*100:.3f}%"></div></div><strong>{pct(m["accuracy"])}</strong></div>')
    training_examples = []
    for label in ("EOT", "NEOT"):
        row = train[label]
        explanation = "Dấu chấm gốc cho biết câu đã kết thúc." if label == "EOT" else "Dấu phẩy gốc nằm trong câu, phần sau còn tiếp."
        training_examples.append('<article class="example"><p class="eyebrow">Ví dụ dùng để học</p>' + visible(row) +
                                 '<div class="answer"><span>Đáp án dạy model</span><strong>' + DECISIONS[label] + ' <small>(' + label + ')</small></strong></div><p>' + explanation + '</p>' + origin(row) + '</article>')
    test_examples = []
    for row, title in chosen:
        correct = row["correct"]
        state = "Khớp đáp án" if correct else "Chưa khớp đáp án"
        tone = "correct" if correct else "incorrect"
        test_examples.append(f'<article class="example test-example" data-language="{row["language"]}"><div class="case-top"><span class="lang-tag">{NAMES[row["language"]]}</span><span class="case-state {tone}">{state}</span></div><h3>{title}</h3>' +
                             visible(row) + f'<p class="small">Model được nhìn thêm {row["lookahead_words"]} từ ở phần màu xanh.</p>' +
                             '<div class="output-pair"><div><span>Model trả về</span><strong>' + DECISIONS[row["predicted"]] + ' <small>(' + row["predicted"] + ')</small></strong></div><div><span>Đáp án theo văn bản gốc</span><strong>' + DECISIONS[row["label"]] + ' <small>(' + row["label"] + ')</small></strong></div></div>' + origin(row) + '</article>')
    total_test = sum(x["metrics"]["samples"] for x in report["languages"].values())
    total_train = sum(x["rows"] for x in coverage.values())
    assert total_test == 119193 and total_train == 2443206
    legacy_comparison = ""
    legacy_path = JOB / "old_dev_c75_vi_report.json"
    if legacy_path.exists():
        legacy = read("old_dev_c75_vi_report.json")
        assert legacy["complete"] and legacy["threshold_not_retuned"] and legacy["model_not_updated"]
        assert legacy["protocol"]["weights_sha256"] == report["languages"]["vi"]["weights_sha256"]
        new_old_dev = legacy["new_model_metrics"]
        old_old_dev = legacy["old_small_model_metrics"]
        assert new_old_dev["samples"] == old_old_dev["samples"] == 17833
        legacy_comparison = f'''<h2 id="old-dev-comparison">Model Việt mới khi chấm trên bộ cũ</h2>
<p>Cùng bộ <strong>17.833 mẫu cũ</strong>, giữ nguyên dữ liệu và ngưỡng quyết định đã chốt cho từng model:</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Accuracy trên bộ cũ</th><th>EOT F1 trên bộ cũ</th></tr></thead><tbody>
<tr><th scope="row">Model Việt mới, mốc 75</th><td><strong>{pct(new_old_dev["accuracy"])}</strong></td><td>{pct(new_old_dev["eot_f1"])}</td></tr>
<tr><th scope="row">Bản Việt nhỏ cũ, GRU INT4</th><td>{pct(old_old_dev["accuracy"])}</td><td>{pct(old_old_dev["eot_f1"])}</td></tr></tbody></table></div>
<p>Với cách chấm này, bản nhỏ cũ có điểm cao hơn trên bộ cũ. Model mới có <strong>{number(new_old_dev["false_eot"])} lần ngắt sớm</strong> và <strong>{number(new_old_dev["missed_eot"])} lần bỏ sót chỗ ngắt</strong>.</p>
<div class="note"><p><strong>Hai điều cần hiểu khi so sánh:</strong> bộ cũ từng được dùng chọn bản nhỏ và ngưỡng của nó. Bộ này chỉ có đoạn chữ đang nói, không có 1–2 từ sau vị trí cần ngắt như cách model mới được huấn luyện. Lần chấm này giữ nguyên phần chữ cũ, không tự thêm từ tương lai và không chỉnh ngưỡng để tăng điểm. Vì vậy kết quả đo cách xử lý đầu vào cũ; chưa tách riêng được tác động của kích thước model.</p></div>'''
        if (JOB / "legacy_comparison_validation_v1.json").exists():
            validation=read("legacy_comparison_validation_v1.json")
            audit=read("legacy_protocol_data_audit_v1.json")
            codec=read("encoder_contract_audit_v1.json")
            assert validation["comparison_status"] == "UNMATCHED_TASK_CONTRACT_NOT_VALID_FOR_MODEL_RANKING"
            assert codec["all_legacy_encodings_equal_official_new_encoder"]
            example=next(x for x in audit["ambiguous_continuation_examples_not_adjudicated"] if x["prefix"] == "liên hợp quốc được thành lập ở đâu")
            legacy_comparison=f'''<h2 id="old-dev-comparison">Phép so sánh bộ cũ chưa đúng chuẩn lookahead</h2>
<div class="note amber"><p><strong>Cần sửa cách hiểu kết quả 78,48%.</strong> Đây là kết quả thử model mới bằng đầu vào kiểu cũ, chưa phải phép chấm phù hợp cách model mới được huấn luyện. Không dùng bảng này để kết luận model mới kém hơn hoặc dữ liệu nhiều hơn làm chất lượng giảm.</p></div>
<h3>Chỗ chưa phù hợp nằm ở đâu</h3>
<div class="grid2"><article class="example"><p class="eyebrow">Model mới đã học</p><div class="input">đoạn chữ cần xét <b class="boundary">▏</b><mark>1 hoặc 2 từ sau điểm xét</mark></div><p>Phần trái là câu đang xét. Phần phải giúp xác định câu còn tiếp hay đã kết thúc.</p></article><article class="example"><p class="eyebrow">Cách đưa bộ cũ vào lần trước</p><div class="input">phần trái rỗng <b class="boundary">▏</b><mark>toàn bộ đoạn đang nói</mark></div><p>Cả 17.833 mẫu không có từ tương lai thật. Có 17.385 mẫu đưa hơn 2 từ vào ô vốn dành cho lookahead.</p></article></div>
<p>Đã kiểm tra trọng số, tokenizer, thứ tự EOT/NEOT và ngưỡng: đều khớp bản đã khóa. Input token của cả 17.833 mẫu cũng khớp mã encode chính thức. Vì vậy vấn đề đã xác nhận là <strong>vai trò của hai phần input và nhiệm vụ đang được chấm không khớp</strong>, chứ chưa phát hiện lỗi đọc model hoặc đảo nhãn.</p>
<h3>Bộ cũ có sai dữ liệu không</h3>
<p>Toàn bộ mẫu và nhãn vẫn khớp file nguồn; không có nhãn đối nghịch trên cùng input trong dev, không có input hoặc nhóm trùng train cũ theo cách kiểm hiện có. Nhưng bộ này thiếu nguồn hội thoại và phần nói tiếp theo, nên chưa thể xác nhận mọi nhãn là đáp án hội thoại do người kiểm tra.</p>
<p><strong>{number(audit["neot_matching_strict_prefix_of_same_group_eot"])} / {number(audit["labels"]["NEOT"])} mẫu NEOT</strong> khớp tiền tố của một mẫu EOT dài hơn trong cùng nhóm. Điều đó cho biết đoạn có thể được cắt từ câu dài; không bảo đảm bản thân đoạn ngắn chưa đủ ý.</p>
<article class="example"><p class="eyebrow">Ví dụ thật trong bộ cũ</p><div class="input">{esc(example["prefix"])}</div><p>Nhãn cũ: <strong>chờ thêm (NEOT)</strong>. Mẫu EOT cùng nhóm là:</p><div class="input">{esc(example["parent_eot_text"])}</div><p>Đoạn ngắn đã có thể là một câu hỏi hoàn chỉnh. Nếu người nói còn nối “và khi nào”, chờ thêm lại hợp lý. Khi không có phần sau, không thể xác định chắc ý định chỉ từ đoạn trước. Đây là sự mơ hồ cần bổ sung ngữ cảnh, không phải lý do tự đổi nhãn để tăng điểm.</p></article>
<h3>Vì sao dữ liệu nhiều hơn chưa cho điểm cao hơn</h3>
<p>Hai model đã học hai kiểu bài khác nhau: bản cũ học nhiều câu hỏi và các tiền tố, còn bản mới học ranh giới dấu câu trên bài viết với 1–2 từ nhìn thêm. Bộ cũ còn từng dùng để chọn phiên bản và ngưỡng cho bản nhỏ, nên không phải bài kiểm tra độc lập đối với nó.</p>
<p>Ở mốc 75, kho Việt có <strong>{number(coverage["vi"]["rows"])} mẫu</strong>, nhưng model mới chỉ có <strong>{number(coverage["vi"]["total_training_row_draws"])} lượt lấy mẫu để học</strong>. Tính cả lượt lặp, độ phủ khác nhau tối đa khoảng 8,14% kho hiện tại. Dữ liệu nằm trong kho nhiều hơn chưa đồng nghĩa model đã học hết chúng. Đây là khác biệt về mức học, chưa phải bằng chứng riêng rằng tăng dữ liệu làm model tệ đi.</p>
<p>Chất lượng dữ liệu mới cũng cần cải thiện. Trong phần EOT tiếng Việt của test prose đã kiểm tra, có <strong>248 / 16.777 mẫu</strong> lấy từ nhìn trước không liền nguồn do bộ tạo dữ liệu bỏ câu ngắn; viết tắt như “TP.” cũng có thể bị coi nhầm là hết câu. Những lỗi này đã có bằng chứng, nhưng chưa đủ để quy toàn bộ khoảng cách điểm cho nhãn sai.</p>
<details class="origin"><summary>Số lần chạy cũ được giữ lại để đối chiếu</summary><div class="table-wrap"><table><thead><tr><th>Model trên đầu vào kiểu cũ</th><th>Accuracy</th><th>EOT F1</th></tr></thead><tbody><tr><th>Model mới mốc 75, lệch quy ước lookahead</th><td>{pct(new_old_dev["accuracy"])}</td><td>{pct(new_old_dev["eot_f1"])}</td></tr><tr><th>Bản nhỏ cũ, đầu vào cùng kiểu đã học</th><td>{pct(old_old_dev["accuracy"])}</td><td>{pct(old_old_dev["eot_f1"])}</td></tr></tbody></table></div><p>Kết quả gốc được giữ nguyên; không chấm lại, không đổi nhãn hoặc chỉnh ngưỡng trong lần kiểm tra này.</p></details>
<div class="note"><p><strong>Cách đánh giá phù hợp tiếp theo:</strong> dùng chuỗi văn bản liên tục có phần trước và 1–2 từ sau thật cho cả hai nhãn, xác định đúng điểm cần ngắt và giữ tài liệu kiểm tra riêng. Không thể tự cắt 1–2 từ cuối của mẫu cũ làm “tương lai”, vì như vậy đã đổi vị trí mà nhãn đang chấm.</p></div>'''
    old_dev = compressed["old_dev_metrics"]
    regression = compressed["seen_single_call_automatic_regression"]["metrics"]
    size_factor = f'{compressed["deployment_size_reduction_factor"]:.1f}'.replace(".", ",")
    compression_section = f'''<section id="compressed-vi"><p class="section-label">03 · Bản tiếng Việt nhỏ đã có</p><h2>File model chỉ khoảng 177 KB</h2>
<p>Đã có một bản Việt gọn tên <strong>GRU QAT INT4</strong>. Model nhỏ học cách trả lời từ một model lớn, rồi lưu trọng số theo cách tiết kiệm dung lượng. Đầu ra vẫn là hai lựa chọn quen thuộc: <strong>ngắt câu</strong> hoặc <strong>chờ thêm</strong>.</p>
<div class="note"><p><strong>Đây là bản từ nhánh huấn luyện trước.</strong> Nó chưa phải bản nén của model Việt mốc 75 trong bảng kết quả phía trên, và chưa được chấm theo bộ test nhìn thêm 1–2 từ mới của báo cáo này.</p></div>
<div class="grid3"><div class="stat"><strong>177 KB</strong><span>file model ONNX: {number(compressed["onnx_bytes"])} byte</span></div><div class="stat"><strong>404 KB</strong><span>cả gói chạy: khoảng 0,404 MB</span></div><div class="stat"><strong>{size_factor} lần</strong><span>nhỏ hơn gói model thầy ở nhánh nén cũ</span></div></div>
<h3 style="margin-top:24px">Model được làm gọn bằng cách nào</h3>
<div class="grid3"><div class="model-card"><h3>Học từ model lớn</h3><p>Model nhỏ học từ đáp án và cách model lớn đánh giá câu. Cách này gọi là <strong>distillation</strong>, có thể hiểu là truyền lại kiến thức.</p></div><div class="model-card"><h3>Dùng bộ đọc chữ nhỏ</h3><p>Thay bằng <strong>GRU một lớp</strong> với 96 đơn vị nhớ, embedding 64 chiều và bộ từ gọn. Bản triển khai có <strong>293.090 tham số</strong>, tức các con số model đã học.</p></div><div class="model-card"><h3>Lưu trọng số bằng 4-bit</h3><p>Model được tập thích nghi với cách lưu ít bit, gọi là <strong>QAT INT4</strong>. Hai giá trị trọng số có thể được đóng trong một byte, giúp giảm dung lượng file.</p></div></div>
<p style="margin-top:18px">Gói chạy gồm model, bộ tách chữ, cấu hình và mã gọi model, tổng cộng <strong>{number(compressed["deployment_bytes"])} byte</strong>. So với gói model thầy <strong>109,61 MB</strong> của chính nhánh cũ, gói này nhỏ hơn khoảng <strong>{size_factor} lần</strong>. Gói có thể chạy bằng CPU qua ONNX Runtime.</p>
<p class="small">Dung lượng trên chưa gồm Python và thư viện đã cài. 4-bit là cách lưu trọng số; khi chạy, các phép tính vẫn dùng số 32-bit. Vì vậy dung lượng file không phải lượng RAM cần khi chạy.</p>
<h3>Chất lượng bản nhỏ đã đo được</h3>
<div class="table-wrap"><table><thead><tr><th>Bộ dữ liệu đánh giá</th><th>Số mẫu</th><th>Accuracy</th><th>EOT F1</th></tr></thead><tbody>
<tr><th scope="row">Tập cũ dùng để chọn phiên bản<br><small>Dev, cùng nhánh nén</small></th><td>{number(old_dev["samples"])}</td><td><strong>{pct(old_dev["accuracy"])}</strong></td><td>{pct(old_dev["eot_f1"])}</td></tr>
<tr><th scope="row">Các điểm trong một cuộc gọi đã xem<br><small>Nhãn tự động, chỉ dùng chẩn đoán</small></th><td>{number(regression["samples"])}</td><td>{pct(regression["accuracy"])}</td><td>{pct(regression["eot_f1"])}</td></tr>
<tr><th scope="row">Test tiếng Việt mới trong báo cáo này</th><td>39.167</td><td colspan="2">Chưa đánh giá bản nhỏ</td></tr>
</tbody></table></div>
<p>Bản nhỏ đã học từ <strong>{number(compressed["train_samples_old_branch"])} mẫu của nhánh cũ</strong>. Tập 17.833 mẫu được dùng chọn phiên bản và ngưỡng quyết định, nên số {pct(old_dev["accuracy"])} chưa chứng minh chất lượng tương tự trên dữ liệu mới. Kết quả thấp trên cuộc gọi cũng cho thấy chất lượng thay đổi nhiều khi đổi loại dữ liệu; nhãn cuộc gọi đó được tạo tự động, chưa phải đáp án do người nghe xác nhận.</p>
<div class="note amber"><p><strong>Kết luận về bản nén:</strong> đã có model Việt rất nhỏ để chạy CPU. Khả năng áp dụng cho cách tách câu nhìn thêm 1–2 từ hiện tại còn cần kiểm tra cùng giao thức. Điểm {pct(old_dev["accuracy"])} của bản nhỏ và 81,14% của model Việt mới đo trên hai bộ khác nhau, nên chưa thể dùng chúng để xếp hạng hai model.</p></div>
</section>'''
    page = '''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Báo cáo kết quả mô hình tách câu Việt Anh Trung</title>
<style>
:root{--ink:#162b42;--muted:#526579;--blue:#2359ab;--soft:#eaf2ff;--green:#14664d;--line:#d9e2ec;--paper:#fff;--bg:#f5f7fa;--amber:#8a500b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.65 "Segoe UI",Arial,sans-serif}main{max-width:1120px;margin:auto;padding:35px 24px 64px}
h1,h2,h3{line-height:1.25;letter-spacing:-.025em}h1{font-size:clamp(30px,4vw,47px);max-width:850px;margin:12px 0 18px}h2{font-size:29px;margin:0 0 20px}h3{font-size:20px;margin:10px 0 14px}p{margin:0 0 16px}a{color:var(--blue)}small,.small{font-size:14px;color:var(--muted)}
.eyebrow{font-weight:700;letter-spacing:.09em;text-transform:uppercase;font-size:12px;color:var(--blue);margin-bottom:10px}.lead{font-size:20px;max-width:920px;color:#344e67}.hero{padding:18px 0 28px}.badge{display:inline-block;border:1px solid #edcd9e;background:#fff5e5;color:var(--amber);padding:5px 13px;border-radius:30px;font-weight:650;font-size:14px}
nav{display:flex;gap:8px 20px;flex-wrap:wrap;padding:17px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-bottom:32px}nav a{text-decoration:none;font-size:15px;font-weight:600}
section{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:30px;margin:25px 0;scroll-margin-top:20px}.section-label{font-size:13px;color:var(--muted);font-weight:700;margin-bottom:7px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
.stat,.model-card{background:#f5f8fd;border:1px solid #e1e9f3;border-radius:11px;padding:19px}.stat strong{font-size:31px;display:block}.stat span{font-size:14px;color:var(--muted)}.model-card p{font-size:15px;margin-bottom:0}.lang-tag{display:inline-block;background:var(--soft);color:var(--blue);font-size:12px;font-weight:750;border-radius:5px;padding:3px 9px}
.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:16px;text-align:left}th,td{padding:13px 16px;border-bottom:1px solid var(--line);vertical-align:top}thead th{background:#eef3f9;color:#344b63;font-size:14px;font-weight:700}tbody th{font-weight:650}td strong{font-size:19px}tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
.note{border-left:4px solid var(--blue);padding:13px 18px;background:#f1f6fe;margin:20px 0}.note p:last-child{margin:0}.note.amber{border-color:#b77a2d;background:#fff7ec}.definitions{font-size:15px;color:var(--muted);margin-top:18px}.definitions p{margin-bottom:8px}
.bar-row{display:grid;grid-template-columns:110px 1fr 85px;gap:14px;align-items:center;margin:13px 0;font-size:14px}.bar-track{height:11px;border-radius:12px;background:#e7edf5;overflow:hidden}.bar-track div{height:100%;border-radius:12px;background:var(--blue)}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:23px 0}.flow>div{position:relative;background:#f1f5fa;border-radius:9px;padding:16px;font-size:15px}.flow>div:not(:last-child):after{content:'→';position:absolute;right:-12px;top:40%;font-weight:800;z-index:1;color:#6a7f97}.flow b{display:block;margin-bottom:8px}.flow p{margin:0;font-size:14px;color:var(--muted)}
.decisions{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.decision{flex:1;min-width:220px;border-radius:9px;padding:17px;background:#edf7f2;color:#16533f}.decision.wait{background:#edf2fa;color:#244d89}.decision strong{display:block;font-size:20px}.decision p{font-size:15px;margin:4px 0 0}
button{font:inherit;cursor:pointer}.switch,.filters{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.switch button,.filters button{border:1px solid #c4d1e2;background:white;color:#344e67;border-radius:7px;padding:8px 15px;font-size:15px}.switch button[aria-pressed=true],.filters button[aria-pressed=true]{background:var(--blue);color:white;border-color:var(--blue)}button:focus-visible,a:focus-visible{outline:3px solid #d58c25;outline-offset:3px}
.input{font-size:18px;line-height:1.9;background:#f2f5f9;border-radius:9px;padding:15px 17px;margin:16px 0;overflow-wrap:anywhere}.input mark{background:#ccebdd;color:#124e3b;border-radius:4px;padding:3px 5px}.boundary{color:#c65b1f;padding:0 5px}.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:14px;color:var(--muted)}.legend .green{color:var(--green);font-weight:650}.demo{border:1px solid #d1dfef;border-radius:12px;padding:20px;background:#f9fbfe}.demo .input{font-size:21px}.demo p:last-child{margin:0}
.example{border:1px solid var(--line);border-radius:11px;padding:19px;min-width:0}.example .input{font-size:16px}.answer{display:flex;flex-direction:column;gap:3px;margin:16px 0}.answer>span,.output-pair span{font-size:13px;color:var(--muted)}.answer>strong{color:var(--green)}.example p{font-size:15px}.origin{font-size:13px;color:var(--muted);margin:12px 0 0}.origin summary{cursor:pointer}.origin a{display:block;overflow-wrap:anywhere;margin-top:7px}.case-top{display:flex;gap:8px;align-items:center;justify-content:space-between}.case-state{font-size:12px;font-weight:700}.correct{color:var(--green)}.incorrect{color:#a65b13}.output-pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;border-top:1px solid var(--line);padding-top:15px}.output-pair strong{display:block;font-size:16px}.test-example[hidden]{display:none}.source-links{font-size:14px;color:var(--muted)}.source-links a{margin-right:12px}.footer{font-size:13px;color:var(--muted);padding:0 7px}.summary-list{padding-left:22px}.summary-list li{margin:9px 0}
@media(max-width:760px){main{padding:20px 12px 40px}section{padding:22px 17px}.grid3,.grid2{grid-template-columns:1fr}.flow{grid-template-columns:1fr 1fr}.flow>div:after{display:none}h2{font-size:25px}.bar-row{grid-template-columns:90px 1fr 75px;gap:8px}th,td{padding:11px 10px}.lead{font-size:18px}.output-pair{grid-template-columns:1fr}.stat strong{font-size:27px}}
@media print{body{background:white;font-size:11pt}main{max-width:none;padding:0}nav,.switch,.filters{display:none}section{break-inside:avoid;border-radius:0;margin:15px 0;padding:18px}.example{break-inside:avoid}.test-example[hidden]{display:block}.grid2{display:block}.grid2 .example{margin:12px 0}.hero{padding-top:0}a{color:inherit;text-decoration:none}}
</style></head><body><main>
<header class="hero"><p class="eyebrow">Kết quả đánh giá mô hình tách câu</p><h1>Tách câu từ chữ thường không có dấu câu</h1><p class="lead">Ba model riêng cho tiếng Việt, Anh và Trung đọc đoạn chữ đã có, nhìn thêm 1 hoặc 2 từ rồi quyết định <strong>ngắt câu</strong> hay <strong>chờ thêm</strong>.</p><span class="badge">Kết quả hiện tại chưa đạt mục tiêu 90%</span></header>
<nav aria-label="Các phần báo cáo"><a href="#results">Kết quả cuối</a><a href="#old-dev-comparison">Kiểm tra phép so sánh</a><a href="#model">Model nhận và trả gì</a><a href="#compressed-vi">Bản Việt nhỏ</a><a href="#lookahead">Lookahead</a><a href="#train">Dữ liệu học</a><a href="#test">Dữ liệu test</a><a href="#outputs">Ví dụ đầu ra</a></nav>

<section id="results"><p class="section-label">01 · Kết quả cuối</p><h2>Đúng khoảng 81–85% vị trí được kiểm tra</h2>
<div class="grid3"><div class="stat"><strong>3</strong><span>model riêng theo ngôn ngữ</span></div><div class="stat"><strong>119.193</strong><span>lượt kiểm tra sau khi loại trùng</span></div><div class="stat"><strong>1.218</strong><span>tài liệu nguồn trong tập được chấm</span></div></div>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Lượt kiểm tra</th><th>Tỷ lệ quyết định đúng<br><small>Accuracy</small></th><th>Điểm tìm chỗ ngắt câu<br><small>EOT F1</small></th></tr></thead><tbody>RESULT_ROWS</tbody></table></div>
OUTCOME_BARS
<div class="definitions"><p><strong>Accuracy:</strong> tỷ lệ model trả lời đúng, tính cả “ngắt câu” và “chờ thêm”.</p><p><strong>F1:</strong> điểm cân bằng giữa ngắt đúng và tìm đủ các chỗ cần ngắt. F1 càng cao càng tốt; đây không phải tỷ lệ mẫu trả lời đúng.</p></div>
<div class="note amber"><p><strong>Tiếng Trung cần chú ý nhất.</strong> Accuracy 84,20% nhưng F1 chỉ 67,18%, vì phần lớn mẫu Trung có đáp án “chờ thêm”. Khi model chọn “ngắt câu”, khoảng <strong>58,19%</strong> lựa chọn khớp đáp án. Vì vậy không thể đọc riêng con số 84,20% để kết luận chất lượng ngắt câu đã cao.</p></div>
<p class="small">Các con số này thuộc bản model đã chốt tại mốc 75. Các lần huấn luyện mới hơn chưa được phản ánh trong bảng.</p></section>

<section id="legacy-comparison">LEGACY_DEV_COMPARISON</section>

<section id="model"><p class="section-label">02 · Cấu trúc và nhiệm vụ</p><h2>Đầu vào là chữ, đầu ra là một quyết định</h2>
<p>Ví dụ, ứng dụng đang có một dòng chữ liên tục từ nhận dạng giọng nói. Tại một vị trí đang xét, model cần quyết định có thể kết thúc câu ở đó hay nội dung vẫn còn tiếp. Quyết định này có thể dùng để gom đoạn trước khi gửi sang bộ dịch.</p>
<div class="grid3">MODEL_CARDS</div>
<p class="small" style="margin-top:14px">Ngôn ngữ được chọn trước để gọi model tương ứng. Ba model cùng dùng nền mDeBERTa v3 base, nhưng học và có trọng số riêng; báo cáo này đánh giá model đọc chữ, không đánh giá âm thanh.</p>
<div class="flow" aria-label="Cấu trúc bên trong model"><div><b>1. Văn bản đầu vào</b><p>Đoạn bên trái và 1–2 từ tiếp theo, toàn bộ chữ thường, không có dấu câu.</p></div><div><b>2. Bộ tách từ thành mảnh</b><p>Chuyển chữ thành các đơn vị nhỏ để model đọc, gọi là token.</p></div><div><b>3. Bộ hiểu văn bản</b><p>mDeBERTa gồm 12 lớp xử lý để đọc quan hệ giữa các từ.</p></div><div><b>4. Bộ quyết định</b><p>Tính điểm rồi chọn một trong hai kết quả bên dưới.</p></div></div>
<div class="decisions"><div class="decision"><strong>Ngắt câu · EOT</strong><p>Có thể kết thúc câu ở vị trí đang xét.</p></div><div class="decision wait"><strong>Chờ thêm · NEOT</strong><p>Chưa kết thúc câu; tiếp tục nhận thêm chữ.</p></div></div>
<p>Model không tự dịch hoặc viết lại câu. Nó cũng không tự chèn một dấu chấm vào nội dung. Kết quả chính là nhãn <strong>EOT</strong> hoặc <strong>NEOT</strong> cho vị trí cần xét.</p>
<div class="note"><p>Chữ hoa và dấu câu nguồn không đi vào model. Dấu tiếng Việt vẫn giữ nguyên. Với đoạn dài, model giữ phần cuối của đoạn trước trong giới hạn 256 token, nên không đọc toàn bộ một bài viết dài.</p></div></section>

COMPRESSION_SECTION

<section id="lookahead"><p class="section-label">04 · Nhìn trước 1 hoặc 2 từ</p><h2>Lookahead là chờ thêm một chút chữ để quyết định</h2>
<p><strong>Lookahead 1</strong> nghĩa là nhìn thêm 1 từ sau vị trí đang xét. <strong>Lookahead 2</strong> nghĩa là nhìn thêm 2 từ. Vị trí cần quyết định vẫn giữ nguyên; phần chữ được nhìn thêm giúp model biết ý phía sau còn nối với phần trước hay đã mở câu mới.</p>
<div class="demo"><p class="eyebrow">Minh họa cách đưa chữ vào model</p><div class="switch" role="group" aria-label="Chọn số từ nhìn thêm"><button type="button" data-horizon="1" aria-pressed="true">Nhìn thêm 1 từ</button><button type="button" data-horizon="2" aria-pressed="false">Nhìn thêm 2 từ</button></div><div class="input"><span>tôi đã hoàn thành công việc</span><b class="boundary">▏</b><mark id="future-words">ngày</mark></div><div class="legend"><span>Chữ bên trái: phần đã có</span><span class="green">Phần xanh: từ được nhìn thêm</span></div><p id="horizon-caption" style="margin-top:14px">Model được nhìn thêm “ngày”, nhưng vẫn quyết định có ngắt sau “công việc” hay không.</p><p class="small">Vạch đứng chỉ đánh dấu vị trí để người đọc dễ hiểu, không phải dấu câu đưa vào model. Đây là ví dụ minh họa cách đóng gói, không phải một dự đoán mới.</p></div>
<p style="margin-top:20px">Nhìn thêm 2 từ thường có thêm thông tin, đồng thời phải chờ nhận thêm 1 từ so với cách dùng 1 từ. Đây không phải thời gian cố định theo giây. Với tiếng Trung, một từ sau khi tách có thể gồm nhiều ký tự.</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Nhìn thêm 1 từ<br><small>Accuracy</small></th><th>Nhìn thêm 2 từ<br><small>Accuracy</small></th><th>Chênh lệch</th></tr></thead><tbody>HORIZON_ROWS</tbody></table></div>
<p class="small">Trong tập test này, 2 từ cho tỷ lệ đúng cao hơn ở cả ba ngôn ngữ. Điều đó không có nghĩa 2 từ luôn đúng hơn trên từng ví dụ.</p></section>

<section id="train"><p class="section-label">05 · Dữ liệu dùng để học</p><h2>Model học từ các ví dụ “ngắt” và “chờ”</h2>
<p>Dữ liệu là các bài viết công khai. Khi tạo bài học, dấu kết thúc câu trong văn bản gốc được dùng làm đáp án “ngắt câu”; dấu phẩy, chấm phẩy hoặc dấu nội bộ được dùng làm đáp án “chờ thêm”. Sau đó dấu câu được bỏ và chữ được chuyển thành chữ thường trước khi cho model đọc.</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Số mẫu trong kho để lấy ra học</th><th>Nguồn</th></tr></thead><tbody>TRAINING_ROWS</tbody></table></div>
<p class="small">Kho ở thời điểm chốt model có tổng cộng 2.443.206 mẫu. Số này là lượng dữ liệu sẵn có để lấy mẫu huấn luyện, không có nghĩa model đã học hết mọi mẫu.</p>
<h3>Hai ví dụ thật trong dữ liệu huấn luyện</h3><p class="small">Phần màu xanh là từ nhìn thêm. Đáp án nằm riêng ở dưới, không được đưa vào phần chữ mà model nhìn thấy.</p><div class="grid2">TRAINING_EXAMPLES</div>
<p style="margin-top:18px">Các ví dụ của tiếng Anh và Trung dùng cùng cách biểu diễn: một đoạn bên trái, 1–2 từ phía sau và một đáp án ngắt hoặc chờ.</p></section>

<section id="test"><p class="section-label">06 · Dữ liệu dùng để kiểm tra</p><h2>Test dùng những tài liệu được giữ riêng</h2>
<p>Tập test lấy từ các tài liệu khác với tài liệu huấn luyện trong campaign, từ những nhóm nguồn dưới đây. Mỗi mẫu cũng có phần chữ bên trái và 1–2 từ nhìn thêm, nhưng đáp án được giữ lại để so với kết quả model trả về.</p>
<div class="table-wrap"><table><thead><tr><th>Ngôn ngữ</th><th>Mẫu được chấm</th><th>Nguồn dữ liệu</th></tr></thead><tbody>SOURCE_ROWS</tbody></table></div>
<p>Những mẫu trùng nội dung đầu vào với train, tập chọn cấu hình hoặc các bộ giữ riêng khác đã được lọc khỏi điểm chính. Tổng cộng <strong>1.882 mẫu bị loại</strong>, còn <strong>119.193 mẫu để chấm</strong>. Một ranh giới có thể có cả phiên bản nhìn thêm 1 từ và 2 từ, nên đây không phải 119.193 câu khác nhau.</p>
<p class="source-links">Thông tin nguồn: <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2">FineWeb-2</a><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">FineWeb</a><a href="https://huggingface.co/datasets/BAAI/IndustryCorpus2_news_media">BAAI tin tức</a><span>và Wikipedia theo ngôn ngữ.</span></p>
<div class="note amber"><p><strong>Điểm số đo khả năng tìm lại ranh giới dấu câu trên văn bản.</strong> Đáp án được tạo tự động từ dấu câu, có thể sai ở một số trường hợp như viết tắt. Tập này chưa phải hội thoại thật được người nghe audio gán nhãn, và kiểm trùng chưa loại hết mọi cách diễn đạt gần giống nhau.</p></div></section>

<section id="outputs"><p class="section-label">07 · Đầu ra thực tế</p><h2>Model trả lời như thế nào trên test</h2>
<p>Dưới đây là các đầu vào và kết quả đã được chấm, gồm cả ví dụ đúng và sai. “Đáp án” nghĩa là nhãn tạo từ dấu câu gốc. Các ví dụ được chọn để dễ đọc; điểm tổng thể được tính trên toàn bộ tập test ở đầu báo cáo.</p>
<div class="filters" role="group" aria-label="Chọn ngôn ngữ của ví dụ"><button type="button" data-filter="vi" aria-pressed="true">Tiếng Việt</button><button type="button" data-filter="en" aria-pressed="false">Tiếng Anh</button><button type="button" data-filter="zh" aria-pressed="false">Tiếng Trung</button><button type="button" data-filter="all" aria-pressed="false">Xem tất cả</button></div>
<div class="grid2" id="test-examples">TEST_EXAMPLES</div></section>

<section id="conclusion"><h2>Kết luận về chất lượng hiện tại</h2><ul class="summary-list"><li>Model đã có khả năng phân biệt chỗ nên ngắt và chỗ nên chờ, nhưng cả ba ngôn ngữ <strong>chưa đạt mục tiêu 90% trên tập test mới</strong>.</li><li>Tiếng Anh có kết quả tốt nhất trong lần test này. Tiếng Trung còn nhiều quyết định ngắt chưa đúng dù accuracy tổng thể khá cao.</li><li>Nhìn thêm 2 từ cho kết quả tổng thể tốt hơn nhìn thêm 1 từ trong bộ test đã chấm, với đánh đổi là phải chờ thêm một từ.</li><li>Đã có bản Việt nhỏ với cả gói khoảng <strong>404 KB</strong> từ nhánh trước. Chất lượng của bản nhỏ trên cách test tách câu mới hiện chưa được xác nhận.</li></ul></section>
<footer class="footer"><p>Báo cáo dành cho người đọc không chuyên kỹ thuật · Test mới của model mốc 75 và bản Việt nén từ nhánh trước · Ngày 27 tháng 10 năm 2026.</p></footer>
</main><script>
document.querySelectorAll('[data-horizon]').forEach(button=>button.addEventListener('click',()=>{
 const k=button.dataset.horizon;document.querySelectorAll('[data-horizon]').forEach(x=>x.setAttribute('aria-pressed',String(x===button)));
 const words=k==='1'?'ngày':'ngày mai';document.getElementById('future-words').textContent=words;
 document.getElementById('horizon-caption').textContent='Model được nhìn thêm “'+words+'”, nhưng vẫn quyết định có ngắt sau “công việc” hay không.';
}));
function filterCases(language){document.querySelectorAll('[data-filter]').forEach(x=>x.setAttribute('aria-pressed',String(x.dataset.filter===language)));document.querySelectorAll('.test-example').forEach(x=>x.hidden=language!=='all'&&x.dataset.language!==language);}
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>filterCases(button.dataset.filter)));filterCases('vi');
</script></body></html>'''
    for key, value in {
        "RESULT_ROWS": "".join(result_rows), "OUTCOME_BARS": "".join(outcome_bars),
        "MODEL_CARDS": "".join(model_cards), "HORIZON_ROWS": "".join(horizon_rows),
        "TRAINING_ROWS": "".join(training_rows), "SOURCE_ROWS": "".join(source_rows),
        "TRAINING_EXAMPLES": "".join(training_examples), "TEST_EXAMPLES": "".join(test_examples),
        "COMPRESSION_SECTION": compression_section,
        "LEGACY_DEV_COMPARISON": legacy_comparison,
    }.items():
        assert page.count(key) == 1
        page = page.replace(key, value)
    original = JOB / "review_complete_v3.html"
    technical = JOB / "review_technical_v3.html"
    if not technical.exists():
        original_bytes = original.read_bytes()
        expected = read("reviewed_analysis_receipt_v3.json")["files_sha256"]["review_complete_v3.html"]
        assert hashlib.sha256(original_bytes).hexdigest() == expected
        technical.write_bytes(original_bytes)
    raw = page.encode("utf-8")
    assert len(raw) < 5 * 1024**2 and "\ufffd" not in page
    for name in ("bao_cao_ket_qua.html", "review_complete_v3.html"):
        (JOB / name).write_bytes(raw)
    receipt = {"schema": 4, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "audience": "Vietnamese nontechnical reader", "raw_report_sha256": report_sha,
               "model_inference_executed": False, "fixed_test_examples": [x[0]["example_id"] for x in chosen],
               "training_examples_are_actual_optimizer_drawn": True,
               "technical_original_preserved": "review_technical_v3.html",
               "technical_original_sha256": hashlib.sha256(technical.read_bytes()).hexdigest(),
               "compression_section_source": "vi_compression_evidence_v1.json",
               "compressed_model_is_prior_branch_not_cycle75": True,
               "new_model_on_legacy_dev_included": bool(legacy_comparison),
               "legacy_comparison_not_valid_for_model_ranking": (JOB / "legacy_comparison_validation_v1.json").exists(),
               "files_sha256": {name: hashlib.sha256((JOB / name).read_bytes()).hexdigest() for name in ("bao_cao_ket_qua.html", "review_complete_v3.html", "train_examples_for_report_v1.json", "train_examples_for_report_v2.json", "vi_compression_evidence_v1.json", "old_dev_c75_vi_report.json", "legacy_comparison_validation_v1.json", "legacy_protocol_data_audit_v1.json", "encoder_contract_audit_v1.json")}}
    (JOB / "plain_report_receipt_v4.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_bytes": len(raw), "test_samples": total_test, "train_pool_rows": total_train,
                      "real_test_examples": len(chosen), "raw_report_sha256_unchanged": report_sha}))


if __name__ == "__main__":
    main()
