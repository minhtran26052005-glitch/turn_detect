"""Score trilingual teacher checkpoints on sentence-prefix / terminal-period behavior.

Runs model inference on the already-connected Colab CPU. It never touches the
training worker, GPU, teacher weights on Windows, training data, or optimizer.
The labels are a punctuation-rule diagnostic over crawled prose, not dialogue
gold or a replacement for the natural observed-floor evaluation.
"""
from __future__ import annotations

import base64
from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote

from colab_runtime import ColabRuntime
from storage_guard import ensure_free_space
from teacher_first_colab import campaign_runtime


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "teacher_first_v14_20261026"
SOURCE = ROOT / "outputs" / "trilingual_stream_20261024" / "crawl" / "paragraphs.jsonl"
REMOTE_ROOT = "/content/turn_detect_v2"
LANGUAGES = ("vi", "en", "zh")
TERMINAL = {".", "。"}
TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)
ZH_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+|[^\s]", re.UNICODE)
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc",
    "co", "no", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split_terminal_sentences(text: str):
    """Yield (start, end, sentence) for period-terminated sentences only."""
    start = 0
    for i, char in enumerate(text):
        if char not in TERMINAL:
            continue
        if char == ".":
            before = text[i - 1] if i else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if before == "." or after == ".":
                continue
            if before.isdigit() and after.isdigit():
                continue
            if after and not after.isspace() and after not in '\"”’»)]}':
                continue
            token = re.search(r"([\w.]+)$", text[start:i], re.UNICODE)
            if token and token.group(1).rstrip(".").lower() in ABBREVIATIONS:
                continue
        end = i + 1
        sentence = text[start:end].strip()
        if len(sentence) >= 12 and sentence[-1] in TERMINAL:
            yield start, end, sentence
        start = end
        while start < len(text) and (text[start].isspace() or text[start] in '\"“”‘’»)]}'):
            start += 1


def build_manifest():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_bytes = SOURCE.read_bytes()
    paragraphs = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    sentences = []
    for paragraph_index, row in enumerate(paragraphs):
        lang = row.get("language")
        if lang not in LANGUAGES:
            continue
        text = row.get("text", "")
        for sentence_index, (start, end, sentence) in enumerate(split_terminal_sentences(text)):
            matches = list((ZH_TOKEN_RE if lang == "zh" else TOKEN_RE).finditer(sentence))
            if not matches or matches[-1].group() not in TERMINAL:
                continue
            sentence_id = hashlib.sha256(
                f"{lang}\0{row.get('source_url', '')}\0{paragraph_index}\0{start}\0{sentence}".encode("utf-8")
            ).hexdigest()[:24]
            for step, match in enumerate(matches):
                prefix = sentence[:match.end()]
                label = "EOT" if match.group() in TERMINAL and match.end() == len(sentence) else "NEOT"
                sentences.append({
                    "case_id": f"{sentence_id}:{step:04d}",
                    "sentence_id": sentence_id,
                    "language": lang,
                    "paragraph_id": paragraph_index,
                    "sentence_index": sentence_index,
                    "source_title": row.get("title", ""),
                    "source_url": row.get("source_url", ""),
                    "sentence": sentence,
                    "step": step,
                    "emitted_piece": match.group(),
                    "prefix": prefix,
                    "expected": label,
                    "is_terminal_period": label == "EOT",
                    "annotation_scope": "RULE_DIAGNOSTIC_CRAWLED_PROSE_NOT_DIALOGUE_GOLD",
                })
    counts = Counter(row["language"] for row in sentences)
    if any(not any(row["language"] == lang for row in sentences) for lang in LANGUAGES):
        raise RuntimeError(f"The existing crawl lacks terminal-period cases for a language: {dict(counts)}")
    return paragraphs, sentences, source_bytes


REMOTE_CODE = r'''import base64, gc, gzip, hashlib, json, os, sqlite3, sys, time
from pathlib import Path
sys.path.insert(0, "/content/turn_detect_v2")
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from turn_v2.common import normalize
from turn_v2.monolingual_data import encode_tail

ROOT = Path("__REMOTE_ROOT__")
RUN = ROOT / "outputs/teacher_first_v14_20261026"
OUT = RUN / "terminal_punctuation_eval___RUN_ID__"
OUT.mkdir(parents=True, exist_ok=False)
packed = base64.b64decode("__MANIFEST_B64__")
rows = json.loads(gzip.decompress(packed).decode("utf-8"))
assert rows and all(r["expected"] in ("NEOT", "EOT") for r in rows)
torch.set_num_threads(2)
torch.set_grad_enabled(False)
free_ram = __import__("psutil").virtual_memory().available
assert free_ram >= 4 * 1024**3, "Not enough Colab RAM for sequential CPU inference"

status = json.loads((RUN / "status.json").read_text())
launcher = json.loads((RUN / "launcher.json").read_text())
child = json.loads((RUN / "child.json").read_text())
def process_matches(receipt):
    pid = receipt.get("pid")
    proc = Path("/proc") / str(pid)
    if not pid or not (proc / "cmdline").exists(): return False
    cmd = (proc / "cmdline").read_bytes()
    ticks = (proc / "stat").read_text().split()[21]
    return b"turn_v2.teacher_first_v14" in cmd and ticks == receipt.get("start_ticks")
assert process_matches(launcher) and process_matches(child), "Training worker identity changed; inference cancelled"

db = sqlite3.connect("file:" + str(RUN / "state.sqlite") + "?mode=ro", uri=True, timeout=5)
champions = {key.split(":", 1)[1]: json.loads(value)
             for key, value in db.execute("SELECT key,value FROM meta WHERE key LIKE 'large_teacher:%'")}
db.close()
trace = []
summaries = {}
codec_equal = {}
for lang in ("vi", "en", "zh"):
    relevant = [r for r in rows if r["language"] == lang]
    if not relevant: continue
    champion = champions[lang]
    model_dir = Path(champion.get("path", RUN / "teachers" / lang / "best"))
    if not model_dir.is_absolute(): model_dir = ROOT / model_dir
    report_path = RUN / "teachers" / lang / "best_report.json"
    report = json.loads(report_path.read_text())
    expected_sha = champion["weights_sha256"]
    weights_path = model_dir / "model.safetensors"
    def file_sha(path):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8*1024*1024), b""): h.update(block)
        return h.hexdigest()
    before_sha = file_sha(weights_path)
    assert before_sha == expected_sha == report["weights_sha256"], f"{lang}: checkpoint changed or hash mismatch before inference"
    threshold = float(champion["semantic_dev"]["threshold"])
    print(json.dumps({"event":"loading_cpu_model","language":lang,"cases":len(relevant),"checkpoint_sha256":before_sha}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32
    ).to("cpu").eval()
    after_load_sha = file_sha(weights_path)
    assert after_load_sha == before_sha, f"{lang}: best checkpoint changed while loading; refusing mixed result"

    encoded = [encode_tail(tokenizer, "", r["prefix"])["input_ids"] for r in relevant]
    probs = []
    batch_size = 32
    for start in range(0, len(encoded), batch_size):
        block = encoded[start:start+batch_size]
        width = max(map(len, block))
        ids = torch.full((len(block), width), int(tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(block), width), dtype=torch.long)
        for i, seq in enumerate(block):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            probs.extend(logits.softmax(-1)[:, 1].tolist())

    without_dot, with_dot = [], []
    for r in relevant:
        if r["is_terminal_period"]:
            without_dot.append(encode_tail(tokenizer, "", r["prefix"][:-1])["input_ids"])
            with_dot.append(encode_tail(tokenizer, "", r["prefix"])["input_ids"])
    same_inputs = sum(a == b for a, b in zip(without_dot, with_dot))
    codec_equal[lang] = {"terminal_pairs":len(with_dot),"identical_token_ids_after_normalize":same_inputs,
                         "punctuation_survives_input_codec":bool(with_dot) and same_inputs != len(with_dot)}

    local_trace = []
    for row, probability in zip(relevant, probs):
        prediction = "EOT" if probability >= threshold else "NEOT"
        local_trace.append({**row, "prob_EOT":float(probability), "threshold":threshold,
                            "prediction":prediction, "correct":prediction == row["expected"]})
    trace.extend(local_trace)
    labels = [r["expected"] for r in local_trace]
    predictions = [r["prediction"] for r in local_trace]
    tp = sum(y == "EOT" and p == "EOT" for y,p in zip(labels,predictions))
    fp = sum(y == "NEOT" and p == "EOT" for y,p in zip(labels,predictions))
    fn = sum(y == "EOT" and p == "NEOT" for y,p in zip(labels,predictions))
    tn = sum(y == "NEOT" and p == "NEOT" for y,p in zip(labels,predictions))
    eot_f1 = (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0
    grouped = {}
    for r in local_trace: grouped.setdefault(r["sentence_id"], []).append(r)
    strict_pass = sum(all(x["prediction"] == x["expected"] for x in sequence) for sequence in grouped.values())
    final = [r for r in local_trace if r["is_terminal_period"]]
    preterminal = [r for r in local_trace if not r["is_terminal_period"]]
    final_end = sum(r["prediction"] == "EOT" for r in final)
    early_end = sum(r["prediction"] == "EOT" for r in preterminal)
    after_sha = file_sha(weights_path)
    assert after_sha == before_sha, f"{lang}: checkpoint changed during inference"
    summaries[lang] = {
      "checkpoint_sha256":before_sha,"threshold_from_frozen_semantic_silver_dev":threshold,
      "sentences":len(grouped),"prefix_events":len(local_trace),
      "accuracy":(tp+tn)/len(local_trace) if local_trace else 0.0,
      "eot_precision":tp/(tp+fp) if tp+fp else 0.0,"eot_recall":tp/(tp+fn) if tp+fn else 0.0,
      "eot_f1":eot_f1,"confusion_NEOT_EOT":[[tn,fp],[fn,tp]],
      "premature_eot_rate_before_terminal_period":early_end/len(preterminal) if preterminal else 0.0,
      "terminal_period_end_rate":final_end/len(final) if final else 0.0,
      "strict_sentence_passes":strict_pass,"strict_sentence_rate":strict_pass/len(grouped) if grouped else 0.0,
      "period_vs_no_period_token_ids_identical":codec_equal[lang]["identical_token_ids_after_normalize"] == codec_equal[lang]["terminal_pairs"],
      "period_codec":codec_equal[lang],"training_stage_when_evaluated":status.get("stage"),
      "training_language_when_evaluated":status.get("language"),
    }
    del model, tokenizer
    gc.collect()
    print(json.dumps({"event":"language_done","language":lang,"metrics":summaries[lang]},ensure_ascii=False), flush=True)

trace_path = OUT / "trace.jsonl"
with trace_path.open("w",encoding="utf-8") as f:
    for row in trace: f.write(json.dumps(row,ensure_ascii=False)+"\n")
source_hash = "__SOURCE_SHA256__"
report = {
  "protocol":"terminal_period_streaming_v1",
  "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
  "scope":"RULE_DIAGNOSTIC_CRAWLED_PROSE_NOT_DIALOGUE_GOLD",
  "rule":"All prefixes before a sentence-final period are NEOT, including meaningful/complete clauses and comma prefixes; the final . or 。 prefix is EOT.",
  "source_corpus":"outputs/trilingual_stream_20261024/crawl/paragraphs.jsonl",
  "source_corpus_sha256":source_hash,
  "paragraphs_in_input":len(set((r["language"],r["paragraph_id"]) for r in rows)),
  "prefix_events":len(trace),"languages":summaries,"terminal_punctuation_tokenization":codec_equal,
  "important_limitations":["Crawled expository text is not real conversation or human EOT gold.",
    "Model thresholds remain selected on frozen Qwen semantic silver development data.",
    "The current model normalization removes punctuation before tokenization; check the per-language codec evidence.",
    "Inference ran sequentially on Colab CPU FP32; the active training worker and GPU were not used."],
  "quality_target_certified":False,"training_worker_modified":False,"teacher_weights_downloaded":False,
}
(OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"event":"complete","remote_output":str(OUT.relative_to(ROOT)),"report":report},ensure_ascii=False),flush=True)
'''


def html_review(report, trace):
    sections = []
    for lang in LANGUAGES:
        value = report["languages"].get(lang)
        if not value:
            continue
        sections.append(
            f"<section><h2>{lang.upper()}</h2><p>Accuracy {value['accuracy']:.1%} · "
            f"EOT F1 {value['eot_f1']:.1%} · trước dấu chấm ngắt sớm {value['premature_eot_rate_before_terminal_period']:.1%} · "
            f"END đúng tại dấu chấm {value['terminal_period_end_rate']:.1%} · "
            f"câu đạt nghiêm ngặt {value['strict_sentence_passes']}/{value['sentences']}</p>"
            f"<p>Model thấy dấu chấm sau normalize: "
            f"{'có' if value['period_codec']['punctuation_survives_input_codec'] else 'không'}; "
            f"cặp token giống hệt {value['period_codec']['identical_token_ids_after_normalize']}/"
            f"{value['period_codec']['terminal_pairs']}.</p></section>"
        )
    rows = []
    grouped = defaultdict(list)
    for item in trace:
        grouped[item["sentence_id"]].append(item)
    for sentence_id, items in grouped.items():
        first = items[0]
        errors = [r for r in items if not r["correct"]]
        early = [r for r in items if r["expected"] == "NEOT" and r["prediction"] == "EOT"]
        final = items[-1]
        source = f'<a href="{html.escape(first["source_url"], quote=True)}">nguồn</a>' if first["source_url"] else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(first['language'].upper())}</td>"
            f"<td>{html.escape(first['source_title'])}</td><td>{source}</td>"
            f"<td>{html.escape(first['sentence'])}</td>"
            f"<td>{len(items)}</td><td>{len(early)}</td>"
            f"<td>{final['prediction']} ({final['prob_EOT']:.3f})</td>"
            f"<td>{'PASS' if not errors else 'FAIL'}</td></tr>"
        )
    return """<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Terminal punctuation streaming evaluation</title><style>
body{font:15px system-ui;margin:24px;color:#172033}h1,h2{margin-bottom:8px}section{padding:12px 16px;margin:12px 0;background:#f1f5f9;border-radius:8px}
.note{background:#fff7ed;padding:12px;border-left:4px solid #f59e0b}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(4){max-width:500px}
</style><h1>Streaming rule: NEOT before sentence-final period, EOT at period</h1>
<div class="note"><b>Phạm vi:</b> đây là benchmark quy tắc trên văn xuôi crawl, không phải hội thoại gold. Mọi prefix trước dấu chấm cuối đều yêu cầu NEOT, kể cả khi mệnh đề đã có nghĩa; prefix có dấu chấm cuối yêu cầu EOT. Ngưỡng model giữ nguyên từ semantic Qwen silver dev. Model chạy CPU FP32 tuần tự; worker train không bị sửa.</div>
""" + "\n".join(sections) + "<h2>Ví dụ từng câu</h2><table><thead><tr><th>Lang</th><th>Nguồn</th><th>Link</th><th>Câu</th><th>Prefix</th><th>EOT sớm</th><th>Dấu chấm cuối</th><th>Kết quả</th></tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></html>"


def recover_existing(run_id: str):
    """Download the already-computed trace after a report-packaging failure."""
    out = RUN / ("terminal_punctuation_eval_" + run_id)
    if not out.is_dir():
        raise FileNotFoundError(out)
    receipt_path = out / "remote_execution_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    logged_metrics = {}
    for item in receipt.get("outputs", []):
        if item.get("output_type") != "stream":
            continue
        for line in item.get("text", "").splitlines():
            if '"event": "language_done"' not in line:
                continue
            value = json.loads(line)
            logged_metrics[value["language"]] = value["metrics"]
    if set(logged_metrics) != set(LANGUAGES):
        raise RuntimeError(f"Incomplete per-language receipts in {receipt_path}")
    runtime = campaign_runtime()
    try:
        runtime.connect()
        remote_output = f"outputs/teacher_first_v14_20261026/terminal_punctuation_eval_{run_id}"
        response = runtime.http.get(
            runtime.base_url + "files/" + quote("content/turn_detect_v2/" + remote_output + "/trace.jsonl", safe="/"),
            stream=True, timeout=180, allow_redirects=False,
        )
        if response.status_code != 200 or "html" in response.headers.get("content-type", ""):
            raise RuntimeError(f"Could not recover completed Colab trace: HTTP {response.status_code}")
        with (out / "trace.jsonl").open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
    finally:
        runtime.close()
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    input_meta = json.loads((out / "input_manifest.json").read_text(encoding="utf-8"))
    assert len(trace) == input_meta["prefix_events"], "Recovered trace and frozen manifest disagree"
    source_hash = digest_bytes(SOURCE.read_bytes())
    report = {
        "protocol":"terminal_period_streaming_v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"RULE_DIAGNOSTIC_CRAWLED_PROSE_NOT_DIALOGUE_GOLD",
        "rule":"All prefixes before a sentence-final period are NEOT, including meaningful/complete clauses and comma prefixes; the final . or 。 prefix is EOT.",
        "source_corpus":"outputs/trilingual_stream_20261024/crawl/paragraphs.jsonl",
        "source_corpus_sha256":source_hash,
        "paragraphs_in_input":input_meta["source_paragraphs"],
        "prefix_events":len(trace),"languages":logged_metrics,
        "terminal_punctuation_tokenization":{lang:logged_metrics[lang]["period_codec"] for lang in LANGUAGES},
        "important_limitations":["Crawled expository text is not real conversation or human EOT gold.",
            "Model thresholds remain selected on frozen Qwen semantic silver development data.",
            "The current model normalization removes punctuation before tokenization; per-language codec evidence confirms identical input ids with and without the final period.",
            "Inference ran sequentially on Colab CPU FP32; the active training worker and GPU were not used."],
        "quality_target_certified":False,"training_worker_modified":False,"teacher_weights_downloaded":False,
        "remote_run_id":run_id,"remote_execution_status":"inference completed; local report packaging recovered from saved Colab trace",
    }
    (out / "review.html").write_text(html_review(report, trace), encoding="utf-8")
    report["local_input_manifest"] = str((out / "sentence_prefix_manifest.jsonl").relative_to(ROOT))
    report["local_review_html"] = str((out / "review.html").relative_to(ROOT))
    report["input_manifest_sha256"] = digest_bytes((out / "sentence_prefix_manifest.jsonl").read_bytes())
    report["trace_sha256"] = digest_bytes((out / "trace.jsonl").read_bytes())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output":str(out),"sentences":input_meta["sentences"],"prefix_events":len(trace),"languages":logged_metrics},ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recover-run-id", help="Recover an already completed Colab inference run without rerunning models")
    args = parser.parse_args()
    if args.recover_run_id:
        recover_existing(args.recover_run_id)
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RUN / ("terminal_punctuation_eval_" + run_id)
    out.mkdir(parents=True, exist_ok=False)
    ensure_free_space(out, 2 * 1024**2)
    paragraphs, sentences, source_bytes = build_manifest()
    manifest_bytes = (json.dumps(sentences, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    (out / "sentence_prefix_manifest.jsonl").write_bytes(
        b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in sentences)
    )
    source_sha = digest_bytes(source_bytes)
    input_meta = {
        "protocol":"terminal_period_streaming_v1", "source_path":str(SOURCE.relative_to(ROOT)),
        "source_sha256":source_sha, "source_paragraphs":len(paragraphs),
        "sentences":len(set(r["sentence_id"] for r in sentences)),
        "prefix_events":len(sentences), "by_language":{
            lang:{"sentences":len({r["sentence_id"] for r in sentences if r["language"]==lang}),
                  "prefix_events":sum(r["language"]==lang for r in sentences),
                  "terminal_periods":sum(r["language"]==lang and r["is_terminal_period"] for r in sentences)}
            for lang in LANGUAGES},
        "label_rule":"all prefixes before final . or 。 are NEOT; final-period prefix EOT",
        "annotation_scope":"RULE_DIAGNOSTIC_CRAWLED_PROSE_NOT_DIALOGUE_GOLD",
    }
    (out / "input_manifest.json").write_text(json.dumps(input_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    packed = base64.b64encode(gzip.compress(manifest_bytes, compresslevel=6)).decode("ascii")
    code = REMOTE_CODE.replace("__REMOTE_ROOT__", REMOTE_ROOT).replace("__RUN_ID__", run_id).replace("__MANIFEST_B64__", packed).replace("__SOURCE_SHA256__", digest_bytes(source_bytes))

    runtime = campaign_runtime()
    try:
        runtime.connect()
        result = runtime.execute(code, timeout=1800, echo=True)
        (out / "remote_execution_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result.get("status") != "ok":
            raise RuntimeError(f"Colab inference failed; inspect {out / 'remote_execution_receipt.json'}")
        streams = [item["text"] for item in result["outputs"] if item.get("output_type") == "stream" and item.get("name") == "stdout"]
        complete_lines = [line for stream in streams for line in stream.splitlines() if '"event": "complete"' in line or '"event":"complete"' in line]
        if not complete_lines:
            raise RuntimeError("Colab returned no completion receipt")
        completion = json.loads(complete_lines[-1])
        remote_output = completion["remote_output"]
        for name in ("report.json", "trace.jsonl"):
            response = runtime.http.get(
                runtime.base_url + "files/" + quote("content/turn_detect_v2/" + remote_output + "/" + name, safe="/"),
                stream=True, timeout=180, allow_redirects=False,
            )
            if response.status_code != 200 or "html" in response.headers.get("content-type", ""):
                raise RuntimeError(f"Could not retrieve Colab {name}: HTTP {response.status_code}")
            target = out / name
            with target.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    f.write(block)
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        trace = [json.loads(line) for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(trace) == len(sentences), "Prediction count does not match the frozen input manifest"
        (out / "review.html").write_text(html_review(report, trace), encoding="utf-8")
        report["local_input_manifest"] = str((out / "sentence_prefix_manifest.jsonl").relative_to(ROOT))
        report["local_review_html"] = str((out / "review.html").relative_to(ROOT))
        report["input_manifest_sha256"] = digest_bytes((out / "sentence_prefix_manifest.jsonl").read_bytes())
        report["trace_sha256"] = digest_bytes((out / "trace.jsonl").read_bytes())
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output":str(out),"sentences":input_meta["sentences"],"prefix_events":len(trace),"languages":report["languages"]},ensure_ascii=False,indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
