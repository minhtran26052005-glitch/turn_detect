"""Build an inspectable dev comparison from a completed training trial."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from turn_v2.common import LABEL2ID, file_hash, read_jsonl, write_json, write_jsonl
from turn_v2.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=Path("outputs/training_trial_20261024"))
    args = parser.parse_args()
    root = args.trial
    report = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    selected = report["dev_selected_threshold"]
    state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    rows = list(read_jsonl(Path("data_v2/frozen/dev.jsonl")))
    saved = np.load(root / "dev_predictions.npz")
    labels, probs = saved["labels"], saved["prob_EOT"]
    assert np.array_equal(labels, [LABEL2ID[row["label"]] for row in rows])
    actual = classification_metrics(labels, probs, selected["threshold"])
    assert actual == selected
    subset = json.loads((root / "comparison_subset.json").read_text(encoding="utf-8"))
    assert subset["dev_sha256"] == file_hash(Path("data_v2/frozen/dev.jsonl"))
    indices = np.asarray(subset["zero_based_dev_indices"])
    old_rows = list(read_jsonl(root / "legacy_dev_2048/predictions.jsonl"))
    assert len(old_rows) == len(indices)
    for old, index in zip(old_rows, indices, strict=True):
        assert old["text"] == rows[index]["text"] and old["context"] == rows[index]["context"]
    old = classification_metrics(labels[indices], [row["prob_EOT"] for row in old_rows])
    new = classification_metrics(labels[indices], probs[indices], selected["threshold"])
    compare = {"samples": len(indices), "selection": subset["selection"], "legacy": old, "new": new,
        "new_at_0.5": classification_metrics(labels[indices], probs[indices], 0.5),
        "accuracy_difference_percentage_points": 100 * (new["accuracy"] - old["accuracy"]),
        "eot_f1_difference_percentage_points": 100 * (new["eot_f1"] - old["eot_f1"]),
        "scope": "Descriptive comparison on the same dev inputs. Legacy training exposure is unknown; architecture and training data also differ."}
    write_json(root / "matched_dev_comparison.json", compare)
    errors = []
    for index, (row, label, prob) in enumerate(zip(rows, labels, probs, strict=True)):
        prediction = int(prob >= selected["threshold"])
        if prediction != label:
            errors.append({**row, "dev_index": index, "prob_EOT": float(prob),
                "prediction": "EOT" if prediction else "NEOT",
                "error_type": "false_eot" if prediction else "missed_eot",
                "wrong_prediction_probability": float(prob if prediction else 1-prob)})
    errors.sort(key=lambda row: row["wrong_prediction_probability"], reverse=True)
    write_jsonl(root / "dev_errors_review.jsonl", errors)
    buckets = []
    lengths = np.asarray([len(row["text"].split()) for row in rows])
    for low, high in [(1, 3), (4, 7), (8, 15), (16, 31), (32, 100000)]:
        mask = (lengths >= low) & (lengths <= high)
        if mask.any():
            buckets.append({"word_count": f"{low}-{high if high < 100000 else '+'}",
                **classification_metrics(labels[mask], probs[mask], selected["threshold"])})
    write_json(root / "dev_length_buckets.json", buckets)
    evaluations = [entry for entry in state["log_history"] if "eval_accuracy" in entry]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, label in [("eval_accuracy", "Accuracy"), ("eval_eot_f1", "EOT F1")]:
        axes[0].plot([v["step"] for v in evaluations], [100*v[key] for v in evaluations], marker="o", label=label)
    axes[0].axhline(90, color="gray", linestyle="--", label="Target 90%")
    axes[0].set(xlabel="Optimizer steps", ylabel="Dev score (%)", title="Dev at threshold 0.5")
    axes[0].legend()
    losses = [v for v in state["log_history"] if "loss" in v]
    axes[1].plot([v["step"] for v in losses], [v["loss"] for v in losses])
    axes[1].set(xlabel="Optimizer steps", ylabel="Training loss", title="Training loss (50-step windows)")
    fig.savefig(root / "learning_curve.png", dpi=160)
    plt.close(fig)
    esc = html.escape
    def metric_row(name, values):
        return f'<tr><th>{esc(name)}</th><td>{values["samples"]:,}</td><td>{values["accuracy"]:.2%}</td><td>{values["eot_f1"]:.2%}</td><td>{values["false_eot"]:,}</td><td>{values["missed_eot"]:,}</td></tr>'
    metric_table = '<table><tr><th>Model / dữ liệu</th><th>Số mẫu</th><th>Accuracy</th><th>F1 EOT</th><th>Cắt sớm</th><th>Bỏ lỡ EOT</th></tr>'
    metric_table += metric_row('Model mới / toàn bộ dev', selected) + metric_row('Model cũ / dev cố định', old) + metric_row('Model mới / cùng dev cố định', new) + '</table>'
    holdout_path = root / "holdout_report.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.exists() else None
    if holdout:
        metric_table = metric_table.removesuffix('</table>') + metric_row('Model mới / holdout cuối', holdout["metrics"]) + '</table>'
    final_text = (f'Holdout: {esc(holdout["status"])}. Accuracy {holdout["metrics"]["accuracy"]:.2%}; F1 EOT {holdout["metrics"]["eot_f1"]:.2%}.'
        if holdout else 'Holdout chưa được mở; chưa có kết luận đạt mục tiêu 90% trên holdout.')
    legacy_path = root / "legacy_regression/report.json"
    if legacy_path.exists():
        regression = json.loads(legacy_path.read_text(encoding="utf-8"))
        original = json.loads(Path("outputs/test_cpu_single/metrics.json").read_text(encoding="utf-8"))["metrics"]
        metric_table = metric_table.removesuffix('</table>')
        metric_table += metric_row('Model cũ / test cũ', original) + metric_row('Model mới / test cũ', regression["metrics"]) + '</table>'
        final_text += f' Test cũ: accuracy {regression["metrics"]["accuracy"]:.2%}, F1 EOT {regression["metrics"]["eot_f1"]:.2%}; F1 chưa đạt 90% trên bộ này. Test cũ chỉ dùng kiểm tra hồi quy sau khi khóa model/ngưỡng.'
    error_rows = ''.join(f'<tr><td>{esc(row["error_type"])}</td><td>{esc(row["text"])}</td><td>{esc(row["label"])}</td><td>{row["prob_EOT"]:.4f}</td></tr>' for row in errors[:80])
    page = f'''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Kết quả train thử turn detector</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172238;background:#f7f9fc}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{text-align:left;padding:10px;border:1px solid #d8dfeb}}img{{max-width:100%}}p{{line-height:1.6}}small{{color:#57647b}}</style>
<h1>Bản train thử turn detector</h1><p>Đã chạy {report["optimizer_steps"]:,} bước optimizer, batch hiệu dụng 32, trên pool {report["train_samples"]:,} mẫu train nhãn sẵn. Ngưỡng {selected["threshold"]:.3f} được chọn từ dev. Checkpoint tốt nhất: {esc(str(state.get("best_model_checkpoint")))}.</p>
<p><b>{final_text}</b></p>{metric_table}<p><small>So sánh model cũ và mới mang tính mô tả. Chưa xác minh tập train của model cũ; backbone cũng khác. Toàn bộ dev được dùng chọn model/ngưỡng. Nhãn tham chiếu do dữ liệu cung cấp; grouping nguồn là heuristic, chưa phải kiểm định hội thoại thực do người rà soát.</small></p>
<img src="learning_curve.png" alt="Learning curves"><p>Export ONNX parity: {esc(str(report.get("export",{}).get("passed")))}. JSON gốc: <a href="training_report.json">training_report.json</a>; <a href="matched_dev_comparison.json">so sánh cùng mẫu</a>; <a href="dev_errors_review.jsonl">toàn bộ lỗi dev để rà nhãn</a>.</p>
<h2>80 lỗi dev có xác suất dự đoán sai cao nhất</h2><p>Dùng để rà dữ liệu và phân tích lỗi. Không tự thay nhãn theo model.</p><table><tr><th>Loại lỗi</th><th>Text</th><th>Nhãn tham chiếu</th><th>P(EOT)</th></tr>{error_rows}</table></html>'''
    (root / "review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"dev": selected, "comparison": compare, "holdout": holdout}, indent=2))


if __name__ == "__main__":
    main()
