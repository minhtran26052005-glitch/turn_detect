"""Exercise the student pipeline on a small synthetic dataset on Colab."""
import json
import os
import argparse
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-lock", type=Path, help="Existing campaign GPU lock for an isolated one-step teacher smoke")
    parser.add_argument("--skip-if-busy", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project), "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="turn-detection-smoke-", dir="/content") as temporary:
        root = Path(temporary)
        run = root / "work"
        teacher = run / "teachers/vi/best"
        dataset = run / "full_epoch_v1/datasets/fixture"
        teacher.mkdir(parents=True)
        dataset.mkdir(parents=True)
        vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3, "xin": 4,
                 "chao": 5, "tiep": 6, "sau": 7}
        backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
                                             cls_token="[CLS]", sep_token="[SEP]")
        tokenizer.save_pretrained(teacher)
        config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32, num_labels=2, pad_token_id=0)
        BertForSequenceClassification(config).save_pretrained(teacher, safe_serialization=True)
        db = sqlite3.connect(dataset / "dataset.sqlite")
        db.execute("CREATE TABLE records(example_id TEXT, language TEXT, split TEXT, label TEXT, k INTEGER, tokens BLOB, disposition TEXT, record_json TEXT)")
        fingers = {}
        for split in ("train", "dev"):
            strata = {f"{label}:{k}": 0 for label in ("NEOT", "EOT") for k in (1, 2)}
            for i in range(12):
                label, k = ("NEOT", "EOT")[i % 2], 1 + (i // 2) % 2
                ids = np.array([1, 4, 5, 2, 6] + ([7] if k == 2 else []) + [2], dtype="<u4").tobytes()
                record = {"label": label, "lookahead_words": k,
                          "target_punctuation": "." if label == "EOT" else ","}
                db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?)", (f"{split}-{i:04d}", "vi", split, label, k, ids, "accepted", json.dumps(record)))
                strata[f"{label}:{k}"] += 1
            fingers[split] = {"rows": 12, "sha256": split + "-fixture", "strata": strata}
        db.commit()
        db.close()
        manifest = {"complete": True, "generation": "fixture", "source_cycle": 0, "database_sha256": "fixture",
                    "fingerprints": {"vi": fingers}}
        (run / "full_epoch_v1/current_dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        student = root / "student"
        commands = [
            [sys.executable, "-m", "turn_detection.distill", "--run-dir", str(run), "--output", str(student),
             "--epochs", "1", "--batch-size", "4", "--device", "cpu"],
            [sys.executable, "-m", "turn_detection.infer", "--model", str(teacher), "--device", "cpu",
             "--context", "xin chao", "--future", "tiep sau", "--threshold", "0.5"],
            [sys.executable, "-m", "turn_detection.infer", "--model", str(student), "--device", "cpu",
             "--context", "xin chao", "--future", "tiep sau"],
        ]
        outputs = []
        for command in commands:
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stdout[-3000:] + "\n" + result.stderr[-3000:])
            outputs.append(result.stdout.strip().splitlines()[-1])
        predictions = [json.loads(value) for value in outputs[1:]]
        if any(value["label"] not in ("NEOT", "EOT") for value in predictions):
            raise ValueError("invalid smoke prediction")
        checkpoint = torch.load(student / "latest.pt", map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 2 and checkpoint["position"] == 0 and checkpoint["step"] == 3
        result = {"distill_train_rows": 12, "distill_dev_rows": 12,
                  "student_optimizer_steps": checkpoint["step"], "teacher_infer": True,
                  "student_infer": True, "test_scored": False}
        if args.gpu_lock:
            import fcntl
            if not args.gpu_lock.is_file():
                raise ValueError("the specified campaign GPU lock does not exist")
            toy_model = BertForSequenceClassification.from_pretrained(teacher, local_files_only=True)
            optimizer = torch.optim.AdamW([
                {"params": [p for k, p in toy_model.named_parameters() if not k.startswith(("classifier.", "pooler."))], "lr": 1e-5},
                {"params": [p for k, p in toy_model.named_parameters() if k.startswith(("classifier.", "pooler."))], "lr": 5e-5}], weight_decay=0.01)
            first = next(toy_model.parameters())
            first.grad = torch.zeros_like(first)
            optimizer.step()
            with args.gpu_lock.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | (fcntl.LOCK_NB if args.skip_if_busy else 0))
                except BlockingIOError:
                    result["teacher_train"] = "skipped_gpu_busy"
                    print(json.dumps(result))
                    return
                checkpoint_path = teacher.parent / "latest_resume.pt"
                order = np.random.default_rng(7).permutation(12).astype(np.int64)
                visits = np.zeros(12, dtype=np.uint8)
                visits[order[:6]] = 1
                sampler = {"generation": "fixture", "fingerprint": fingers["train"]["sha256"],
                           "epoch": 1, "eligible_rows": 12, "position": 6, "order": order,
                           "visits": visits, "epoch_steps": 1, "loss_sum": 4.0, "loss_den": 6.0}
                torch.save({"model": toy_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "cycle": 0, "cycle_step": 1, "total_steps": 1, "full_epoch": sampler,
                            "training_fingerprint": fingers["train"]["sha256"],
                            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state()}, checkpoint_path)
                (teacher.parent / "resume_receipt.json").write_text(
                    json.dumps({"bytes": checkpoint_path.stat().st_size, "total_steps": 1,
                                "full_epoch_generation": "fixture", "full_epoch_number": 1,
                                "full_epoch_cursor": 6}), encoding="utf-8")
                command = [sys.executable, "-m", "turn_detection.train", "--run-dir", str(run),
                           "--language", "vi", "--generation", "fixture", "--epoch", "1", "--budget", "1"]
                trained = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
                if trained.returncode:
                    raise RuntimeError(trained.stdout[-3000:] + "\n" + trained.stderr[-3000:])
            receipt = json.loads((teacher.parent / "resume_receipt.json").read_text(encoding="utf-8"))
            assert receipt["full_epoch_complete"] and receipt["full_epoch_cursor"] == 12 and receipt["cycle_step"] == 2
            assert (run / "full_epoch_v1/runs/fixture/vi/epoch_0001.json").exists()
            result.update(teacher_resumed_cursor=6, teacher_train_optimizer_steps=1,
                          teacher_full_epoch_rows=12, teacher_dev_complete=True)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
