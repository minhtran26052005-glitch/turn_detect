"""Distill a frozen teacher into a small GRU using TRAIN and DEV only."""
import argparse
import json
from pathlib import Path


def batch_inputs(encoded, indices, device):
    import numpy as np
    import torch

    ids, lengths, labels, _ = encoded
    width = int(lengths[indices].max())
    x = torch.from_numpy(np.asarray(ids[indices, :width], dtype=np.int64)).to(device)
    lens = torch.from_numpy(np.asarray(lengths[indices], dtype=np.int64)).to(device)
    mask = (torch.arange(width, device=device)[None, :] < lens[:, None]).long()
    target = torch.from_numpy(np.asarray(labels[indices], dtype=np.int64)).to(device)
    return x, mask, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("vi", "en", "zh"), default="vi")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) <= 0 or args.lr <= 0:
        parser.error("epochs, batch size and learning rate must be positive")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from . import cache, data
    from .base import file_sha
    from .metrics import calibrate_on_dev
    from .student import TinyTurnGRU

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    root = args.run_dir.resolve()
    data.ROOT, data.JOB = root, root / "full_epoch_v1"
    manifest = json.loads((data.JOB / "current_dataset.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("dataset generation is incomplete")
    teacher_dir = root / "teachers" / args.language / "best"
    teacher_hash = file_sha(teacher_dir / "model.safetensors")
    identity = {"language": args.language, "generation": manifest["generation"],
                "train_fingerprint": manifest["fingerprints"][args.language]["train"]["sha256"],
                "dev_fingerprint": manifest["fingerprints"][args.language]["dev"]["sha256"],
                "teacher_sha256": teacher_hash, "batch_size": args.batch_size,
                "learning_rate": args.lr, "seed": args.seed}
    if args.resume:
        existing = json.loads((args.output / "identity.json").read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("resume configuration or frozen inputs differ")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    train = cache.cache(manifest, args.language, "train")
    dev = cache.cache(manifest, args.language, "dev")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(teacher_dir, local_files_only=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir, local_files_only=True).to(args.device).eval()
    teacher.requires_grad_(False)
    config = {"vocab_size": len(tokenizer), "embedding_size": 64, "hidden_size": 96,
              "layers": 1, "teacher_size": teacher.config.hidden_size}
    student = TinyTurnGRU(**config).to(args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    epoch, position, step, best = 1, 0, 0, -1.0
    checkpoint_path = args.output / "latest.pt"
    if args.resume:
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        student.load_state_dict(saved["student"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, position, step, best = saved["epoch"], saved["position"], saved["step"], saved["best"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())

    def save(next_epoch, cursor):
        temporary = args.output / "latest.pt.partial"
        if temporary.exists():
            raise FileExistsError(f"preserve and inspect partial checkpoint: {temporary}")
        parameter_bytes = sum(p.numel() * p.element_size() for p in student.parameters())
        required = max(checkpoint_path.stat().st_size if checkpoint_path.exists() else 0, 3 * parameter_bytes)
        data.ensure_space(args.output, required + 16 * 1024**2)
        torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": next_epoch, "position": cursor, "step": step, "best": best,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if args.device == "cuda" else None}, temporary)
        temporary.replace(checkpoint_path)

    while epoch <= args.epochs:
        order = np.random.default_rng(args.seed + epoch).permutation(len(train[2]))
        if not 0 <= position <= len(order):
            raise ValueError("invalid saved cursor")
        student.train()
        for start in range(position, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x, mask, labels = batch_inputs(train, indices, args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(input_ids=x, attention_mask=mask).logits.float()
            logits = student(x, mask).float()
            ce = F.cross_entropy(logits, labels, label_smoothing=0.02)
            kd = F.kl_div(F.log_softmax(logits / 2, dim=-1),
                          F.softmax(teacher_logits / 2, dim=-1), reduction="batchmean") * 4
            loss = 0.5 * ce + 0.5 * kd
            if not torch.isfinite(loss):
                raise ValueError("non-finite distillation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            position, step = start + len(indices), step + 1
            if step % 500 == 0:
                save(epoch, position)
                print(json.dumps({"epoch": epoch, "step": step, "rows": position, "loss": float(loss.detach())}), flush=True)
        student.eval()
        probs = []
        with torch.inference_mode():
            for start in range(0, len(dev[2]), args.batch_size):
                indices = np.arange(start, min(start + args.batch_size, len(dev[2])))
                x, mask, _ = batch_inputs(dev, indices, args.device)
                probs.append(student(x, mask).float().softmax(-1)[:, 1].cpu().numpy())
        metrics = calibrate_on_dev(dev[2], np.concatenate(probs))
        if metrics["joint_score"] > best:
            best = metrics["joint_score"]
            temporary = args.output / "student.pt.partial"
            if temporary.exists():
                raise FileExistsError(f"preserve and inspect partial weights: {temporary}")
            data.ensure_space(args.output, sum(p.numel() * p.element_size() for p in student.parameters()) + 16 * 1024**2)
            torch.save(student.state_dict(), temporary)
            temporary.replace(args.output / "student.pt")
            tokenizer.save_pretrained(args.output)
            config_path = args.output / "student_config.json"
            config_path.write_text(json.dumps({"model": config, "threshold": metrics["threshold"],
                                              "selection": "dev", "metrics": metrics}, indent=2) + "\n", encoding="utf-8")
        epoch, position = epoch + 1, 0
        save(epoch, position)
        print(json.dumps({"epoch_complete": epoch - 1, "dev": metrics}), flush=True)


if __name__ == "__main__":
    main()
