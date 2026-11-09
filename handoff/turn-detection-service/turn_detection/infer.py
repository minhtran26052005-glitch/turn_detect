"""Run a teacher or a distilled student on one boundary."""
import argparse
import json
from pathlib import Path

from .codec import encode_pair, validate_future


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--future", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    validate_future(args.future)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu when appropriate")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    contract = args.model / "student_config.json"
    if contract.exists():
        from .student import TinyTurnGRU

        settings = json.loads(contract.read_text(encoding="utf-8"))
        model = TinyTurnGRU(**settings["model"])
        model.load_state_dict(torch.load(args.model / "student.pt", map_location="cpu", weights_only=True))
        threshold = settings["threshold"] if args.threshold is None else args.threshold
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        report_path = args.model.parent / "best_report.json"
        if args.threshold is None and not report_path.exists():
            parser.error("provide --threshold or place best_report.json next to the model directory")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        threshold = report.get("threshold_selected_on_dev") if args.threshold is None else args.threshold
    if threshold is None or not 0 <= threshold <= 1:
        parser.error("a DEV-selected threshold between 0 and 1 is required")
    model = model.to(args.device).eval()
    encoded = encode_pair(tokenizer, args.context, args.future)
    inputs = {key: torch.tensor([value], device=args.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(**inputs)
        logits = output if contract.exists() else output.logits
        probability = float(logits.float().softmax(-1)[0, 1])
    print(json.dumps({"label": "EOT" if probability >= threshold else "NEOT",
                      "prob_eot": probability, "threshold": threshold}, ensure_ascii=False))


if __name__ == "__main__":
    main()
