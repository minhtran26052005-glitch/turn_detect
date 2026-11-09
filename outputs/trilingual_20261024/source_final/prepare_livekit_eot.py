"""Download EN/ZH transcript/timing metadata only; no audio, no training use."""
import json
from pathlib import Path
import time

import requests

from turn_v2.common import file_hash, write_json, write_jsonl


def main():
    root = Path("data_v3/livekit_eot_20261024")
    root.mkdir(parents=True, exist_ok=False)
    http = requests.Session()
    http.headers["User-Agent"] = "TurnDetectorResearch/0.3 (text-only frozen external evaluation)"
    info = http.get("https://huggingface.co/api/datasets/livekit/eot-bench-data", timeout=45)
    info.raise_for_status(); revision = info.json()["sha"]
    rows, receipts = [], []
    for language, expected_total in [("en", 400), ("zh", 344)]:
        collected = []
        for offset in range(0, expected_total, 100):
            params = {"dataset": "livekit/eot-bench-data", "config": language, "split": "validation", "offset": offset, "length": min(100, expected_total - offset)}
            response = http.get("https://datasets-server.huggingface.co/rows", params=params, timeout=60)
            response.raise_for_status(); payload = response.json()
            assert payload["num_rows_total"] == expected_total
            for value in payload["rows"]:
                record = value["row"]
                # Exclude audio URLs/bytes from saved records and logs.
                collected.append({k: record[k] for k in ["id", "language", "duration", "silence_spans", "words", "messages"]})
            receipts.append({"config": language, "offset": offset, "rows": len(payload["rows"]), "http_status": response.status_code})
            time.sleep(.2)
        assert len(collected) == expected_total
        rows.extend(collected)
    after = http.get("https://huggingface.co/api/datasets/livekit/eot-bench-data", timeout=45)
    after.raise_for_status(); assert after.json()["sha"] == revision, "Dataset changed while paging metadata"
    write_jsonl(root / "turns.jsonl", rows)
    cases = []
    skipped = []
    for row in rows:
        spans = row["silence_spans"]
        for index, span in enumerate(spans):
            if span["end"] - span["start"] < .2 - 1e-6:
                skipped.append({"id": row["id"], "span": index, "reason": "silence_shorter_than_200ms"})
                continue
            timestamp = span["start"] + .2
            visible = [w["word"] for w in row["words"] if w["end"] <= timestamp + 1e-8]
            text = ("" if row["language"] == "zh" else " ").join(visible)
            history = " ".join(m["content"] for m in row["messages"][-2:] if isinstance(m.get("content"), str))
            cases.append({"text": text, "context": history, "language": row["language"], "source_id": row["id"], "span_index": index,
                "label": "EOT" if index == len(spans) - 1 else "NEOT", "timestamp_seconds": timestamp,
                "silence_seconds_at_score": .2, "transcript_lag_seconds": 0., "label_source": "final_silence_eot_earlier_silences_hold",
                "source_url": "https://huggingface.co/datasets/livekit/eot-bench-data", "dataset_revision": revision})
    write_jsonl(root / "cases.jsonl", cases)
    write_json(root / "sources.json", {"dataset": "livekit/eot-bench-data", "url": "https://huggingface.co/datasets/livekit/eot-bench-data",
        "revision_before_and_after": revision, "license": "Apache-2.0 per upstream benchmark README", "split": "validation",
        "turns": len(rows), "cases": len(cases), "skipped": skipped, "requests": receipts, "audio_downloaded": False,
        "cases_sha256": file_hash(root / "cases.jsonl"), "transcript_protocol": "Causal words.end <= silence.start+0.2s; EN space-join; ZH concat; prior2messages",
        "scope": "FROZEN_EXTERNAL_TEXT_AT_REAL_PAUSES; not official audio/policy-sweep leaderboard, not used for training/threshold selection"})
    print(json.dumps({"turns": len(rows), "cases": len(cases), "revision": revision, "audio_downloaded": False}))


if __name__ == "__main__": main()
