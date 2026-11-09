"""Collect a bounded set of Vietnamese Wikipedia extracts with provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import requests

from .common import digest, normalize, write_json, write_jsonl

DEFAULT_TITLES = ["Hà Nội", "Đà Nẵng", "Du lịch Việt Nam", "Giao thông Việt Nam"]


def candidates_from_extract(title, page_id, extract):
    source_id = f"viwiki:{page_id}"
    url = "https://vi.wikipedia.org/?curid=" + str(page_id)
    results, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", extract):
        text = normalize(sentence)
        words = text.split()
        if not 8 <= len(words) <= 80 or text in seen:
            continue
        seen.add(text)
        # These are CANDIDATES; a cut prefix is not automatically labeled NEOT.
        texts = [text, " ".join(words[:max(2, int(len(words) * 0.5))]),
                 " ".join(words[:max(2, int(len(words) * 0.85))])]
        for current in dict.fromkeys(texts):
            results.append({"candidate_id": digest(source_id + "\0" + current),
                "text": current, "context": "", "source_id": source_id,
                "candidate_kind": "sentence" if current == text else "prefix",
                "source_url": url, "source_title": title,
                "source_license": "Wikipedia text terms; see https://vi.wikipedia.org/wiki/Wikipedia:Quyền_tác_giả",
                "source_sentence_sha256": digest(sentence), "label_source": "unlabeled_crawl"})
    return results


def iter_candidates(titles, max_candidates=300, sources=None):
    """Yield one candidate at a time without fetching subsequent pages first."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    sources = [] if sources is None else sources
    http = requests.Session()
    http.headers["User-Agent"] = "VietnameseTurnDetector/0.1 (bounded notebook research; provenance preserved)"
    emitted, seen = 0, set()
    for title in titles:
        response = http.get("https://vi.wikipedia.org/w/api.php", params={"action": "query",
            "format": "json", "formatversion": 2, "prop": "extracts", "explaintext": 1,
            "redirects": 1, "titles": title}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError("Wikipedia API error: " + payload["error"].get("code", "unknown"))
        for page in payload.get("query", {}).get("pages", []):
            if page.get("missing") or not page.get("extract"):
                continue
            candidates = candidates_from_extract(page["title"], page["pageid"], page["extract"])
            sources.append({"title": page["title"], "page_id": page["pageid"],
                            "extract_sha256": digest(page["extract"]), "candidates": len(candidates)})
            for row in candidates:
                if row["candidate_id"] in seen:
                    continue
                seen.add(row["candidate_id"])
                yield row
                emitted += 1
                if emitted >= max_candidates:
                    return
        time.sleep(0.5)


def crawl(titles, output, max_candidates=300):
    sources = []
    rows = list(iter_candidates(titles, max_candidates, sources))
    if not rows:
        raise RuntimeError("No usable candidates were collected")
    write_jsonl(output, rows)
    write_json(output.with_suffix(".sources.json"), {"sources": sources, "candidates": len(rows),
        "labels_created": False, "domain_limit": "Wikipedia prose is not a representative speech/dialogue benchmark"})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titles", nargs="+", default=DEFAULT_TITLES)
    parser.add_argument("--output", type=Path, default=Path("data_v2/candidates.jsonl"))
    parser.add_argument("--max-candidates", type=int, default=300)
    args = parser.parse_args()
    rows = crawl(args.titles, args.output, args.max_candidates)
    print(json.dumps({"candidates": len(rows), "output": str(args.output)}))
