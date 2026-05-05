"""Build the arXiv prompt corpus.

Run once. Output: client/prompts/arxiv_corpus.jsonl, one JSON object per
line with at least a `text` field.

Strategy. We use the arXiv REST API (https://export.arxiv.org/api/query),
which returns Atom XML and respects a published rate limit (3 s
between requests is the polite recommendation). We sample papers across
several CS categories to get topic variety, harvest titles + abstracts,
and concatenate them into one block per item. We do NOT download full
text; abstracts plus titles are enough to reach a few thousand tokens
when concatenated, which is what the sampler needs.

The script is deliberately simple and slow. Run it once, commit the
output, and never run it again unless you want a fresh corpus.

Usage:

    python build_corpus.py --output prompts/arxiv_corpus.jsonl --target 3000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET


ARXIV_API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
}

# A spread of CS categories, plus a couple of others for variety.
CATEGORIES = [
    "cs.SE",
    "cs.DC",
    "cs.AI",
    "cs.LG",
    "cs.OS",
    "cs.PF",
    "cs.CR",
    "cs.NI",
    "cs.SY",
    "stat.ML",
    "eess.SP",
]


def fetch_batch(category: str, start: int, max_results: int) -> list[dict]:
    params = {
        "search_query": f"cat:{category}",
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "llm-serving-bench/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    root = ET.fromstring(body)
    items: list[dict] = []
    for entry in root.findall("atom:entry", NS):
        title_el = entry.find("atom:title", NS)
        summary_el = entry.find("atom:summary", NS)
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (summary_el.text or "").strip() if summary_el is not None else ""
        if not summary:
            continue
        items.append({"category": category, "title": title, "abstract": summary})
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target", type=int, default=3000, help="Approximate number of corpus items to collect.")
    p.add_argument("--per-category", type=int, default=300)
    p.add_argument("--polite-delay-s", type=float, default=3.0)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen_titles: set[str] = set()
    n_written = 0
    page_size = 100

    with args.output.open("w") as out:
        for cat in CATEGORIES:
            collected_for_cat = 0
            start = 0
            while collected_for_cat < args.per_category and n_written < args.target:
                try:
                    batch = fetch_batch(cat, start=start, max_results=page_size)
                except Exception as e:
                    print(f"[warn] fetch {cat} start={start} failed: {e}")
                    time.sleep(args.polite_delay_s * 2)
                    break
                if not batch:
                    break
                for item in batch:
                    title = item["title"]
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    text = f"{item['title']}\n\n{item['abstract']}"
                    record = {"text": text, "title": title, "category": item["category"]}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
                    collected_for_cat += 1
                    if n_written >= args.target:
                        break
                start += page_size
                time.sleep(args.polite_delay_s)
            print(f"[ok] {cat}: collected {collected_for_cat}, total {n_written}")
            if n_written >= args.target:
                break

    print(f"[done] wrote {n_written} items to {args.output}")


if __name__ == "__main__":
    main()
