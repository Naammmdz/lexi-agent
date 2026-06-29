"""Build Zalo-format gap corpus from Hugging Face vbpl.vn legal documents.

Source dataset (public, reproducible):
  https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents

Harvests QH / NĐ-CP / Thông tư missing from the Zalo 2021 corpus, parses HTML
full-text into article records, and writes gap JSON for ``build_merged_corpus_v2``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corpus.vlsp_html_parser import doc_name_to_title, parse_vlsp_document
from utils.submission_formatter import canonical_law_id, format_law_title

HF_DATASET = "th1nhng0/vietnamese-legal-documents"
ZALO_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus.json"
MERGED_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
OUTPUT_CORPUS = REPO_ROOT / "data" / "augmented" / "hf_vbpl_gap_corpus.json"
OUTPUT_TITLES = REPO_ROOT / "data" / "augmented" / "hf_vbpl_gap_titles.json"
OUTPUT_REPORT = REPO_ROOT / "data" / "augmented" / "hf_vbpl_gap_report.json"

ALLOWED = re.compile(r"/(QH\d+|NĐ-CP|TT-[A-ZĐ]+)$", re.IGNORECASE)
SKIP = re.compile(r"/QĐ-", re.IGNORECASE)


def load_existing_law_ids() -> set[str]:
    ids: set[str] = set()
    for path in (ZALO_CORPUS, MERGED_CORPUS):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ids.update(canonical_law_id(doc.get("law_id", "")) for doc in data if doc.get("law_id"))
    return ids


def allowed_law(law_id: str) -> bool:
    law_id = canonical_law_id(law_id)
    if not law_id or SKIP.search(law_id):
        return False
    return bool(ALLOWED.search(law_id))


def collect_metadata_candidates(existing: set[str]) -> dict[str, dict[str, str]]:
    """Map HF document id -> {law_id, title} for laws missing from Zalo."""
    stream = load_dataset(HF_DATASET, "metadata", split="data", streaming=True)
    by_law: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen = 0
    for row in stream:
        seen += 1
        if seen % 50000 == 0:
            print(f"  metadata scanned {seen:,}, candidate laws {len(by_law):,}", flush=True)
        law_id = canonical_law_id(str(row.get("so_ky_hieu") or ""))
        if not law_id or law_id in existing or not allowed_law(law_id):
            continue
        by_law[law_id].append(
            {
                "hf_id": str(row.get("id") or ""),
                "title": str(row.get("title") or "").strip(),
                "ngay_ban_hanh": str(row.get("ngay_ban_hanh") or ""),
            }
        )
    print(f"  metadata done: {seen:,} rows, {len(by_law):,} unique gap laws", flush=True)

    candidates: dict[str, dict[str, str]] = {}
    for law_id, rows in by_law.items():
        rows.sort(key=lambda r: r.get("ngay_ban_hanh") or "", reverse=True)
        best = rows[0]
        if best.get("hf_id"):
            candidates[best["hf_id"]] = {"law_id": law_id, "title": best["title"]}
    return candidates


def harvest_content(
    candidates: dict[str, dict[str, str]],
) -> tuple[list[dict], dict[str, str], dict[str, Any]]:
    wanted = set(candidates)
    grouped_html: dict[str, list[str]] = defaultdict(list)
    stream = load_dataset(HF_DATASET, "content", split="data", streaming=True)
    seen = 0
    for row in stream:
        seen += 1
        if seen % 50000 == 0:
            print(f"  content scanned {seen:,}, matched {len(grouped_html):,}", flush=True)
        doc_id = str(row.get("id") or "")
        if doc_id not in wanted:
            continue
        html = str(row.get("content_html") or "")
        if html.strip():
            grouped_html[doc_id].append(html)

    documents: list[dict] = []
    titles: dict[str, str] = {}
    report: dict[str, Any] = {
        "source": HF_DATASET,
        "source_url": "https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents",
        "license": "CC BY 4.0",
        "found": [],
        "empty": [],
        "law_count": 0,
        "article_count": 0,
    }

    for hf_id, meta in sorted(candidates.items(), key=lambda x: x[1]["law_id"]):
        law_id = meta["law_id"]
        html_chunks = grouped_html.get(hf_id) or []
        articles: list[dict[str, str]] = []
        for html in html_chunks:
            articles = parse_vlsp_document(html)
            if articles:
                break
        if not articles:
            report["empty"].append(law_id)
            continue
        raw_title = meta.get("title") or law_id
        titles[law_id] = format_law_title(law_id, doc_name_to_title(raw_title, law_id))
        documents.append(
            {
                "law_id": law_id.lower(),
                "articles": articles,
                "source": "hf_vbpl",
                "hf_dataset": HF_DATASET,
                "hf_doc_id": hf_id,
                "title": raw_title,
            }
        )
        report["found"].append({"law_id": law_id, "articles": len(articles), "hf_id": hf_id})

    report["law_count"] = len(documents)
    report["article_count"] = sum(len(d.get("articles", [])) for d in documents)
    return documents, titles, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harvest gap laws from th1nhng0/vietnamese-legal-documents (vbpl.vn)."
    )
    parser.add_argument("--max-laws", type=int, default=0, help="Cap laws (0=all).")
    args = parser.parse_args()

    existing = load_existing_law_ids()
    print(f"existing_laws={len(existing)} (zalo+merged)", flush=True)

    candidates = collect_metadata_candidates(existing)
    if args.max_laws and len(candidates) > args.max_laws:
        keys = sorted(candidates)[: args.max_laws]
        candidates = {k: candidates[k] for k in keys}
        print(f"capped to max_laws={args.max_laws}", flush=True)

    documents, titles, report = harvest_content(candidates)
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CORPUS.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TITLES.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"hf_vbpl_gap laws={report['law_count']} articles={report['article_count']} "
        f"empty={len(report['empty'])}",
        flush=True,
    )
    print(f"wrote {OUTPUT_CORPUS}", flush=True)


if __name__ == "__main__":
    main()
