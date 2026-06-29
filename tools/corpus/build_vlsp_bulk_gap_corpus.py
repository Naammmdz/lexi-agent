"""Bulk VLSP harvest: all QH / NĐ-CP / TT laws missing from Zalo corpus."""

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

from tools.corpus.vlsp_html_parser import doc_name_to_title, parse_vlsp_document, pick_best_row
from utils.submission_formatter import canonical_law_id, format_law_title

ZALO_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus.json"
MERGED_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
OUTPUT_CORPUS = REPO_ROOT / "data" / "augmented" / "vlsp_bulk_gap_corpus.json"
OUTPUT_TITLES = REPO_ROOT / "data" / "augmented" / "vlsp_bulk_gap_titles.json"
OUTPUT_REPORT = REPO_ROOT / "data" / "augmented" / "vlsp_bulk_gap_report.json"

# QH, NĐ-CP, TT — skip QĐ-UBND and similar local decisions.
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


def stream_group_missing(existing: set[str]) -> dict[str, list[dict[str, Any]]]:
    dataset = load_dataset("VLSP2025-LegalSML/legal-pretrain", split="train", streaming=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = 0
    for row in dataset:
        seen += 1
        if seen % 50000 == 0:
            print(f"  scanned {seen:,} rows, candidates {len(grouped):,}", flush=True)
        meta = row.get("metadata") or {}
        ident = canonical_law_id(str(meta.get("DocIdentity") or ""))
        if not ident or ident in existing or not allowed_law(ident):
            continue
        grouped[ident].append(row)
    print(f"  done scan {seen:,} rows, unique gap laws {len(grouped):,}", flush=True)
    return grouped


def build_documents(
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict], dict[str, str], dict[str, Any]]:
    documents: list[dict] = []
    titles: dict[str, str] = {}
    report: dict[str, Any] = {"found": [], "empty": [], "law_count": 0, "article_count": 0}

    for law_id in sorted(grouped):
        rows = grouped[law_id]
        row = pick_best_row(rows)
        meta = row.get("metadata") or {}
        articles = parse_vlsp_document(row.get("doc_content") or "")
        if not articles:
            report["empty"].append(law_id)
            continue
        raw_title = doc_name_to_title(str(meta.get("DocName") or ""), law_id)
        titles[law_id] = format_law_title(law_id, raw_title)
        documents.append(
            {
                "law_id": law_id.lower(),
                "articles": articles,
                "source": "vlsp2025_bulk",
                "doc_name": meta.get("DocName"),
                "issue_date": str(meta.get("IssueDate") or ""),
                "organ_name": meta.get("OrganName"),
            }
        )
        report["found"].append({"law_id": law_id, "articles": len(articles), "variants": len(rows)})

    report["law_count"] = len(documents)
    report["article_count"] = sum(len(d.get("articles", [])) for d in documents)
    return documents, titles, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk VLSP gap harvest (QH/NĐ-CP/TT).")
    parser.add_argument("--max-laws", type=int, default=0, help="Cap laws (0=all).")
    args = parser.parse_args()

    existing = load_existing_law_ids()
    print(f"existing_laws={len(existing)} (zalo+merged)", flush=True)

    grouped = stream_group_missing(existing)
    if args.max_laws and len(grouped) > args.max_laws:
        keys = sorted(grouped.keys())[: args.max_laws]
        grouped = {k: grouped[k] for k in keys}
        print(f"capped to max_laws={args.max_laws}", flush=True)

    documents, titles, report = build_documents(grouped)
    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CORPUS.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TITLES.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"bulk_gap laws={report['law_count']} articles={report['article_count']} "
        f"empty={len(report['empty'])}",
        flush=True,
    )
    print(f"wrote {OUTPUT_CORPUS}", flush=True)


if __name__ == "__main__":
    main()
