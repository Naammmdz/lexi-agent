#!/usr/bin/env python3
"""Build a clean submission directly from retrieval cache.

This does not rely on any previous submission file.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT

DEFAULT_INPUT = REPO_ROOT / "R2AIStage1DATA.json"
DEFAULT_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_rrf_wide_merged.json"
DEFAULT_OUTPUT = REPO_ROOT / "submission_variants" / "submission_cache_only_top1.zip"


def pick_from_cache(
    candidates: list[dict[str, Any]],
    cap_articles: int,
    cap_docs: int,
    companion_max_gap: float,
) -> tuple[list[str], list[str]]:
    docs: list[str] = []
    articles: list[str] = []
    seen_docs: set[str] = set()
    seen_articles: set[str] = set()

    top_score = float(candidates[0].get("score", 0.0) or 0.0) if candidates else 0.0

    for idx, cand in enumerate(candidates):
        if len(articles) >= cap_articles and len(docs) >= cap_docs:
            break

        article_ref = str(cand.get("article_ref", "")).strip()
        doc_ref = str(cand.get("doc_ref", "")).strip()
        score = float(cand.get("score", 0.0) or 0.0)

        # Always allow top-1. Companion slots are gated by score gap.
        if idx > 0 and companion_max_gap >= 0 and (top_score - score) > companion_max_gap:
            continue

        if article_ref and article_ref not in seen_articles and len(articles) < cap_articles:
            articles.append(article_ref)
            seen_articles.add(article_ref)

        if doc_ref and doc_ref not in seen_docs and len(docs) < cap_docs:
            docs.append(doc_ref)
            seen_docs.add(doc_ref)

    return docs, articles


def build_rows(
    input_rows: list[dict[str, Any]],
    cache: dict[str, list[dict[str, Any]]],
    cap_articles: int,
    cap_docs: int,
    companion_max_gap: float,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    no_cache = 0

    for item in input_rows:
        row_id = str(item["id"])
        candidates = cache.get(row_id, [])
        if not candidates:
            no_cache += 1

        docs, articles = pick_from_cache(
            candidates,
            cap_articles=cap_articles,
            cap_docs=cap_docs,
            companion_max_gap=companion_max_gap,
        )
        rows.append(
            {
                "id": item["id"],
                "question": item.get("question", ""),
                "answer": "Theo các quy định pháp luật liên quan.",
                "relevant_docs": docs,
                "relevant_articles": articles,
            }
        )

    return rows, no_cache


def write_zip(rows: list[dict[str, Any]], output_zip: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    output_json = output_zip.with_suffix(".json")
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname="results.json")
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create submission directly from cache.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cap-articles", type=int, default=1)
    parser.add_argument("--cap-docs", type=int, default=1)
    parser.add_argument(
        "--companion-max-gap",
        type=float,
        default=-1.0,
        help="If >=0, only add non-top1 candidates when (top1_score - cand_score) <= this gap.",
    )
    args = parser.parse_args()

    input_rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))

    rows, no_cache = build_rows(
        input_rows,
        cache,
        cap_articles=args.cap_articles,
        cap_docs=args.cap_docs,
        companion_max_gap=args.companion_max_gap,
    )
    output_json = write_zip(rows, Path(args.output))

    print(
        json.dumps(
            {
                "rows": len(rows),
                "cache_rows": len(cache),
                "no_cache_rows": no_cache,
                "cap_articles": args.cap_articles,
                "cap_docs": args.cap_docs,
                "companion_max_gap": args.companion_max_gap,
                "output_zip": args.output,
                "output_json": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
