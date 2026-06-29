"""Boost recall on the curated base by unioning live retrieval candidates.

Motivation
----------
The leaderboard metric is macro F2 (recall weighted 4x precision).  The current
curated submission is precision-heavy but recall-starved (ARTICLES_RECALL ~0.49).

A leaderboard-style benchmark on ``train_qna.csv`` using the *live*
``hybrid_rerank`` pipeline shows a clear optimum:

    k   ARTICLES_F2  ARTICLES_R   DOCS_F2
    1   0.518        0.517        0.671
    2   0.543        0.634        0.734   <- best articles
    3   0.536        0.703        0.749   <- best docs

So emitting ~2 articles and ~3 docs per row maximises F2.  The curated base
averages only 1.57 articles / 1.x docs.  This layer keeps every curated anchor
(precision) and tops each row up with the highest-ranked *live* retrieval
candidates (recall), up to configurable caps.

Live candidates come from ``data/augmented/live_retrieval_test.json`` produced by
``cache_live_retrieval.py`` (the stale root ``results.json`` shares its anchor
with the curated base for only ~9% of rows, so it is unusable as a recall
source).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import (
    append_refs,
    article_key,
    doc_key,
    doc_ref,
    load_rows,
    update_answer,
)
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
LIVE_CACHE_V2 = REPO_ROOT / "data" / "augmented" / "live_retrieval_test_v2.json"
LIVE_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_test.json"


def resolve_live_cache() -> Path:
    if LIVE_CACHE_V2.exists():
        return LIVE_CACHE_V2
    return LIVE_CACHE

DEFAULT_BASE = REPO_ROOT / "submission.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_recall_boost.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_recall_boost_debug.csv"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"


def label_to_article_id(label: str) -> str:
    return label.lower().replace("điều", "").strip()


def article_ref_key(ref: str) -> tuple[str, str]:
    parts = str(ref).split("|")
    law_id = canonical_law_id(parts[0]) if parts else ""
    label = parts[-1].strip().lower() if len(parts) >= 3 else ""
    return law_id, label


def maybe_prefer_cache_top1(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    article_min_score: float,
    replace_min_gap: float,
    cap_docs: int,
) -> bool:
    """Replace row top-1 article with cache top-1 when reranker prefers it."""
    articles = list(row.get("relevant_articles", []))
    if not articles or not candidates:
        return False

    top_cand = candidates[0]
    top_score = float(top_cand.get("score", 0.0) or 0.0)
    if top_score < article_min_score:
        return False

    new_ref = str(top_cand.get("article_ref", "")).strip()
    if not new_ref:
        return False

    current_key = article_ref_key(articles[0])
    new_key = article_ref_key(new_ref)
    if current_key == new_key:
        return False

    current_score = None
    for cand in candidates:
        cand_ref = str(cand.get("article_ref", "")).strip()
        if cand_ref and article_ref_key(cand_ref) == current_key:
            current_score = float(cand.get("score", 0.0) or 0.0)
            break

    if current_score is not None and (top_score - current_score) < replace_min_gap:
        return False

    remaining = [ref for ref in articles if article_ref_key(ref) != new_key]
    row["relevant_articles"] = [new_ref] + remaining

    new_law = new_key[0]
    if new_law:
        existing_docs = list(row.get("relevant_docs", []))
        existing_laws = {doc_key(d) for d in existing_docs if doc_key(d)}
        if new_law not in existing_laws:
            new_doc = doc_ref(new_law, mapping)
            if cap_docs <= 1:
                row["relevant_docs"] = [new_doc]
            else:
                row["relevant_docs"] = [new_doc] + [d for d in existing_docs if doc_key(d) != new_law]
    return True


def trim_row_caps(row: dict[str, Any], cap_articles: int, cap_docs: int) -> None:
    articles = list(row.get("relevant_articles", []))
    if len(articles) > cap_articles:
        row["relevant_articles"] = articles[:cap_articles]
    docs = list(row.get("relevant_docs", []))
    if len(docs) > cap_docs:
        row["relevant_docs"] = docs[:cap_docs]


def boost_single_article_same_law(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    cap_articles: int,
    article_min_score: float,
    article_min_gap_from_top1: float,
) -> None:
    """Add Điều 2/3 only when row has exactly one article and cache agrees (same VB)."""
    articles = list(row.get("relevant_articles", []))
    if len(articles) != 1 or len(candidates) < 2:
        return

    anchor_law = article_ref_key(articles[0])[0]
    if not anchor_law:
        return

    top_score = float(candidates[0].get("score", 0.0) or 0.0)
    existing_keys = {article_key(r) for r in articles if article_key(r)}

    for cand in candidates[1:]:
        if len({article_key(r) for r in row.get("relevant_articles", [])}) >= cap_articles:
            break
        score = float(cand.get("score", 0.0) or 0.0)
        if score < article_min_score:
            continue
        if (top_score - score) > article_min_gap_from_top1:
            continue
        cand_law = canonical_law_id(cand.get("law_id", ""))
        if cand_law != anchor_law:
            continue
        article_id = label_to_article_id(cand.get("label", ""))
        if not article_id:
            continue
        key = (cand_law, article_id)
        if key in existing_keys:
            continue
        append_refs(row, mapping, [key])
        existing_keys.add(key)


def boost_row(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    cap_articles: int,
    cap_docs: int,
    min_score: float,
    article_same_law_only: bool,
    add_new_docs: bool,
    article_min_score: float,
    article_min_gap_from_top1: float,
    prefer_cache_top1: bool = False,
    replace_top1_min_gap: float = 0.0,
    single_article_same_law_boost: bool = False,
) -> None:
    if single_article_same_law_boost:
        boost_single_article_same_law(
            row,
            candidates,
            mapping,
            cap_articles,
            article_min_score,
            article_min_gap_from_top1,
        )
        return

    if prefer_cache_top1:
        maybe_prefer_cache_top1(
            row,
            candidates,
            mapping,
            article_min_score=article_min_score,
            replace_min_gap=replace_top1_min_gap,
            cap_docs=cap_docs,
        )
    existing_doc_keys = {doc_key(r) for r in row.get("relevant_docs", [])}
    # Phase A: top up articles (also pulls in their parent doc).
    for cand in candidates:
        if len({article_key(r) for r in row.get("relevant_articles", [])}) >= cap_articles:
            break
        if cand["score"] < min_score:
            continue
        if cand["score"] < article_min_score:
            continue
        cand_law = canonical_law_id(cand.get("law_id", ""))
        if article_same_law_only and cand_law not in existing_doc_keys:
            continue
        top_score = float(candidates[0]["score"]) if candidates else 0.0
        # Only add companion article when retrieval is ambiguous enough (small gap).
        if (top_score - float(cand["score"])) > article_min_gap_from_top1:
            continue
        article_id = label_to_article_id(cand.get("label", ""))
        if not article_id:
            continue
        append_refs(row, mapping, [(cand_law, article_id)])

    # Phase B: top up docs only (document-level recall).
    if add_new_docs:
        for cand in candidates:
            existing_docs = {doc_key(r) for r in row.get("relevant_docs", [])}
            if len(existing_docs) >= cap_docs:
                break
            if cand["score"] < min_score:
                continue
            law_id = canonical_law_id(cand["law_id"])
            if law_id in existing_docs:
                continue
            row.setdefault("relevant_docs", []).append(doc_ref(law_id, mapping))

    if prefer_cache_top1:
        trim_row_caps(row, cap_articles, cap_docs)


def create_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    cap_articles: int,
    cap_docs: int,
    min_score: float,
    article_same_law_only: bool,
    add_new_docs: bool,
    article_min_score: float,
    article_min_gap_from_top1: float,
    copy_to_submission: bool,
    live_cache: Path | None = None,
    mapping_path: Path | None = None,
    prefer_cache_top1: bool = False,
    replace_top1_min_gap: float = 0.0,
    single_article_same_law_boost: bool = False,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path or MAPPING_PATH)
    rows = load_rows(base_zip)
    cache_path = live_cache or resolve_live_cache()
    live = json.loads(cache_path.read_text(encoding="utf-8"))

    debug_rows: list[dict[str, Any]] = []
    top1_replaced = 0
    for row in rows:
        row_id = str(row["id"])
        candidates = live.get(row_id, [])
        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        before_top = before_articles[0] if before_articles else ""
        boost_row(
            row,
            candidates,
            mapping,
            cap_articles,
            cap_docs,
            min_score,
            article_same_law_only=article_same_law_only,
            add_new_docs=add_new_docs,
            article_min_score=article_min_score,
            article_min_gap_from_top1=article_min_gap_from_top1,
            prefer_cache_top1=prefer_cache_top1,
            replace_top1_min_gap=replace_top1_min_gap,
            single_article_same_law_boost=single_article_same_law_boost,
        )
        after_top = row.get("relevant_articles", [""])[0] if row.get("relevant_articles") else ""
        if before_top and after_top and before_top != after_top:
            top1_replaced += 1
        update_answer(row)
        if row.get("relevant_articles") != before_articles or row.get("relevant_docs") != before_docs:
            debug_rows.append(
                {
                    "id": row_id,
                    "question": row.get("question", ""),
                    "before_articles": " || ".join(before_articles),
                    "after_articles": " || ".join(row.get("relevant_articles", [])),
                    "before_docs": " || ".join(before_docs),
                    "after_docs": " || ".join(row.get("relevant_docs", [])),
                }
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    with debug_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "question", "before_articles", "after_articles", "before_docs", "after_docs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    return {
        "rows": len(rows),
        "changed_rows": len(debug_rows),
        "avg_articles": round(sum(len(r.get("relevant_articles", [])) for r in rows) / len(rows), 3),
        "avg_docs": round(sum(len(r.get("relevant_docs", [])) for r in rows) / len(rows), 3),
        "cap_articles": cap_articles,
        "cap_docs": cap_docs,
        "min_score": min_score,
        "article_same_law_only": article_same_law_only,
        "add_new_docs": add_new_docs,
        "article_min_score": article_min_score,
        "article_min_gap_from_top1": article_min_gap_from_top1,
        "prefer_cache_top1": prefer_cache_top1,
        "replace_top1_min_gap": replace_top1_min_gap,
        "top1_replaced": top1_replaced,
        "single_article_same_law_boost": single_article_same_law_boost,
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--cap-articles", type=int, default=2)
    parser.add_argument("--cap-docs", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--article-same-law-only", action="store_true")
    parser.add_argument("--add-new-docs", action="store_true")
    parser.add_argument("--article-min-score", type=float, default=0.0)
    parser.add_argument("--article-min-gap-from-top1", type=float, default=0.0)
    parser.add_argument("--live-cache", default="")
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--copy-to-submission", action="store_true")
    parser.add_argument("--prefer-cache-top1", action="store_true")
    parser.add_argument("--replace-top1-min-gap", type=float, default=0.0)
    parser.add_argument(
        "--single-article-same-law-boost",
        action="store_true",
        help="Only boost rows with exactly 1 article; add cache #2+ same VB if score≥min and gap≤threshold",
    )
    args = parser.parse_args()
    stats = create_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        args.cap_articles,
        args.cap_docs,
        args.min_score,
        args.article_same_law_only,
        args.add_new_docs,
        args.article_min_score,
        args.article_min_gap_from_top1,
        args.copy_to_submission,
        live_cache=Path(args.live_cache) if args.live_cache else None,
        mapping_path=Path(args.mapping) if args.mapping else None,
        prefer_cache_top1=args.prefer_cache_top1,
        replace_top1_min_gap=args.replace_top1_min_gap,
        single_article_same_law_boost=args.single_article_same_law_boost,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
