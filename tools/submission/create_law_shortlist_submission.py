"""Create submission using law-shortlist + within-law article selection.

This is an alternative to pure threshold gating. It first scores laws from live
retrieval candidates, then picks top articles within the shortlisted laws.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import append_refs, article_key, doc_key, load_rows, update_answer
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
LIVE_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_test.json"

DEFAULT_BASE = REPO_ROOT / "submission.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_law_shortlist.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_law_shortlist_debug.csv"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"


def label_to_article_id(label: str) -> str:
    return label.lower().replace("điều", "").strip()


def shortlist_laws(candidates: list[dict[str, Any]], top_laws: int) -> list[str]:
    law_scores: dict[str, float] = defaultdict(float)
    for cand in candidates:
        law = canonical_law_id(cand.get("law_id", ""))
        if not law:
            continue
        law_scores[law] += float(cand.get("score", 0.0) or 0.0)
    ranked = sorted(law_scores.items(), key=lambda x: x[1], reverse=True)
    return [law for law, _ in ranked[:top_laws]]


def boost_row(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    top_laws: int,
    cap_articles: int,
    min_score: float,
    max_gap_from_top1: float,
    allow_top1_new_law: bool,
    keep_docs_fixed: bool,
) -> None:
    shortlisted = set(shortlist_laws(candidates, top_laws=top_laws))
    if not shortlisted:
        return

    # Keep base laws to avoid aggressive off-domain drift.
    base_laws = {doc_key(ref) for ref in row.get("relevant_docs", [])}
    allowed_laws = shortlisted | {law for law in base_laws if law}
    if keep_docs_fixed:
        allowed_laws = {law for law in base_laws if law}
        if allow_top1_new_law and candidates:
            top1_law = canonical_law_id(candidates[0].get("law_id", ""))
            if top1_law:
                allowed_laws.add(top1_law)

    top_score = float(candidates[0].get("score", 0.0) or 0.0) if candidates else 0.0
    for cand in candidates:
        if len({article_key(r) for r in row.get("relevant_articles", []) if article_key(r)}) >= cap_articles:
            break
        score = float(cand.get("score", 0.0) or 0.0)
        if score < min_score:
            continue
        if (top_score - score) > max_gap_from_top1:
            continue
        law = canonical_law_id(cand.get("law_id", ""))
        if law not in allowed_laws:
            continue
        article_id = label_to_article_id(cand.get("label", ""))
        if not article_id:
            continue
        if keep_docs_fixed and law not in {law for law in base_laws if law}:
            # Do not let append_refs expand document set in strict mode.
            continue
        append_refs(row, mapping, [(law, article_id)])


def create_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    top_laws: int,
    cap_articles: int,
    min_score: float,
    max_gap_from_top1: float,
    allow_top1_new_law: bool,
    keep_docs_fixed: bool,
    copy_to_submission: bool,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    live = json.loads(LIVE_CACHE.read_text(encoding="utf-8"))

    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        candidates = live.get(row_id, [])
        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        boost_row(
            row,
            candidates,
            mapping,
            top_laws=top_laws,
            cap_articles=cap_articles,
            min_score=min_score,
            max_gap_from_top1=max_gap_from_top1,
            allow_top1_new_law=allow_top1_new_law,
            keep_docs_fixed=keep_docs_fixed,
        )
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
        "top_laws": top_laws,
        "cap_articles": cap_articles,
        "min_score": min_score,
        "max_gap_from_top1": max_gap_from_top1,
        "allow_top1_new_law": allow_top1_new_law,
        "keep_docs_fixed": keep_docs_fixed,
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--top-laws", type=int, default=2)
    parser.add_argument("--cap-articles", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.88)
    parser.add_argument("--max-gap-from-top1", type=float, default=0.03)
    parser.add_argument("--allow-top1-new-law", action="store_true")
    parser.add_argument("--keep-docs-fixed", action="store_true")
    parser.add_argument("--copy-to-submission", action="store_true")
    args = parser.parse_args()

    stats = create_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        top_laws=args.top_laws,
        cap_articles=args.cap_articles,
        min_score=args.min_score,
        max_gap_from_top1=args.max_gap_from_top1,
        allow_top1_new_law=args.allow_top1_new_law,
        keep_docs_fixed=args.keep_docs_fixed,
        copy_to_submission=args.copy_to_submission,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
