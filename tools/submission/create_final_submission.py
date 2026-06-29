"""Build final submission: audited repairs + recall boost from live cache.

Pipeline:
1. Start from best verified base (tight_v1 @ ART_F2=0.51).
2. Apply guarded off-domain repairs only when audit flags domain mismatch.
3. Apply same-law companion boost from live retrieval cache (v2 preferred).
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
from audit_suspicious_rows import DOMAIN_TERMS, LAW_DOMAIN_HINTS, domains_from_text
from create_domain_repair_submission import (
    append_refs,
    article_key,
    doc_key,
    load_rows,
    set_refs,
    update_answer,
)
from create_train_exact_repair_submission import REPLACE_BY_ID, clean_refs
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
CACHE_V2 = REPO_ROOT / "data" / "augmented" / "live_retrieval_test_v2.json"
CACHE_V1 = REPO_ROOT / "data" / "augmented" / "live_retrieval_test.json"

DEFAULT_BASE = OUTPUT_DIR / "submission_recall_boost_tight_v1.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_final_v1.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_final_v1_debug.csv"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"


def label_to_article_id(label: str) -> str:
    return label.lower().replace("điều", "").strip()


def is_off_domain(row: dict[str, Any]) -> bool:
    qdomains = domains_from_text(row.get("question", ""), DOMAIN_TERMS)
    if not qdomains:
        return False
    doc_domains: set[str] = set()
    for ref in row.get("relevant_docs", []):
        doc_domains.update(domains_from_text(ref, LAW_DOMAIN_HINTS))
    return not (qdomains & doc_domains)


def apply_audited_repairs(row: dict[str, Any], mapping: dict[str, str]) -> str:
    row_id = int(row["id"])
    if row_id not in REPLACE_BY_ID:
        return ""
    if not is_off_domain(row):
        return ""
    reason, refs = REPLACE_BY_ID[row_id]
    set_refs(row, mapping, clean_refs(refs))
    return reason


def apply_recall_boost(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    cap_articles: int,
    article_min_score: float,
    article_max_gap: float,
) -> bool:
    existing_docs = {doc_key(r) for r in row.get("relevant_docs", []) if doc_key(r)}
    if not candidates:
        return False
    global_score = float(candidates[0].get("score", 0.0) or 0.0)
    changed = False
    for cand in candidates[1:]:
        if len({article_key(r) for r in row.get("relevant_articles", []) if article_key(r)}) >= cap_articles:
            break
        score = float(cand.get("score", 0.0) or 0.0)
        if score < article_min_score:
            continue
        if (global_score - score) > article_max_gap:
            continue
        law = canonical_law_id(cand.get("law_id", ""))
        if law not in existing_docs:
            continue
        art = label_to_article_id(cand.get("label", ""))
        if not art:
            continue
        before = len(row.get("relevant_articles", []))
        append_refs(row, mapping, [(law, art)])
        if len(row.get("relevant_articles", [])) > before:
            changed = True
    return changed


def create_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    cache_path: Path,
    cap_articles: int,
    article_min_score: float,
    article_max_gap: float,
    copy_to_submission: bool,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    live = json.loads(cache_path.read_text(encoding="utf-8"))
    debug_rows: list[dict[str, Any]] = []
    repair_count = 0
    boost_count = 0

    for row in rows:
        row_id = str(row["id"])
        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        modes: list[str] = []

        reason = apply_audited_repairs(row, mapping)
        if reason:
            modes.append(f"repair:{reason}")
            repair_count += 1

        if apply_recall_boost(
            row,
            live.get(row_id, []),
            mapping,
            cap_articles=cap_articles,
            article_min_score=article_min_score,
            article_max_gap=article_max_gap,
        ):
            modes.append("recall_boost")
            boost_count += 1

        if modes:
            update_answer(row)
            debug_rows.append(
                {
                    "id": row_id,
                    "modes": " || ".join(modes),
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
        fieldnames = ["id", "modes", "question", "before_articles", "after_articles", "before_docs", "after_docs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    return {
        "rows": len(rows),
        "changed_rows": len(debug_rows),
        "repair_rows": repair_count,
        "boost_rows": boost_count,
        "avg_articles": round(sum(len(r.get("relevant_articles", [])) for r in rows) / len(rows), 3),
        "avg_docs": round(sum(len(r.get("relevant_docs", [])) for r in rows) / len(rows), 3),
        "cache": str(cache_path),
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--cache", default="")
    parser.add_argument("--cap-articles", type=int, default=2)
    parser.add_argument("--article-min-score", type=float, default=0.9)
    parser.add_argument("--article-max-gap", type=float, default=0.03)
    parser.add_argument("--copy-to-submission", action="store_true")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else (CACHE_V2 if CACHE_V2.exists() else CACHE_V1)
    stats = create_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        cache_path,
        args.cap_articles,
        args.article_min_score,
        args.article_max_gap,
        args.copy_to_submission,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
