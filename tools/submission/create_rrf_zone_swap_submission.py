#!/usr/bin/env python3
"""Swap top-1 only for rows covered by a partial RRF cache zone."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import update_answer
from create_recall_boost_submission import article_ref_key, maybe_prefer_cache_top1
from utils.cross_doc_cap import apply_conditional_second_doc
from utils.submission_formatter import load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
DEFAULT_BASE = OUTPUT_DIR / "submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.json"
DEFAULT_ZONE_CACHE = REPO_ROOT / "data" / "augmented/live_retrieval_rrf_wide_merged.json"
DEFAULT_MAPPING = REPO_ROOT / "data/law_id_to_title_merged.json"


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def write_submission(rows: list[dict[str, Any]], output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")


def create_rrf_zone_swap(
    base_path: Path,
    output_zip: Path,
    debug_path: Path,
    zone_cache_path: Path,
    mapping_path: Path,
    replace_min_gap: float,
    article_min_score: float,
    cap_docs: int,
    exclude_ids: set[str] | None = None,
    conditional_cap_docs_2: bool = False,
    cross_doc_min_score: float = 0.9,
    cross_doc_max_gap: float = 0.03,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path)
    rows = load_rows(base_path)
    cache = json.loads(zone_cache_path.read_text(encoding="utf-8"))
    zone_ids = set(cache.keys())
    exclude_ids = exclude_ids or set()

    debug_rows: list[dict[str, Any]] = []
    swapped = 0
    skipped_gap = 0
    skipped_score = 0
    skipped_same = 0
    skipped_exclude = 0
    zone_seen = 0
    second_doc_added = 0

    for row in rows:
        row_id = str(row["id"])
        if row_id not in zone_ids:
            continue
        zone_seen += 1
        if row_id in exclude_ids:
            skipped_exclude += 1
            continue
        candidates = cache.get(row_id, [])
        if not candidates:
            continue

        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        if not before_articles:
            continue

        top_cand = candidates[0]
        top_score = float(top_cand.get("score", 0.0) or 0.0)
        if top_score < article_min_score:
            skipped_score += 1
            continue

        new_ref = str(top_cand.get("article_ref", "")).strip()
        if not new_ref:
            continue

        current_key = article_ref_key(before_articles[0])
        new_key = article_ref_key(new_ref)
        if current_key == new_key:
            skipped_same += 1
            continue

        current_score = None
        for cand in candidates:
            cand_ref = str(cand.get("article_ref", "")).strip()
            if cand_ref and article_ref_key(cand_ref) == current_key:
                current_score = float(cand.get("score", 0.0) or 0.0)
                break

        if current_score is not None and (top_score - current_score) < replace_min_gap:
            skipped_gap += 1
            continue

        if maybe_prefer_cache_top1(
            row,
            candidates,
            mapping,
            article_min_score=article_min_score,
            replace_min_gap=replace_min_gap,
            cap_docs=cap_docs,
        ):
            update_answer(row)
            swapped += 1
            debug_rows.append(
                {
                    "id": row_id,
                    "gap": round(top_score - (current_score or 0.0), 4),
                    "top_score": round(top_score, 4),
                    "question": row.get("question", ""),
                    "before_articles": " || ".join(before_articles),
                    "after_articles": " || ".join(row.get("relevant_articles", [])),
                    "before_docs": " || ".join(before_docs),
                    "after_docs": " || ".join(row.get("relevant_docs", [])),
                }
            )

        if conditional_cap_docs_2 and candidates:
            before_docs = list(row.get("relevant_docs", []))
            if apply_conditional_second_doc(
                row,
                candidates,
                mapping,
                min_score=cross_doc_min_score,
                max_gap=cross_doc_max_gap,
                cap_docs=2,
            ):
                update_answer(row)
                second_doc_added += 1
                if not any(d.get("id") == row_id for d in debug_rows):
                    debug_rows.append(
                        {
                            "id": row_id,
                            "gap": "",
                            "top_score": "",
                            "question": row.get("question", ""),
                            "before_articles": " || ".join(row.get("relevant_articles", [])),
                            "after_articles": " || ".join(row.get("relevant_articles", [])),
                            "before_docs": " || ".join(before_docs),
                            "after_docs": " || ".join(row.get("relevant_docs", [])),
                        }
                    )

    write_submission(rows, output_zip)
    with debug_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "gap",
                "top_score",
                "question",
                "before_articles",
                "after_articles",
                "before_docs",
                "after_docs",
            ],
        )
        writer.writeheader()
        writer.writerows(debug_rows)

    return {
        "rows": len(rows),
        "zone_ids": len(zone_ids),
        "zone_seen": zone_seen,
        "top1_swapped": swapped,
        "skipped_same": skipped_same,
        "skipped_gap": skipped_gap,
        "skipped_score": skipped_score,
        "skipped_exclude": skipped_exclude,
        "second_doc_added": second_doc_added,
        "conditional_cap_docs_2": conditional_cap_docs_2,
        "replace_min_gap": replace_min_gap,
        "article_min_score": article_min_score,
        "output": str(output_zip),
        "debug": str(debug_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(OUTPUT_DIR / "rrf_swap_v1.zip"))
    parser.add_argument("--debug", default=str(OUTPUT_DIR / "rrf_swap_v1_debug.csv"))
    parser.add_argument("--zone-cache", default=str(DEFAULT_ZONE_CACHE))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--replace-min-gap", type=float, default=0.03)
    parser.add_argument("--article-min-score", type=float, default=0.9)
    parser.add_argument("--cap-docs", type=int, default=1)
    parser.add_argument(
        "--conditional-cap-docs-2",
        action="store_true",
        help="Add 2nd doc when top-2 retrieval laws differ with close scores",
    )
    parser.add_argument("--cross-doc-min-score", type=float, default=0.9)
    parser.add_argument("--cross-doc-max-gap", type=float, default=0.03)
    parser.add_argument(
        "--exclude-ids",
        default="",
        help="Comma-separated row ids to keep unchanged (e.g. companion audit cases)",
    )
    parser.add_argument(
        "--protect-companion-audit",
        action="store_true",
        help="Skip swap for all 30 benchmark_companion_candidate CASES ids",
    )
    args = parser.parse_args()

    exclude: set[str] = set()
    if args.exclude_ids.strip():
        exclude.update(x.strip() for x in args.exclude_ids.split(",") if x.strip())
    if args.protect_companion_audit:
        from benchmark_companion_candidate import CASES

        exclude.update(str(case["id"]) for case in CASES)

    stats = create_rrf_zone_swap(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        Path(args.zone_cache),
        Path(args.mapping),
        replace_min_gap=args.replace_min_gap,
        article_min_score=args.article_min_score,
        cap_docs=args.cap_docs,
        exclude_ids=exclude,
        conditional_cap_docs_2=args.conditional_cap_docs_2,
        cross_doc_min_score=args.cross_doc_min_score,
        cross_doc_max_gap=args.cross_doc_max_gap,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
