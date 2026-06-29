"""Refine submission anchors using live retrieval cache.

Two guarded operations on top of the current best submission:

1. **Within-law swap**: if the row already cites a law but likely the wrong
   article within that law, replace that article with the best-scoring live
   candidate from the same law (when confidence is high).

2. **Same-law companion** (optional): add a second article from an existing law
   when top-2 live scores are close (ambiguous retrieval).

This targets ARTICLES_F2 without expanding DOCS (precision-safe for DOCS_*).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import (
    append_refs,
    article_key,
    article_ref,
    doc_key,
    load_rows,
    update_answer,
)
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
LIVE_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_test.json"

DEFAULT_BASE = OUTPUT_DIR / "submission_recall_boost_tight_v1.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_anchor_refine.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_anchor_refine_debug.csv"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"

ARTICLE_RE = re.compile(r"điều\s*([0-9]+[a-z]?)", re.IGNORECASE)


def label_to_article_id(label: str) -> str:
    match = ARTICLE_RE.search(str(label))
    if match:
        return match.group(1).lower()
    return str(label).lower().replace("điều", "").strip()


def best_by_law(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        law = canonical_law_id(cand.get("law_id", ""))
        if law:
            grouped[law].append(cand)
    return {law: max(items, key=lambda x: float(x.get("score", 0.0) or 0.0)) for law, items in grouped.items()}


def row_article_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    keys = []
    for ref in row.get("relevant_articles", []):
        key = article_key(ref)
        if key:
            keys.append((canonical_law_id(key[0]), key[1].replace("điều", "").strip()))
    return keys


def replace_primary_article(
    row: dict[str, Any],
    mapping: dict[str, str],
    law: str,
    new_article_id: str,
) -> bool:
    """Replace the first article under ``law`` without dropping other refs."""
    law = canonical_law_id(law)
    articles = row.get("relevant_articles", [])
    if not articles:
        return False
    for idx, ref in enumerate(articles):
        key = article_key(ref)
        if not key or canonical_law_id(key[0]) != law:
            continue
        new_ref = article_ref(law, new_article_id, mapping)
        if new_ref == ref:
            return False
        articles[idx] = new_ref
        row["relevant_articles"] = articles
        return True
    return False


def refine_row(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    swap_min_score: float,
    swap_max_gap_from_global: float,
    swap_require_global_same_law: bool,
    companion_min_score: float,
    companion_max_gap: float,
    cap_articles: int,
) -> tuple[str, str]:
    """Return (mode, reason) describing the main change."""
    if not candidates:
        return "", ""

    global_top = candidates[0]
    global_law = canonical_law_id(global_top.get("law_id", ""))
    global_score = float(global_top.get("score", 0.0) or 0.0)
    by_law = best_by_law(candidates)
    current_keys = row_article_keys(row)
    current_laws = {doc_key(ref) for ref in row.get("relevant_docs", []) if doc_key(ref)}
    mode = ""

    # Phase 1: within-law swap on primary anchor (first article).
    if current_keys:
        primary_law, primary_art = current_keys[0]
        best = by_law.get(primary_law)
        if best:
            best_art = label_to_article_id(best.get("label", ""))
            best_score = float(best.get("score", 0.0) or 0.0)
            gap_global = global_score - best_score
            if (
                best_art
                and (primary_law, best_art) != (primary_law, primary_art)
                and best_score >= swap_min_score
                and gap_global <= swap_max_gap_from_global
                and (not swap_require_global_same_law or global_law == primary_law)
            ):
                if replace_primary_article(row, mapping, primary_law, best_art):
                    mode = "swap"
                    current_keys = row_article_keys(row)

    # Phase 2: same-law companion when ambiguous.
    existing = {article_key(ref) for ref in row.get("relevant_articles", []) if article_key(ref)}
    if len(existing) < cap_articles:
        for cand in candidates[1:]:
            if len({article_key(r) for r in row.get("relevant_articles", []) if article_key(r)}) >= cap_articles:
                break
            score = float(cand.get("score", 0.0) or 0.0)
            if score < companion_min_score:
                continue
            if (global_score - score) > companion_max_gap:
                continue
            law = canonical_law_id(cand.get("law_id", ""))
            if law not in current_laws:
                continue
            art = label_to_article_id(cand.get("label", ""))
            if not art:
                continue
            if (law, art) in {(k[0], k[1].replace("điều", "").strip()) for k in existing if k}:
                continue
            append_refs(row, mapping, [(law, art)])
            mode = mode or "companion"

    return mode, ""


def create_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    swap_min_score: float,
    swap_max_gap_from_global: float,
    swap_require_global_same_law: bool,
    companion_min_score: float,
    companion_max_gap: float,
    cap_articles: int,
    copy_to_submission: bool,
    mapping_path: Path | None = None,
    live_cache_path: Path | None = None,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path or MAPPING_PATH)
    rows = load_rows(base_zip)
    live = json.loads((live_cache_path or LIVE_CACHE).read_text(encoding="utf-8"))
    debug_rows: list[dict[str, Any]] = []
    modes: dict[str, int] = defaultdict(int)

    for row in rows:
        row_id = str(row["id"])
        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        mode, _ = refine_row(
            row,
            live.get(row_id, []),
            mapping,
            swap_min_score=swap_min_score,
            swap_max_gap_from_global=swap_max_gap_from_global,
            swap_require_global_same_law=swap_require_global_same_law,
            companion_min_score=companion_min_score,
            companion_max_gap=companion_max_gap,
            cap_articles=cap_articles,
        )
        update_answer(row)
        if row.get("relevant_articles") != before_articles or row.get("relevant_docs") != before_docs:
            if mode:
                modes[mode] += 1
            debug_rows.append(
                {
                    "id": row_id,
                    "mode": mode,
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
        fieldnames = ["id", "mode", "question", "before_articles", "after_articles", "before_docs", "after_docs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    return {
        "rows": len(rows),
        "changed_rows": len(debug_rows),
        "modes": dict(modes),
        "avg_articles": round(sum(len(r.get("relevant_articles", [])) for r in rows) / len(rows), 3),
        "avg_docs": round(sum(len(r.get("relevant_docs", [])) for r in rows) / len(rows), 3),
        "swap_min_score": swap_min_score,
        "swap_max_gap_from_global": swap_max_gap_from_global,
        "swap_require_global_same_law": swap_require_global_same_law,
        "companion_min_score": companion_min_score,
        "companion_max_gap": companion_max_gap,
        "cap_articles": cap_articles,
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--swap-min-score", type=float, default=0.92)
    parser.add_argument("--swap-max-gap-from-global", type=float, default=0.08)
    parser.add_argument("--swap-require-global-same-law", action="store_true", default=True)
    parser.add_argument("--no-swap-require-global-same-law", action="store_false", dest="swap_require_global_same_law")
    parser.add_argument("--companion-min-score", type=float, default=0.9)
    parser.add_argument("--companion-max-gap", type=float, default=0.03)
    parser.add_argument("--cap-articles", type=int, default=2)
    parser.add_argument("--copy-to-submission", action="store_true")
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--live-cache", default=str(LIVE_CACHE))
    args = parser.parse_args()

    stats = create_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        args.swap_min_score,
        args.swap_max_gap_from_global,
        args.swap_require_global_same_law,
        args.companion_min_score,
        args.companion_max_gap,
        args.cap_articles,
        args.copy_to_submission,
        mapping_path=Path(args.mapping),
        live_cache_path=Path(args.live_cache),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
