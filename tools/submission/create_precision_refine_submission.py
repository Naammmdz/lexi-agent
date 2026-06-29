"""Precision refine: same-law reorder + trim noisy same-law companions."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import load_rows, update_answer
from create_recall_boost_submission import article_ref_key, doc_ref
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
DEFAULT_BASE = OUTPUT_DIR / "submission_recall_boost_merged_vn_rerank_tight_v1.zip"
DEFAULT_CACHE = REPO_ROOT / "data/augmented/live_retrieval_vn_rerank_tuned_merged.json"
MAPPING = REPO_ROOT / "data" / "law_id_to_title_merged.json"


def cache_score_for_key(candidates: list[dict[str, Any]], key: tuple[str, str]) -> float | None:
    for cand in candidates:
        ref = str(cand.get("article_ref", "")).strip()
        if ref and article_ref_key(ref) == key:
            return float(cand.get("score", 0.0) or 0.0)
    return None


def best_same_law_candidate(
    candidates: list[dict[str, Any]],
    law_id: str,
) -> tuple[str, tuple[str, str], float] | None:
    best: tuple[str, tuple[str, str], float] | None = None
    for cand in candidates:
        ref = str(cand.get("article_ref", "")).strip()
        if not ref:
            continue
        key = article_ref_key(ref)
        if canonical_law_id(key[0]) != canonical_law_id(law_id):
            continue
        score = float(cand.get("score", 0.0) or 0.0)
        if best is None or score > best[2]:
            best = (ref, key, score)
    return best


def maybe_same_law_reorder(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    *,
    min_score: float,
    not_in_cache_only: bool = True,
) -> bool:
    """Promote best same-law cache article when anchor is missing or very weak in cache."""
    articles = list(row.get("relevant_articles", []))
    if not articles or not candidates:
        return False

    current_ref = articles[0]
    current_key = article_ref_key(current_ref)
    law = current_key[0]
    if not law:
        return False

    best = best_same_law_candidate(candidates, law)
    if not best:
        return False
    new_ref, new_key, new_score = best
    if new_key == current_key or new_score < min_score:
        return False

    current_score = cache_score_for_key(candidates, current_key)
    if not_in_cache_only:
        if current_score is not None:
            return False
    elif current_score is not None and current_score >= new_score * 0.85:
        return False

    remaining = [ref for ref in articles if article_ref_key(ref) != new_key]
    row["relevant_articles"] = [new_ref] + remaining
    new_doc = doc_ref(law, mapping)
    existing_docs = list(row.get("relevant_docs", []))
    existing_laws = {canonical_law_id(d.split("|")[0]) for d in existing_docs if "|" in d}
    if canonical_law_id(law) not in existing_laws:
        row["relevant_docs"] = [new_doc] + existing_docs
    return True


def trim_same_law_noise_companion(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> bool:
    """Drop companion when it is cache top-1 but submission anchor is a different cached article."""
    articles = list(row.get("relevant_articles", []))
    if len(articles) < 2 or not candidates:
        return False

    top_cand = candidates[0]
    cache_top_ref = str(top_cand.get("article_ref", "")).strip()
    if not cache_top_ref:
        return False
    cache_top_key = article_ref_key(cache_top_ref)

    anchor_key = article_ref_key(articles[0])
    companion_key = article_ref_key(articles[1])
    if canonical_law_id(companion_key[0]) != canonical_law_id(anchor_key[0]):
        return False
    if companion_key != cache_top_key or anchor_key == cache_top_key:
        return False
    if cache_score_for_key(candidates, anchor_key) is None:
        return False

    row["relevant_articles"] = articles[:1]
    return True


def create_precision_refine_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    live_cache: Path,
    mapping_path: Path,
    *,
    enable_reorder: bool = True,
    enable_trim: bool = True,
    reorder_min_score: float = 0.9,
    reorder_not_in_cache_only: bool = True,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path)
    rows = load_rows(base_zip)
    cache = json.loads(live_cache.read_text(encoding="utf-8"))
    debug_rows: list[dict[str, Any]] = []
    reordered = 0
    trimmed = 0

    for row in rows:
        rid = str(row["id"])
        candidates = cache.get(rid, [])
        before_articles = list(row.get("relevant_articles", []))

        changed = False
        reason = []

        if enable_reorder and maybe_same_law_reorder(
            row,
            candidates,
            mapping,
            min_score=reorder_min_score,
            not_in_cache_only=reorder_not_in_cache_only,
        ):
            reordered += 1
            changed = True
            reason.append("same_law_reorder")

        if enable_trim and trim_same_law_noise_companion(row, candidates):
            trimmed += 1
            changed = True
            reason.append("trim_noise_companion")

        if changed:
            update_answer(row)
            debug_rows.append(
                {
                    "id": rid,
                    "reason": "|".join(reason),
                    "question": row.get("question", ""),
                    "before_articles": " || ".join(before_articles),
                    "after_articles": " || ".join(row.get("relevant_articles", [])),
                }
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    with debug_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "reason", "question", "before_articles", "after_articles"],
        )
        writer.writeheader()
        writer.writerows(debug_rows)

    avg_articles = round(sum(len(r.get("relevant_articles", [])) for r in rows) / len(rows), 3)
    return {
        "rows": len(rows),
        "reordered": reordered,
        "trimmed": trimmed,
        "changed_rows": len(debug_rows),
        "avg_articles": avg_articles,
        "output": str(output_zip),
        "debug": str(debug_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", required=True)
    parser.add_argument("--live-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--no-reorder", action="store_true")
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--reorder-min-score", type=float, default=0.9)
    parser.add_argument("--reorder-allow-weak-in-cache", action="store_true")
    args = parser.parse_args()

    stats = create_precision_refine_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        Path(args.live_cache),
        Path(args.mapping),
        enable_reorder=not args.no_reorder,
        enable_trim=not args.no_trim,
        reorder_min_score=args.reorder_min_score,
        reorder_not_in_cache_only=not args.reorder_allow_weak_in_cache,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
