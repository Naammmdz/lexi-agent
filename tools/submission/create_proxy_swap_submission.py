"""Build proxy-gated top1 swap candidate and benchmark vs base."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from audit_mismatch_top1 import (
    cache_top1,
    load_submission,
    load_train,
    neighbor_proxy,
    pick_winner,
    score_gap,
    top1_key,
    tokens,
)
from create_domain_repair_submission import load_rows, update_answer
from create_recall_boost_submission import article_ref_key, maybe_prefer_cache_top1
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


DEFAULT_BASE = REPO_ROOT / "submission_variants/submission_recall_boost_merged_vn_rerank_tight_v1.zip"
DEFAULT_CACHE = REPO_ROOT / "data/augmented/live_retrieval_vn_rerank_tuned_merged.json"
OUT_DIR = REPO_ROOT / "submission_variants"


def should_swap(
    row: dict[str, Any],
    cands: list[dict[str, Any]],
    train,
    *,
    same_law_only: bool,
    min_ratio: float,
    min_sim: float,
    proxy_winner: str,
) -> bool:
    sk = top1_key(row)
    ck = cache_top1(cands)
    if sk == ck or not cands:
        return False
    sl = sk and ck and canonical_law_id(sk[0]) == canonical_law_id(ck[0])
    if same_law_only and not sl:
        return False
    _top, _gap, ratio = score_gap(cands, sk)
    if ratio < min_ratio:
        return False
    proxy = neighbor_proxy(row.get("question", ""), train, min_sim)
    if not proxy:
        return False
    winner = pick_winner(sk, ck, proxy[1])
    return winner == proxy_winner


def build_candidate(
    base_zip: Path,
    output_zip: Path,
    live_cache: Path,
    mapping_path: Path,
    *,
    same_law_only: bool,
    min_ratio: float,
    min_sim: float,
    proxy_winner: str = "cache",
    article_min_score: float = 0.9,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path)
    rows = load_rows(base_zip)
    cache = json.loads(live_cache.read_text(encoding="utf-8"))
    train = load_train()
    swapped = 0

    for row in rows:
        rid = str(row["id"])
        cands = cache.get(rid, [])
        if not should_swap(
            row,
            cands,
            train,
            same_law_only=same_law_only,
            min_ratio=min_ratio,
            min_sim=min_sim,
            proxy_winner=proxy_winner,
        ):
            continue
        gap = 0.15 if min_ratio <= 0 else min_ratio
        if maybe_prefer_cache_top1(
            row,
            cands,
            mapping,
            article_min_score=article_min_score,
            replace_min_gap=gap,
            cap_docs=1,
        ):
            update_answer(row)
            swapped += 1

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")
    return {"output": str(output_zip), "top1_swapped": swapped, "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="proxy_same_law_r025")
    parser.add_argument("--same-law-only", action="store_true", default=True)
    parser.add_argument("--min-ratio", type=float, default=0.25)
    parser.add_argument("--min-sim", type=float, default=0.72)
    args = parser.parse_args()

    out = OUT_DIR / f"submission_recall_boost_merged_vn_rerank_{args.name}.zip"
    stats = build_candidate(
        DEFAULT_BASE,
        out,
        DEFAULT_CACHE,
        REPO_ROOT / "data/law_id_to_title_merged.json",
        same_law_only=args.same_law_only,
        min_ratio=args.min_ratio,
        min_sim=args.min_sim,
    )
    stats["name"] = args.name
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
