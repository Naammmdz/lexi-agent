#!/usr/bin/env python3
"""Generate and compare next-step variants on top of Vietnamese_Reranker cache."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from _paths import REPO_ROOT
from utils.submission_formatter import canonical_law_id

PY = sys.executable
OUTPUT_DIR = REPO_ROOT / "submission_variants"
BENCHMARK_DIR = OUTPUT_DIR / "local_benchmark"
CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_vn_rerank_merged.json"
MAPPING = REPO_ROOT / "data" / "law_id_to_title_merged.json"
BASE = REPO_ROOT / "submission.zip"
BEST = OUTPUT_DIR / "submission_recall_boost_merged_vn_rerank_tight_v1.zip"

COMMON = [
    "--base", str(BASE),
    "--live-cache", str(CACHE),
    "--mapping", str(MAPPING),
    "--cap-articles", "2",
    "--cap-docs", "1",
    "--article-same-law-only",
]

VARIANTS = [
    {
        "name": "s085_g003",
        "args": ["--article-min-score", "0.85", "--article-min-gap-from-top1", "0.03"],
    },
    {
        "name": "s085_g002",
        "args": ["--article-min-score", "0.85", "--article-min-gap-from-top1", "0.02"],
    },
    {
        "name": "s080_g003",
        "args": ["--article-min-score", "0.80", "--article-min-gap-from-top1", "0.03"],
    },
    {
        "name": "top1_g0",
        "args": [
            "--article-min-score", "0.9",
            "--article-min-gap-from-top1", "0.03",
            "--prefer-cache-top1",
            "--replace-top1-min-gap", "0.0",
        ],
    },
    {
        "name": "top1_g003",
        "args": [
            "--article-min-score", "0.9",
            "--article-min-gap-from-top1", "0.03",
            "--prefer-cache-top1",
            "--replace-top1-min-gap", "0.03",
        ],
    },
    {
        "name": "top1_g005",
        "args": [
            "--article-min-score", "0.9",
            "--article-min-gap-from-top1", "0.03",
            "--prefer-cache-top1",
            "--replace-top1-min-gap", "0.05",
        ],
    },
    {
        "name": "top1_s085",
        "args": [
            "--article-min-score", "0.85",
            "--article-min-gap-from-top1", "0.03",
            "--prefer-cache-top1",
            "--replace-top1-min-gap", "0.03",
        ],
    },
]


def article_key(ref: str) -> tuple[str, str]:
    parts = str(ref).split("|")
    law = canonical_law_id(parts[0]) if parts else ""
    label = parts[-1].strip().lower() if len(parts) >= 3 else ""
    return law, label


def load_zip(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as zf:
        rows = json.loads(zf.read("results.json"))
    return {int(r["id"]): r for r in rows}


def cache_top1(cache: dict, row_id: int) -> tuple[str, str] | None:
    cands = cache.get(str(row_id), [])
    if not cands:
        return None
    ref = str(cands[0].get("article_ref", "")).strip()
    if not ref:
        return None
    return article_key(ref)


def summarize(path: Path, cache: dict) -> dict:
    rows = load_zip(path)
    cache_hits = 0
    for row_id, row in rows.items():
        top = row.get("relevant_articles", [""])[0] if row.get("relevant_articles") else ""
        if not top:
            continue
        ct = cache_top1(cache, row_id)
        if ct and article_key(top) == ct:
            cache_hits += 1
    return {
        "path": str(path.name),
        "avg_articles": round(sum(len(r.get("relevant_articles", [])) for r in rows.values()) / len(rows), 3),
        "avg_docs": round(sum(len(r.get("relevant_docs", [])) for r in rows.values()) / len(rows), 3),
        "cache_top1_hits": cache_hits,
        "cache_top1_rate": round(cache_hits / len(rows), 4),
    }


def main() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    results: list[dict] = []

    for variant in VARIANTS:
        name = variant["name"]
        out = OUTPUT_DIR / f"submission_recall_boost_merged_vn_rerank_{name}.zip"
        debug = OUTPUT_DIR / f"submission_recall_boost_merged_vn_rerank_{name}_debug.csv"
        cmd = [
            PY,
            "tools/submission/create_recall_boost_submission.py",
            *COMMON,
            "--output", str(out),
            "--debug", str(debug),
            *variant["args"],
        ]
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        stats = json.loads(proc.stdout)
        summary = summarize(out, cache)
        summary.update(stats)
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    best_summary = summarize(BEST, cache)
    best_summary["path"] = BEST.name
    best_summary["public_articles_f2"] = 0.5322
    results.insert(0, best_summary)

    out_path = BENCHMARK_DIR / "vn_rerank_variant_compare.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)

    ranked = sorted(
        [r for r in results if r.get("top1_replaced") is not None or r.get("changed_rows") is not None],
        key=lambda r: (r.get("cache_top1_rate", 0), r.get("avg_articles", 0)),
        reverse=True,
    )
    print("\nRanked by cache_top1_rate / avg_articles:")
    for row in ranked[:8]:
        print(
            f"  {row['path']:55} cache_top1={row.get('cache_top1_rate', 0):.4f} "
            f"avg_art={row.get('avg_articles', 0):.3f} "
            f"top1_rep={row.get('top1_replaced', '-')} "
            f"changed={row.get('changed_rows', '-')}"
        )


if __name__ == "__main__":
    main()
