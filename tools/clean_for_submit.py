#!/usr/bin/env python3
"""Remove dev artifacts before hand-in. Keeps final submission zips + source code."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

KEEP_SUBMISSION_ZIPS = {
    "rrf_swap_g008.zip",  # best IR ~0.6308
    "qa_promote_g008_ollama.zip",  # best QA
    "submission_cache_only_top1.zip",  # pure retrieval reference ~0.5895
}

KEEP_AUGMENTED = {
    "live_retrieval_rrf_wide_merged.json",
    "merged_v2_report.json",
}

ROOT_REMOVE = [
    "results.json",
    "submission_benchmark_details.jsonl",
    ".DS_Store",
]

AUGMENTED_REMOVE = [
    "vlsp_bulk_gap_corpus.json",
    "vlsp_bulk_gap_report.json",
    "vlsp_bulk_gap_titles.json",
    "db_gap_corpus.json",
    "db_gap_report.json",
    "db_gap_titles.json",
    "db_seed_articles.json",
    "vlsp_gap_corpus.json",
    "augmented_bm25.pkl",
    "live_retrieval_test.json",
    "live_retrieval_test_v2.json",
    "live_retrieval_test_v3_merged.json",
    "live_retrieval_vn_rerank_tuned_merged.json",
    "live_rrf_hybrid.json",
]


def rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"removed {path.relative_to(REPO)}")


def main() -> None:
    for name in ROOT_REMOVE:
        rm(REPO / name)

    for p in REPO.rglob("__pycache__"):
        rm(p)

    logs = REPO / "logs"
    if logs.is_dir():
        for f in logs.iterdir():
            rm(f)

    aug = REPO / "data" / "augmented"
    if aug.is_dir():
        for name in AUGMENTED_REMOVE:
            rm(aug / name)
        for f in aug.glob("*.bak*"):
            rm(f)

    variants = REPO / "submission_variants"
    if variants.is_dir():
        rm(variants / "local_benchmark")
        rm(variants / "legacy_root_outputs")
        for f in variants.rglob("*debug.csv"):
            rm(f)
        for f in variants.glob("*.json"):
            rm(f)
        for f in variants.glob("*.zip"):
            if f.name not in KEEP_SUBMISSION_ZIPS:
                rm(f)
        for d in list(variants.iterdir()):
            if d.is_dir() and d.name != "checkpoints":
                shutil.rmtree(d)
                print(f"removed {d.relative_to(REPO)}/")

    final_dir = variants / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in KEEP_SUBMISSION_ZIPS:
        src = variants / name
        if src.exists():
            dst = final_dir / name
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"copied {name} -> submission_variants/final/")
            rm(src)

    for name in ("R2AI_MENTORDAY2.pdf", "first50_audit.tsv", "public50_current.tsv"):
        rm(variants / name)

    ir_best = variants / "rrf_swap_g008.zip"
    if ir_best.exists():
        shutil.copy2(ir_best, REPO / "submission.zip")
        print("updated submission.zip from rrf_swap_g008.zip")

    qa_best = variants / "qa_promote_g008_ollama.zip"
    if qa_best.exists():
        shutil.copy2(qa_best, REPO / "submission_qa.zip")
        print("wrote submission_qa.zip from qa_promote_g008_ollama.zip")

    print("done.")


if __name__ == "__main__":
    main()
