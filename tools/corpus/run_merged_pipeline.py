#!/usr/bin/env python3
"""Run merged-corpus pipeline: index -> cache retrieval -> recall-boost submission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "submission_variants" / "checkpoints"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=REPO_ROOT, env=merged_env, check=True)


def main() -> None:
    os.chdir(REPO_ROOT)
    py = sys.executable
    merged_env = {"USE_MERGED_CORPUS": "1"}

    run([py, "tools/corpus/build_bm25_merged.py"], env=merged_env)

    run(
        [
            py,
            "tools/submission/cache_live_retrieval.py",
            "--output",
            "data/augmented/live_retrieval_test_v3_merged.json",
            "--mapping",
            "data/law_id_to_title_merged.json",
            "--bm25-only",
        ],
        env=merged_env,
    )

    output_zip = REPO_ROOT / "submission_variants" / "submission_recall_boost_merged_tight_v1.zip"
    debug_csv = REPO_ROOT / "submission_variants" / "submission_recall_boost_merged_tight_v1_debug.csv"
    run(
        [
            py,
            "tools/submission/create_recall_boost_submission.py",
            "--base",
            "submission.zip",
            "--output",
            str(output_zip),
            "--debug",
            str(debug_csv),
            "--cap-articles",
            "2",
            "--cap-docs",
            "1",
            "--article-same-law-only",
            "--article-min-score",
            "0.9",
            "--article-min-gap-from-top1",
            "0.03",
            "--live-cache",
            "data/augmented/live_retrieval_test_v3_merged.json",
            "--mapping",
            "data/law_id_to_title_merged.json",
        ]
    )

    checkpoint = CHECKPOINT_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_merged_tight_v1"
    checkpoint.mkdir(parents=True, exist_ok=True)
    meta = {
        "checkpoint_name": checkpoint.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission": str(output_zip),
        "params": {
            "cap_articles": 2,
            "cap_docs": 1,
            "article_same_law_only": True,
            "add_new_docs": False,
            "article_min_score": 0.9,
            "article_min_gap_from_top1": 0.03,
        },
        "corpus": "data/corpus/legal_corpus_merged.json",
        "live_cache": "data/augmented/live_retrieval_test_v3_merged.json",
    }
    (checkpoint / "checkpoint.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE submission={output_zip}")
    print(f"checkpoint={checkpoint / 'checkpoint.json'}")


if __name__ == "__main__":
    main()
