#!/usr/bin/env python3
"""Tag train_qna rows for subset benchmarks (multi-hop, cross-doc, easy)."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

from _paths import REPO_ROOT
from config import Config
from submission_benchmark import load_train_rows
from utils.submission_formatter import canonical_law_id

OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"
DEFAULT_TAGS = OUT_DIR / "train_subset_tags.json"
DEFAULT_MISSING = OUT_DIR / "missing_vb_audit.json"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def gold_law_ids(relevant_articles: list[dict]) -> set[str]:
    out: set[str] = set()
    for art in relevant_articles:
        law = canonical_law_id(art.get("law_id", ""))
        if law:
            out.add(law)
    return out


def classify_row(question: str, relevant_articles: list[dict]) -> dict[str, bool]:
    laws = gold_law_ids(relevant_articles)
    n_articles = len(relevant_articles)
    wc = word_count(question)
    n_qmark = question.count("?")
    n_semi = question.count(";")

    multi_hop = wc >= 50 or n_qmark >= 2 or n_semi >= 1
    cross_doc = len(laws) >= 2
    easy = len(laws) <= 1 and n_articles <= 1 and wc < 30

    return {
        "multi_hop": multi_hop,
        "cross_doc": cross_doc,
        "easy": easy,
    }


def corpus_law_ids(corpus_path: Path) -> set[str]:
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    return {canonical_law_id(item.get("law_id", "")) for item in data if item.get("law_id")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=0, help="0 = all rows")
    parser.add_argument("--tags-output", default=str(DEFAULT_TAGS))
    parser.add_argument("--missing-output", default=str(DEFAULT_MISSING))
    args = parser.parse_args()

    train_path = REPO_ROOT / "data" / "train" / "train_qna.csv"
    max_q = args.max_questions or 10_000
    rows = load_train_rows(train_path, max_q)

    tagged: list[dict] = []
    counts = Counter()
    gold_laws: set[str] = set()

    for idx, row in enumerate(rows):
        tags = classify_row(row["question"], row["relevant_articles"])
        laws = sorted(gold_law_ids(row["relevant_articles"]))
        gold_laws.update(laws)
        for key, val in tags.items():
            if val:
                counts[key] += 1
        tagged.append(
            {
                "index": idx,
                "question_id": row.get("question_id", ""),
                "word_count": word_count(row["question"]),
                "gold_laws": laws,
                "n_gold_articles": len(row["relevant_articles"]),
                **tags,
            }
        )

    corpus_ids = corpus_law_ids(Path(Config.CORPUS_PATH))
    missing = sorted(gold_laws - corpus_ids)

    summary = {
        "total": len(rows),
        "multi_hop": counts["multi_hop"],
        "cross_doc": counts["cross_doc"],
        "easy": counts["easy"],
        "multi_hop_pct": round(counts["multi_hop"] / len(rows), 4) if rows else 0.0,
        "cross_doc_pct": round(counts["cross_doc"] / len(rows), 4) if rows else 0.0,
        "easy_pct": round(counts["easy"] / len(rows), 4) if rows else 0.0,
        "corpus_path": Config.CORPUS_PATH,
        "corpus_laws": len(corpus_ids),
        "gold_laws_unique": len(gold_laws),
        "gold_laws_missing_from_corpus": len(missing),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tags_out = Path(args.tags_output)
    tags_out.write_text(
        json.dumps({"summary": summary, "rows": tagged}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missing_out = Path(args.missing_output)
    missing_out.write_text(
        json.dumps(
            {
                "missing_count": len(missing),
                "missing_law_ids": missing,
                "sample": missing[:50],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {tags_out}")
    print(f"Wrote {missing_out}")


if __name__ == "__main__":
    main()
