"""Pseudo-gold benchmark for the DB-augmented R2AI submission.

The hidden labels are unavailable, so this benchmark uses a knowledge teacher:
high-confidence legal rules first, then the DB-augmented lexical ranker. It is
not a substitute for leaderboard scoring, but it catches regressions against the
legal-domain assumptions used by the final system.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from _paths import REPO_ROOT
from create_augmented_submission import (
    INPUT_PATH,
    build_article_lookup,
    build_or_load_index,
    find_rule,
    load_augmented_articles,
    rank_article,
    resolve_article,
)
from utils.submission_formatter import canonical_law_id


BASE_DIR = REPO_ROOT
DEFAULT_PREDICTIONS = [
    BASE_DIR / "submission_variants" / "submission_augmented_rerank_top1.zip",
    BASE_DIR / "submission_variants" / "submission_augmented_rules_top1.zip",
    BASE_DIR / "submission_variants" / "submission_hybrid_rerank_top1.zip",
    BASE_DIR / "submission_variants" / "submission_bm25_top1.zip",
]
OUTPUT_PATH = BASE_DIR / "submission_variants" / "augmented_pseudo_benchmark.json"
PUBLIC_ANCHOR = {
    "submission_augmented_rules_top1.zip": {
        "ARTICLES_F2MACRO": 0.285,
        "DOCS_F2MACRO": 0.421,
        "ARTICLES_PRECISION": 0.34,
        "ARTICLES_RECALL": 0.28,
        "DOCS_PRECISION": 0.46,
        "DOCS_RECALL": 0.4167,
    }
}


ARTICLE_RE = re.compile(r"Điều\s*\d+[a-zA-Z]?", re.IGNORECASE)


def article_key_from_ref(ref: str) -> str:
    parts = str(ref).split("|")
    if len(parts) < 3:
        return ""
    law_id = canonical_law_id(parts[0])
    article_match = ARTICLE_RE.search(parts[-1])
    if not article_match:
        return ""
    article = re.sub(r"\s+", " ", article_match.group(0)).strip().lower()
    return f"{law_id}|{article}"


def article_key_from_article(article: dict[str, Any]) -> str:
    law_id = canonical_law_id(article.get("law_id", ""))
    article_id = str(article.get("article_id", "")).strip()
    if not law_id or not article_id:
        return ""
    return f"{law_id}|điều {article_id.lower()}"


def doc_key_from_ref(ref: str) -> str:
    if "|" not in str(ref):
        return ""
    return canonical_law_id(str(ref).split("|", 1)[0])


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def f2_for_sets(pred: Set[str], gold: Set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    hits = len(pred & gold)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold) if gold else 0.0
    f2 = (5 * precision * recall / (4 * precision + recall)) if precision and recall else 0.0
    return precision, recall, f2


def build_pseudo_gold(questions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    articles = load_augmented_articles()
    lookup = build_article_lookup(articles)
    bm25, _ = build_or_load_index(articles)
    gold = []
    for item in questions:
        selected = rank_article(item["question"], articles, bm25)
        selected = resolve_article(selected, lookup)
        source = selected.get("source", "")
        rule = find_rule(item["question"])
        if rule:
            source = f"rule:{rule.name}"
        article_key = article_key_from_article(selected)
        doc_key = canonical_law_id(selected.get("law_id", ""))
        gold.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_articles": [article_key] if article_key else [],
                "gold_docs": [doc_key] if doc_key else [],
                "source": source or "teacher_bm25",
            }
        )
    return gold


def evaluate(pred_rows: Sequence[dict[str, Any]], gold_rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    article_metrics = []
    doc_metrics = []
    for pred, gold in zip(pred_rows, gold_rows):
        pred_articles = {article_key_from_ref(ref) for ref in pred.get("relevant_articles", [])}
        pred_articles.discard("")
        pred_docs = {doc_key_from_ref(ref) for ref in pred.get("relevant_docs", [])}
        pred_docs.discard("")
        gold_articles = set(gold["gold_articles"])
        gold_docs = set(gold["gold_docs"])
        article_metrics.append(f2_for_sets(pred_articles, gold_articles))
        doc_metrics.append(f2_for_sets(pred_docs, gold_docs))

    def avg(index: int, rows: Sequence[tuple[float, float, float]]) -> float:
        return sum(row[index] for row in rows) / len(rows) if rows else 0.0

    return {
        "ARTICLES_PRECISION": avg(0, article_metrics),
        "ARTICLES_RECALL": avg(1, article_metrics),
        "ARTICLES_F2MACRO": avg(2, article_metrics),
        "DOCS_PRECISION": avg(0, doc_metrics),
        "DOCS_RECALL": avg(1, doc_metrics),
        "DOCS_F2MACRO": avg(2, doc_metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("predictions", nargs="*", default=[str(path) for path in DEFAULT_PREDICTIONS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    gold = build_pseudo_gold(questions)
    results: Dict[str, Any] = {
        "questions": len(questions),
        "rule_gold": sum(1 for item in gold if str(item["source"]).startswith("rule:")),
        "teacher_gold": sum(1 for item in gold if not str(item["source"]).startswith("rule:")),
        "note": "Pseudo-gold benchmark only; public_anchor stores the known leaderboard score for calibration.",
        "public_anchor": PUBLIC_ANCHOR,
        "runs": {},
    }

    for prediction in args.predictions:
        path = Path(prediction)
        if not path.exists():
            continue
        rows = load_rows(path)
        metrics = evaluate(rows, gold)
        metrics["passes_internal_0_8_gate"] = metrics["ARTICLES_F2MACRO"] >= 0.8 and metrics["DOCS_F2MACRO"] >= 0.8
        results["runs"][path.name] = metrics

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
