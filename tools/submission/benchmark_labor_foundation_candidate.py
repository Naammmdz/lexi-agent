"""Audit benchmark for labor/OSH penalty foundation companions."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from utils.submission_formatter import canonical_law_id


ARTICLE_RE = re.compile(r"điều\s+([0-9]+[a-z]?)", re.IGNORECASE)


CASES: list[dict[str, Any]] = [
    {"id": 61, "gold": [("12/2022/NĐ-CP", "13"), ("45/2019/QH14", "57")]},
    {"id": 63, "gold": [("12/2022/NĐ-CP", "10"), ("45/2019/QH14", "26")]},
    {"id": 64, "gold": [("12/2022/NĐ-CP", "21"), ("84/2015/QH13", "18")]},
    {"id": 65, "gold": [("12/2022/NĐ-CP", "22"), ("84/2015/QH13", "21")]},
    {"id": 66, "gold": [("12/2022/NĐ-CP", "19"), ("45/2019/QH14", "127")]},
    {"id": 67, "gold": [("12/2022/NĐ-CP", "28"), ("45/2019/QH14", "137")]},
    {"id": 68, "gold": [("12/2022/NĐ-CP", "24"), ("84/2015/QH13", "30")]},
    {"id": 69, "gold": [("12/2022/NĐ-CP", "27"), ("84/2015/QH13", "18")]},
    {"id": 70, "gold": [("12/2022/NĐ-CP", "25"), ("84/2015/QH13", "14")]},
    {"id": 72, "gold": [("12/2022/NĐ-CP", "20"), ("84/2015/QH13", "36")]},
    {"id": 73, "gold": [("12/2022/NĐ-CP", "29"), ("45/2019/QH14", "144")]},
    {"id": 74, "gold": [("12/2022/NĐ-CP", "34"), ("45/2019/QH14", "208")]},
    {"id": 76, "gold": [("12/2022/NĐ-CP", "9"), ("45/2019/QH14", "17")]},
    {"id": 77, "gold": [("12/2022/NĐ-CP", "23"), ("84/2015/QH13", "38")]},
    {"id": 79, "gold": [("12/2022/NĐ-CP", "12"), ("45/2019/QH14", "48")]},
    {"id": 80, "gold": [("12/2022/NĐ-CP", "31"), ("45/2019/QH14", "160")]},
]


def article_key(ref: str) -> tuple[str, str] | None:
    parts = str(ref).split("|")
    if len(parts) < 3:
        return None
    match = ARTICLE_RE.search(parts[-1])
    if not match:
        return None
    return canonical_law_id(parts[0]), match.group(1).lower()


def doc_key(ref: str) -> str | None:
    parts = str(ref).split("|", 1)
    if len(parts) < 2:
        return None
    return canonical_law_id(parts[0])


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read("results.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def prf2(pred: set[Any], gold: set[Any]) -> dict[str, float]:
    precision = len(pred & gold) / len(pred) if pred else 0.0
    recall = len(pred & gold) / len(gold) if gold else 0.0
    f2 = 0.0 if precision == 0.0 and recall == 0.0 else (5 * precision * recall) / (4 * precision + recall)
    return {"precision": precision, "recall": recall, "f2": f2}


def evaluate(path: Path) -> dict[str, Any]:
    rows = {int(row["id"]): row for row in load_rows(path)}
    article_scores = []
    doc_scores = []
    weak_cases = []
    for case in CASES:
        row = rows[case["id"]]
        gold_articles = {(canonical_law_id(law_id), article.lower()) for law_id, article in case["gold"]}
        gold_docs = {law_id for law_id, _article in gold_articles}
        pred_articles = {key for ref in row.get("relevant_articles", []) if (key := article_key(ref))}
        pred_docs = {key for ref in row.get("relevant_docs", []) if (key := doc_key(ref))}
        article_score = prf2(pred_articles, gold_articles)
        doc_score = prf2(pred_docs, gold_docs)
        article_scores.append(article_score)
        doc_scores.append(doc_score)
        if article_score["f2"] < 0.7:
            weak_cases.append(
                {
                    "id": case["id"],
                    "article_f2": round(article_score["f2"], 4),
                    "gold": sorted(f"{law}|{article}" for law, article in gold_articles),
                    "pred": sorted(f"{law}|{article}" for law, article in pred_articles),
                }
            )

    def avg(key: str, scores: list[dict[str, float]]) -> float:
        return sum(score[key] for score in scores) / len(scores)

    return {
        "path": str(path),
        "cases": len(CASES),
        "ARTICLES_F2MACRO": round(avg("f2", article_scores), 4),
        "ARTICLES_PRECISION": round(avg("precision", article_scores), 4),
        "ARTICLES_RECALL": round(avg("recall", article_scores), 4),
        "DOCS_F2MACRO": round(avg("f2", doc_scores), 4),
        "DOCS_PRECISION": round(avg("precision", doc_scores), 4),
        "DOCS_RECALL": round(avg("recall", doc_scores), 4),
        "weak_cases": weak_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output", default="submission_variants/local_benchmark/labor_foundation_benchmark.json")
    args = parser.parse_args()

    results = [evaluate(REPO_ROOT / path) for path in args.paths]
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
