"""Audit benchmark for union, BHXH, and collective-bargaining foundations."""

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
    {"id": 75, "gold": [("12/2022/NĐ-CP", "38"), ("12/2012/QH13", "26"), ("191/2013/NĐ-CP", "5")]},
    {"id": 83, "gold": [("12/2022/NĐ-CP", "35"), ("12/2012/QH13", "9"), ("12/2012/QH13", "16")]},
    {"id": 87, "gold": [("12/2022/NĐ-CP", "40"), ("41/2024/QH15", "9")]},
    {"id": 627, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70")]},
    {"id": 785, "gold": [("12/2022/NĐ-CP", "35"), ("12/2012/QH13", "9"), ("45/2019/QH14", "175")]},
    {"id": 801, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "86")]},
    {"id": 890, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70"), ("45/2019/QH14", "80")]},
    {"id": 955, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70"), ("45/2019/QH14", "84")]},
    {"id": 1028, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70"), ("45/2019/QH14", "85")]},
    {"id": 1413, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70")]},
    {"id": 1789, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70"), ("45/2019/QH14", "80")]},
    {"id": 1799, "gold": [("12/2022/NĐ-CP", "39"), ("12/2022/NĐ-CP", "40"), ("41/2024/QH15", "40"), ("41/2024/QH15", "9")]},
    {"id": 1816, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70")]},
    {"id": 1819, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70"), ("45/2019/QH14", "84")]},
    {"id": 1836, "gold": [("12/2022/NĐ-CP", "39"), ("38/2013/QH13", "43"), ("41/2024/QH15", "40")]},
    {"id": 1895, "gold": [("12/2022/NĐ-CP", "40"), ("41/2024/QH15", "9")]},
    {"id": 1918, "gold": [("12/2022/NĐ-CP", "16"), ("45/2019/QH14", "70")]},
    {"id": 1999, "gold": [("12/2022/NĐ-CP", "38"), ("125/2020/NĐ-CP", "16"), ("12/2012/QH13", "26"), ("191/2013/NĐ-CP", "5"), ("38/2019/QH14", "142")]},
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
        if article_score["f2"] < 0.75:
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
    parser.add_argument("--output", default="submission_variants/local_benchmark/union_bhxh_foundation_benchmark.json")
    args = parser.parse_args()

    results = [evaluate(REPO_ROOT / path) for path in args.paths]
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
