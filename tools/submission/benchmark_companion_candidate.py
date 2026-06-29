"""Benchmark high-risk companion-document repairs against prior submissions.

This benchmark is intentionally small and audit-focused.  It covers rows where
the current plateau submission looked legally incomplete or clearly off-domain:
tax procedure companions, construction permits/acceptance, arbitration, and
spam-call rules.  It uses the same macro F2 shape as the public scorer for the
retrieval columns, but the gold labels are manual legal-audit labels.
"""

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
    {"id": 90, "gold": [("123/2020/NĐ-CP", "13"), ("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34")]},
    {"id": 398, "gold": [("38/2019/QH14", "131"), ("126/2020/NĐ-CP", "35")]},
    {"id": 618, "gold": [("126/2020/NĐ-CP", "23"), ("38/2019/QH14", "85"), ("126/2020/NĐ-CP", "24")]},
    {"id": 880, "gold": [("54/2010/QH12", "16"), ("54/2010/QH12", "33"), ("36/2005/QH11", "319")]},
    {"id": 1055, "gold": [("38/2019/QH14", "62"), ("38/2019/QH14", "65"), ("126/2020/NĐ-CP", "19")]},
    {"id": 1090, "gold": [("38/2019/QH14", "85"), ("126/2020/NĐ-CP", "24"), ("126/2020/NĐ-CP", "38")]},
    {"id": 1126, "gold": [("38/2019/QH14", "85"), ("126/2020/NĐ-CP", "24"), ("126/2020/NĐ-CP", "38")]},
    {"id": 1160, "gold": [("38/2019/QH14", "85"), ("126/2020/NĐ-CP", "24"), ("126/2020/NĐ-CP", "38")]},
    {"id": 1243, "gold": [("06/2021/NĐ-CP", "22"), ("06/2021/NĐ-CP", "23")]},
    {"id": 1253, "gold": [("54/2010/QH12", "54"), ("54/2010/QH12", "57")]},
    {"id": 1360, "gold": [("50/2014/QH13", "97")]},
    {"id": 1363, "gold": [("06/2021/NĐ-CP", "21"), ("06/2021/NĐ-CP", "22"), ("06/2021/NĐ-CP", "23")]},
    {"id": 1436, "gold": [("06/2021/NĐ-CP", "23")]},
    {"id": 1478, "gold": [("91/2020/NĐ-CP", "13")]},
    {"id": 1497, "gold": [("50/2014/QH13", "99")]},
    {"id": 1532, "gold": [("50/2014/QH13", "91")]},
    {"id": 1578, "gold": [("50/2014/QH13", "90")]},
    {"id": 1629, "gold": [("50/2014/QH13", "89")]},
    {"id": 1638, "gold": [("54/2010/QH12", "16"), ("54/2010/QH12", "33")]},
    {"id": 1647, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34"), ("125/2020/NĐ-CP", "4")]},
    {"id": 1649, "gold": [("38/2019/QH14", "62"), ("38/2019/QH14", "65"), ("126/2020/NĐ-CP", "19")]},
    {"id": 1683, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34"), ("125/2020/NĐ-CP", "44")]},
    {"id": 1686, "gold": [("38/2019/QH14", "65"), ("126/2020/NĐ-CP", "19")]},
    {"id": 1689, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34")]},
    {"id": 1703, "gold": [("38/2019/QH14", "62"), ("38/2019/QH14", "65"), ("126/2020/NĐ-CP", "19")]},
    {"id": 1719, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34")]},
    {"id": 1728, "gold": [("38/2019/QH14", "130"), ("126/2020/NĐ-CP", "32")]},
    {"id": 1889, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34")]},
    {"id": 1936, "gold": [("38/2019/QH14", "125"), ("126/2020/NĐ-CP", "34")]},
    {"id": 1985, "gold": [("38/2019/QH14", "62"), ("126/2020/NĐ-CP", "19")]},
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
    if not pred:
        precision = 0.0
    else:
        precision = len(pred & gold) / len(pred)
    recall = len(pred & gold) / len(gold) if gold else 0.0
    f2 = 0.0 if precision == 0.0 and recall == 0.0 else (5 * precision * recall) / (4 * precision + recall)
    return {"precision": precision, "recall": recall, "f2": f2}


def evaluate(path: Path) -> dict[str, Any]:
    rows = {int(row["id"]): row for row in load_rows(path)}
    article_scores = []
    doc_scores = []
    details = []
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
        details.append(
            {
                "id": case["id"],
                "article_f2": round(article_score["f2"], 4),
                "doc_f2": round(doc_score["f2"], 4),
                "gold": sorted([f"{law}|{article}" for law, article in gold_articles]),
                "pred": sorted([f"{law}|{article}" for law, article in pred_articles]),
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
        "weak_cases": [detail for detail in details if detail["article_f2"] < 0.7],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[
        "submission_variants/submission_domain_repair_v12_final_candidate.zip",
        "submission_variants/submission_domain_repair_v26_public_sensitive_refine.zip",
        "submission_variants/submission_domain_repair_v30_companion_benchmark.zip",
    ])
    parser.add_argument("--output", default="submission_variants/local_benchmark/companion_benchmark_v30.json")
    args = parser.parse_args()

    results = [evaluate(REPO_ROOT / path) for path in args.paths]
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
