"""Find close train-set neighbors whose gold refs differ from a submission."""

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
from create_domain_repair_submission import article_key
from utils.submission_formatter import canonical_law_id


STOPWORDS = {
    "của",
    "và",
    "về",
    "cho",
    "các",
    "những",
    "được",
    "không",
    "như",
    "nào",
    "trong",
    "theo",
    "khi",
    "nếu",
    "phải",
    "cần",
    "quy",
    "định",
    "doanh",
    "nghiệp",
    "công",
    "ty",
    "trường",
    "hợp",
}


def norm(text: str) -> str:
    text = str(text or "").lower().replace("ð", "đ")
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[\w/.-]+", norm(text), flags=re.UNICODE) if len(tok) >= 3 and tok not in STOPWORDS}


def f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(b)
    return 2 * precision * recall / (precision + recall)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def parse_train_refs(raw: str) -> set[tuple[str, str]]:
    refs = set()
    for item in ast.literal_eval(raw):
        law_id = canonical_law_id(item.get("law_id", ""))
        article_id = str(item.get("article_id", "")).lower()
        if law_id and article_id:
            refs.add((law_id, article_id))
    return refs


def current_refs(row: dict[str, Any]) -> set[tuple[str, str]]:
    return {key for ref in row.get("relevant_articles", []) if (key := article_key(ref))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=str(REPO_ROOT / "submission.zip"))
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    train = []
    with (REPO_ROOT / "data" / "train" / "train_qna.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            refs = parse_train_refs(r["relevant_articles"])
            train.append((r["question"], tokens(r["question"]), refs))

    printed = 0
    for row in load_rows(Path(args.submission)):
        qtokens = tokens(row["question"])
        best = sorted(((f1(qtokens, ttoks), tq, refs) for tq, ttoks, refs in train), reverse=True, key=lambda x: x[0])[:3]
        cur = current_refs(row)
        for score, tq, refs in best[:1]:
            if score < args.threshold or refs <= cur:
                continue
            printed += 1
            print(f"\nID {row['id']} sim={score:.3f}")
            print("Q:", row["question"])
            print("TRAIN:", tq)
            print("current:", sorted(cur))
            print("train_refs:", sorted(refs))
            if printed >= args.limit:
                return


if __name__ == "__main__":
    main()
