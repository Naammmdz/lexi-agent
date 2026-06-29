"""Create top-k R2AI submission variants from an existing results.json."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from _paths import REPO_ROOT
from utils.submission_formatter import canonical_law_id


BASE_DIR = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE_DIR / "results.json"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "submission_variants"))
    parser.add_argument("--topks", default="1,2,3")
    parser.add_argument(
        "--article-doc-pairs",
        default="",
        help="Comma-separated article:doc cutoffs, e.g. 1:2,1:3,2:3.",
    )
    return parser.parse_args()


def article_law_id(article_ref: str) -> str:
    return canonical_law_id(str(article_ref).split("|", 1)[0])


def doc_law_id(doc_ref: str) -> str:
    return canonical_law_id(str(doc_ref).split("|", 1)[0])


def build_variant(rows: List[Dict[str, Any]], topk: int) -> List[Dict[str, Any]]:
    variant = []
    for row in rows:
        new_row = dict(row)
        kept_articles = list(row.get("relevant_articles", []))[:topk]
        kept_law_ids = {article_law_id(ref) for ref in kept_articles}
        kept_docs = [
            ref
            for ref in row.get("relevant_docs", [])
            if doc_law_id(ref) in kept_law_ids
        ]

        new_row["relevant_articles"] = kept_articles
        new_row["relevant_docs"] = kept_docs
        variant.append(new_row)
    return variant


def build_pair_variant(rows: List[Dict[str, Any]], article_topk: int, doc_topk: int) -> List[Dict[str, Any]]:
    variant = []
    for row in rows:
        new_row = dict(row)
        new_row["relevant_articles"] = list(row.get("relevant_articles", []))[:article_topk]
        new_row["relevant_docs"] = list(row.get("relevant_docs", []))[:doc_topk]
        variant.append(new_row)
    return variant


def write_variant(rows: List[Dict[str, Any]], output_dir: Path, topk: int) -> None:
    json_path = output_dir / f"results_top{topk}.json"
    zip_path = output_dir / f"submission_hybrid_rerank_top{topk}.zip"

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    avg_docs = sum(len(row.get("relevant_docs", [])) for row in rows) / len(rows)
    avg_articles = sum(len(row.get("relevant_articles", [])) for row in rows) / len(rows)
    print(
        f"top{topk}: {zip_path} "
        f"(avg_docs={avg_docs:.2f}, avg_articles={avg_articles:.2f})"
    )


def write_pair_variant(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    article_topk: int,
    doc_topk: int,
) -> None:
    json_path = output_dir / f"results_articles_top{article_topk}_docs_top{doc_topk}.json"
    zip_path = output_dir / f"submission_articles_top{article_topk}_docs_top{doc_topk}.zip"

    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    avg_docs = sum(len(row.get("relevant_docs", [])) for row in rows) / len(rows)
    avg_articles = sum(len(row.get("relevant_articles", [])) for row in rows) / len(rows)
    print(
        f"articles_top{article_topk}_docs_top{doc_topk}: {zip_path} "
        f"(avg_docs={avg_docs:.2f}, avg_articles={avg_articles:.2f})"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    topks = [int(value.strip()) for value in args.topks.split(",") if value.strip()]

    for topk in topks:
        write_variant(build_variant(rows, topk), output_dir, topk)

    if args.article_doc_pairs:
        for pair in args.article_doc_pairs.split(","):
            article_topk, doc_topk = [int(value.strip()) for value in pair.split(":", 1)]
            write_pair_variant(
                build_pair_variant(rows, article_topk, doc_topk),
                output_dir,
                article_topk,
                doc_topk,
            )


if __name__ == "__main__":
    main()
