"""Leaderboard-style retrieval benchmark for R2AI submissions.

This evaluates predicted relevant_docs/relevant_articles instead of MRR/MAP.
It uses train_qna.csv labels and reports macro Precision/Recall/F2 for docs
and articles across retrieval methods and top-k cutoffs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import time
import io
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from _paths import REPO_ROOT
from config import Config
from main.chatbot import VietnameseLegalRAG
from utils.data_loader import LegalDataLoader
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    dedupe_keep_order,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)


BASE_DIR = REPO_ROOT
DEFAULT_TRAIN_FILE = BASE_DIR / "data" / "train" / "train_qna.csv"
DEFAULT_MAPPING_FILE = BASE_DIR / "data" / "law_id_to_title.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--methods", default="bm25,hybrid,hybrid_rerank")
    parser.add_argument("--cutoffs", default="1,2,3,5,8,20")
    parser.add_argument("--output", default=str(BASE_DIR / "submission_benchmark_results.csv"))
    parser.add_argument("--details-output", default=str(BASE_DIR / "submission_benchmark_details.jsonl"))
    parser.add_argument("--setup-indices", action="store_true")
    parser.add_argument("--verbose-retrieval", action="store_true")
    return parser.parse_args()


def load_train_rows(path: Path, max_questions: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            relevant_articles = ast.literal_eval(row["relevant_articles"])
            rows.append(
                {
                    "question_id": row.get("question_id", ""),
                    "question": row["question"],
                    "relevant_articles": relevant_articles,
                }
            )
            if max_questions and len(rows) >= max_questions:
                break
    return rows


def gold_sets(relevant_articles: Sequence[Dict[str, Any]]) -> Tuple[set[str], set[Tuple[str, str]]]:
    gold_docs = set()
    gold_articles = set()
    for article in relevant_articles:
        law_id = canonical_law_id(article.get("law_id", ""))
        article_id = str(article.get("article_id", "")).strip()
        if not law_id:
            continue
        gold_docs.add(law_id)
        if article_id:
            gold_articles.add((law_id, f"Điều {article_id}"))
    return gold_docs, gold_articles


def f2(precision: float, recall: float) -> float:
    denom = 4 * precision + recall
    if denom == 0:
        return 0.0
    return 5 * precision * recall / denom


def prf(pred: set[Any], gold: set[Any]) -> Tuple[float, float, float]:
    if not pred:
        precision = 0.0
    else:
        precision = len(pred & gold) / len(pred)

    if not gold:
        recall = 0.0
    else:
        recall = len(pred & gold) / len(gold)

    return precision, recall, f2(precision, recall)


def macro_average(values: Iterable[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0, 0.0
    return (
        sum(v[0] for v in values) / len(values),
        sum(v[1] for v in values) / len(values),
        sum(v[2] for v in values) / len(values),
    )


def refs_from_docs(
    docs: Sequence[Dict[str, Any]],
    mapping: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    doc_refs = []
    article_refs = []

    for doc in docs:
        metadata = doc.get("metadata", {})
        law_id = canonical_law_id(metadata.get("law_id", ""))
        if not law_id:
            continue

        title = format_law_title(law_id, get_mapping_title(mapping, law_id))
        doc_refs.append(f"{law_id}|{title}")

        label = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
        if label:
            article_refs.append(f"{law_id}|{title}|{label}")

    return dedupe_keep_order(doc_refs), dedupe_keep_order(article_refs)


def pred_sets(doc_refs: Sequence[str], article_refs: Sequence[str]) -> Tuple[set[str], set[Tuple[str, str]]]:
    pred_docs = set()
    pred_articles = set()

    for ref in doc_refs:
        law_id = ref.split("|", 1)[0].strip()
        if law_id:
            pred_docs.add(canonical_law_id(law_id))

    for ref in article_refs:
        parts = ref.split("|")
        if len(parts) != 3:
            continue
        law_id, _title, article = parts
        label = article_label(article)
        if law_id.strip() and label:
            pred_articles.add((canonical_law_id(law_id), label))

    return pred_docs, pred_articles


@contextmanager
def retrieval_mode(rag: VietnameseLegalRAG, method: str):
    original_vector_store = rag.vector_store
    original_bm25 = rag.bm25_retriever
    try:
        if method == "bm25":
            rag.vector_store = None
            yield False, False
        elif method == "vector":
            rag.bm25_retriever = None
            yield False, False
        elif method == "hybrid":
            yield True, False
        elif method == "hybrid_rerank":
            yield True, True
        else:
            raise ValueError(f"Unknown method: {method}")
    finally:
        rag.vector_store = original_vector_store
        rag.bm25_retriever = original_bm25


def retrieve(
    rag: VietnameseLegalRAG,
    question: str,
    method: str,
    verbose_retrieval: bool,
) -> List[Dict[str, Any]]:
    with retrieval_mode(rag, method) as (use_hybrid, use_reranking):
        if verbose_retrieval:
            return rag.retrieve_documents(
                question,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
            )

        with redirect_stdout(io.StringIO()):
            return rag.retrieve_documents(
                question,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
            )


def evaluate_method(
    rag: VietnameseLegalRAG,
    rows: Sequence[Dict[str, Any]],
    method: str,
    cutoffs: Sequence[int],
    mapping: Dict[str, str],
    details_writer: Any,
    verbose_retrieval: bool,
) -> List[Dict[str, Any]]:
    per_cutoff = {
        cutoff: {
            "doc_scores": [],
            "article_scores": [],
            "pred_docs": [],
            "pred_articles": [],
            "latencies": [],
        }
        for cutoff in cutoffs
    }

    for idx, row in enumerate(rows, 1):
        start = time.time()
        docs = retrieve(rag, row["question"], method, verbose_retrieval)
        latency = time.time() - start
        gold_docs, gold_articles = gold_sets(row["relevant_articles"])

        for cutoff in cutoffs:
            kept_docs = docs[:cutoff]
            doc_refs, article_refs = refs_from_docs(kept_docs, mapping)
            predicted_docs, predicted_articles = pred_sets(doc_refs, article_refs)

            bucket = per_cutoff[cutoff]
            bucket["doc_scores"].append(prf(predicted_docs, gold_docs))
            bucket["article_scores"].append(prf(predicted_articles, gold_articles))
            bucket["pred_docs"].append(len(predicted_docs))
            bucket["pred_articles"].append(len(predicted_articles))
            bucket["latencies"].append(latency)

            details_writer.write(
                json.dumps(
                    {
                        "method": method,
                        "cutoff": cutoff,
                        "question_index": idx,
                        "question_id": row["question_id"],
                        "question": row["question"],
                        "gold_docs": sorted(gold_docs),
                        "gold_articles": sorted([f"{a[0]}|{a[1]}" for a in gold_articles]),
                        "pred_docs": sorted(predicted_docs),
                        "pred_articles": sorted([f"{a[0]}|{a[1]}" for a in predicted_articles]),
                        "retrieved_ids": [doc.get("id", "") for doc in kept_docs],
                        "doc_refs": doc_refs,
                        "article_refs": article_refs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if idx % 10 == 0 or idx == len(rows):
            print(f"  {method}: {idx}/{len(rows)} questions")

    summaries = []
    for cutoff in cutoffs:
        bucket = per_cutoff[cutoff]
        docs_precision, docs_recall, docs_f2 = macro_average(bucket["doc_scores"])
        article_precision, article_recall, article_f2 = macro_average(bucket["article_scores"])
        summaries.append(
            {
                "method": method,
                "cutoff": cutoff,
                "questions": len(rows),
                "DOCS_PRECISION": docs_precision,
                "DOCS_RECALL": docs_recall,
                "DOCS_F2MACRO": docs_f2,
                "ARTICLES_PRECISION": article_precision,
                "ARTICLES_RECALL": article_recall,
                "ARTICLES_F2MACRO": article_f2,
                "AVG_PRED_DOCS": sum(bucket["pred_docs"]) / len(bucket["pred_docs"]),
                "AVG_PRED_ARTICLES": sum(bucket["pred_articles"]) / len(bucket["pred_articles"]),
                "AVG_LATENCY_SEC": sum(bucket["latencies"]) / len(bucket["latencies"]),
            }
        )
    return summaries


def write_summary(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "cutoff",
        "questions",
        "DOCS_PRECISION",
        "DOCS_RECALL",
        "DOCS_F2MACRO",
        "ARTICLES_PRECISION",
        "ARTICLES_RECALL",
        "ARTICLES_F2MACRO",
        "AVG_PRED_DOCS",
        "AVG_PRED_ARTICLES",
        "AVG_LATENCY_SEC",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def print_summary(summaries: Sequence[Dict[str, Any]]) -> None:
    ranked = sorted(summaries, key=lambda row: row["ARTICLES_F2MACRO"], reverse=True)
    print("\nTop configs by ARTICLES_F2MACRO")
    print("method          k   art_f2  art_p   art_r   doc_f2  avg_art")
    for row in ranked[:20]:
        print(
            f"{row['method']:<14} {row['cutoff']:<3} "
            f"{row['ARTICLES_F2MACRO']:.4f}  {row['ARTICLES_PRECISION']:.4f}  "
            f"{row['ARTICLES_RECALL']:.4f}  {row['DOCS_F2MACRO']:.4f}  "
            f"{row['AVG_PRED_ARTICLES']:.2f}"
        )


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cutoffs = [int(k.strip()) for k in args.cutoffs.split(",") if k.strip()]
    rows = load_train_rows(Path(args.train_file), args.max_questions)
    mapping = load_law_title_mapping(DEFAULT_MAPPING_FILE)

    print(f"Loaded {len(rows)} questions from {args.train_file}")
    print(f"Methods: {methods}")
    print(f"Cutoffs: {cutoffs}")

    rag = VietnameseLegalRAG()
    if args.setup_indices:
        loader = LegalDataLoader()
        documents = loader.prepare_documents_for_indexing()
        rag.setup_indices(documents, force_rebuild=False)
    elif rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    all_summaries = []
    details_path = Path(args.details_output)
    with details_path.open("w", encoding="utf-8") as details_writer:
        for method in methods:
            print(f"\nEvaluating {method}")
            summaries = evaluate_method(
                rag,
                rows,
                method,
                cutoffs,
                mapping,
                details_writer,
                args.verbose_retrieval,
            )
            all_summaries.extend(summaries)

    output_path = Path(args.output)
    write_summary(output_path, all_summaries)
    print_summary(all_summaries)
    print(f"\nWrote summary: {output_path}")
    print(f"Wrote details: {details_path}")


if __name__ == "__main__":
    main()
