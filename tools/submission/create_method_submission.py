"""Create R2AI submission variants by running a specific retrieval method."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import zipfile
import io
from contextlib import contextmanager
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from _paths import REPO_ROOT
from main.chatbot import VietnameseLegalRAG
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    dedupe_keep_order,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)


BASE_DIR = REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["bm25", "hybrid", "hybrid_rerank"])
    parser.add_argument("--topks", default="1,2")
    parser.add_argument("--input", default=str(BASE_DIR / "R2AIStage1DATA.json"))
    parser.add_argument("--output-dir", default=str(BASE_DIR / "submission_variants"))
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--verbose-retrieval", action="store_true")
    return parser.parse_args()


@contextmanager
def retrieval_mode(rag: VietnameseLegalRAG, method: str):
    original_vector_store = rag.vector_store
    original_bm25 = rag.bm25_retriever
    try:
        if method == "bm25":
            rag.vector_store = None
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


def refs_from_docs(docs: List[Dict[str, Any]], mapping: Dict[str, str]) -> tuple[List[str], List[str]]:
    doc_refs = []
    article_refs = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        law_id = canonical_law_id(metadata.get("law_id", ""))
        if not law_id:
            continue

        law_title = format_law_title(law_id, get_mapping_title(mapping, law_id))
        doc_refs.append(f"{law_id}|{law_title}")

        label = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
        if label:
            article_refs.append(f"{law_id}|{law_title}|{label}")

    return dedupe_keep_order(doc_refs), dedupe_keep_order(article_refs)


def write_submission(rows: List[Dict[str, Any]], output_dir: Path, method: str, topk: int) -> None:
    json_path = output_dir / f"results_{method}_top{topk}.json"
    zip_path = output_dir / f"submission_{method}_top{topk}.zip"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    avg_docs = sum(len(row["relevant_docs"]) for row in rows) / len(rows)
    avg_articles = sum(len(row["relevant_articles"]) for row in rows) / len(rows)
    print(f"{zip_path} avg_docs={avg_docs:.2f} avg_articles={avg_articles:.2f}")


def main() -> None:
    args = parse_args()
    topks = [int(value.strip()) for value in args.topks.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_law_title_mapping(BASE_DIR / "data" / "law_id_to_title.json")

    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.max_questions:
        questions = questions[: args.max_questions]

    print(f"Loading RAG for method={args.method}; questions={len(questions)}; topks={topks}")
    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    all_rows = {topk: [] for topk in topks}
    with retrieval_mode(rag, args.method) as (use_hybrid, use_reranking):
        for index, item in enumerate(questions, 1):
            start = time.time()
            if args.verbose_retrieval:
                docs = rag.retrieve_documents(
                    item["question"],
                    use_hybrid=use_hybrid,
                    use_reranking=use_reranking,
                )
            else:
                with redirect_stdout(io.StringIO()):
                    docs = rag.retrieve_documents(
                        item["question"],
                        use_hybrid=use_hybrid,
                        use_reranking=use_reranking,
                    )
            elapsed = time.time() - start

            for topk in topks:
                doc_refs, article_refs = refs_from_docs(docs[:topk], mapping)
                all_rows[topk].append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "answer": "Theo các quy định pháp luật liên quan.",
                        "relevant_docs": doc_refs,
                        "relevant_articles": article_refs,
                    }
                )

            if index % 25 == 0 or index == len(questions):
                print(f"{args.method}: {index}/{len(questions)} ({elapsed:.2f}s last)", flush=True)

            if index % 100 == 0:
                try:
                    import torch

                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    elif torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

    for topk, rows in all_rows.items():
        write_submission(rows, output_dir, args.method, topk)


if __name__ == "__main__":
    main()
