"""Cache fresh live hybrid_rerank retrieval candidates for the test questions.

The current curated submission only shares its top anchor with the stale
``results.json`` for ~9% of rows, so ``results.json`` cannot be used as a recall
source.  This script runs the *live* ``VietnameseLegalRAG`` hybrid+rerank
pipeline over the 2000 R2AI test questions and stores the ordered candidate
(law_id, article_id, titles) per row.  The cache is then consumed by
``create_recall_boost_submission.py``.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from main.chatbot import VietnameseLegalRAG
from utils.submission_formatter import (
    article_label,
    canonical_law_id,
    format_law_title,
    get_mapping_title,
    load_law_title_mapping,
)

INPUT_PATH = REPO_ROOT / "R2AIStage1DATA.json"
MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
OUTPUT_PATH = REPO_ROOT / "data" / "augmented" / "live_retrieval_test_v2.json"
OUTPUT_PATH_V3 = REPO_ROOT / "data" / "augmented" / "live_retrieval_test_v3_merged.json"
LEGACY_OUTPUT_PATH = REPO_ROOT / "data" / "augmented" / "live_retrieval_test.json"


def extract_candidates(docs: list[dict[str, Any]], mapping: dict[str, str], keep: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        metadata = doc.get("metadata", {})
        law_id = canonical_law_id(metadata.get("law_id", ""))
        if not law_id:
            continue
        label = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
        key = (law_id, label.lower())
        if key in seen:
            continue
        seen.add(key)
        title = format_law_title(law_id, get_mapping_title(mapping, law_id))
        out.append(
            {
                "law_id": law_id,
                "label": label,
                "doc_ref": f"{law_id}|{title}",
                "article_ref": f"{law_id}|{title}|{label}" if label else "",
                "score": float(doc.get("score", 0.0) or 0.0),
            }
        )
        if len(out) >= keep:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--mapping", default=str(MAPPING_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--legacy-output", default=str(LEGACY_OUTPUT_PATH))
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    parser.add_argument("--start", type=int, default=0, help="Skip first N questions (0-based offset)")
    parser.add_argument("--resume", action="store_true", help="Load existing --output and skip cached ids")
    parser.add_argument("--bm25-only", action="store_true", help="Skip vector search (no Qdrant needed).")
    args = parser.parse_args()

    mapping = load_law_title_mapping(Path(args.mapping))
    questions = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.start:
        questions = questions[args.start :]
    if args.limit:
        questions = questions[: args.limit]

    output_path = Path(args.output)
    cache: dict[str, list[dict[str, str]]] = {}
    if args.resume and output_path.exists() and output_path.stat().st_size > 2:
        cache = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Resume: loaded {len(cache)} cached rows from {output_path}", flush=True)

    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    total = len(questions)
    processed = 0
    for item in questions:
        row_id = str(item["id"])
        if args.resume and row_id in cache:
            continue
        question = item["question"]
        with redirect_stdout(io.StringIO()):
            docs = rag.retrieve_documents(
                question,
                use_hybrid=not args.bm25_only,
                use_reranking=True,
            )
        cache[row_id] = extract_candidates(docs, mapping, args.keep)
        processed += 1
        if processed % 25 == 0 or processed == total or len(cache) % 25 == 0:
            done = len(cache)
            print(f"{done}/2000 cached (+{processed} this run)", flush=True)
            output_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    output_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if args.legacy_output:
        Path(args.legacy_output).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"DONE wrote {len(cache)} rows -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
