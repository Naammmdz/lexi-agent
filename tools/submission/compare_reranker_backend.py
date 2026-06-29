#!/usr/bin/env python3
"""Compare local Vietnamese reranker vs FPT Cloud bge-reranker-v2-m3 on train."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_MERGED_CORPUS", "1")

from _paths import REPO_ROOT

os.chdir(REPO_ROOT)
from config import Config
from main.chatbot import VietnameseLegalRAG
from run_retrieval_tune_benchmark import apply_config
from submission_benchmark import (
    gold_sets,
    load_train_rows,
    macro_average,
    prf,
    pred_sets,
    refs_from_docs,
    retrieve,
)
from utils.submission_formatter import load_law_title_mapping

OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"


def eval_backend(
    rag: VietnameseLegalRAG,
    rows: list,
    mapping: dict,
    backend: str,
    cutoffs: list[int],
) -> dict:
    rag.set_reranker_backend(backend)
    label = f"{backend}_{apply_config(0.92, 0.78, False)}"
    per_cutoff = {c: {"doc_scores": [], "article_scores": [], "latencies": []} for c in cutoffs}

    for idx, row in enumerate(rows, 1):
        start = time.time()
        docs = retrieve(rag, row["question"], "hybrid_rerank", verbose_retrieval=False)
        latency = time.time() - start
        gold_docs, gold_articles = gold_sets(row["relevant_articles"])
        for cutoff in cutoffs:
            doc_refs, article_refs = refs_from_docs(docs[:cutoff], mapping)
            pred_docs, pred_articles = pred_sets(doc_refs, article_refs)
            bucket = per_cutoff[cutoff]
            bucket["doc_scores"].append(prf(pred_docs, gold_docs))
            bucket["article_scores"].append(prf(pred_articles, gold_articles))
            bucket["latencies"].append(latency)
        if idx % 20 == 0 or idx == len(rows):
            print(f"  [{backend}] {idx}/{len(rows)}", flush=True)

    out = {"backend": backend, "label": label, "cutoffs": {}}
    for cutoff in cutoffs:
        bucket = per_cutoff[cutoff]
        dp, dr, df = macro_average(bucket["doc_scores"])
        ap, ar, af = macro_average(bucket["article_scores"])
        out["cutoffs"][str(cutoff)] = {
            "ARTICLES_F2MACRO": round(af, 4),
            "ARTICLES_PRECISION": round(ap, 4),
            "ARTICLES_RECALL": round(ar, 4),
            "DOCS_F2MACRO": round(df, 4),
            "AVG_LATENCY_SEC": round(sum(bucket["latencies"]) / len(bucket["latencies"]), 3),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--output", default=str(OUT_DIR / "reranker_api_compare100.json"))
    args = parser.parse_args()

    if not os.getenv("FPT_RERANK_API_KEY"):
        print("Set FPT_RERANK_API_KEY env var for API backend", file=sys.stderr)
        sys.exit(1)

    apply_config(0.92, 0.78, False)
    Config.HYBRID_FUSION = "max"

    rows = load_train_rows(REPO_ROOT / "data" / "train" / "train_qna.csv", args.max_questions)
    mapping = load_law_title_mapping(Path(Config.LAW_TITLE_MAPPING_PATH))
    cutoffs = [1, 2]

    print(f"Comparing reranker backends on {len(rows)} train rows (no_wl, max fusion, narrow pool)")
    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    local = eval_backend(rag, rows, mapping, "local", cutoffs)
    api = eval_backend(rag, rows, mapping, "api", cutoffs)

    ref_k1 = local["cutoffs"]["1"]["ARTICLES_F2MACRO"]
    ref_k2 = local["cutoffs"]["2"]["ARTICLES_F2MACRO"]
    api_k1 = api["cutoffs"]["1"]["ARTICLES_F2MACRO"]
    api_k2 = api["cutoffs"]["2"]["ARTICLES_F2MACRO"]

    summary = {
        "local": local,
        "api": api,
        "delta_k1": round(api_k1 - ref_k1, 4),
        "delta_k2": round(api_k2 - ref_k2, 4),
        "api_worse_k1": api_k1 < ref_k1,
        "api_worse_k2": api_k2 < ref_k2,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RERANKER BACKEND COMPARE ===")
    print(f"local  k1={ref_k1} k2={ref_k2} lat={local['cutoffs']['1']['AVG_LATENCY_SEC']}s")
    print(f"api    k1={api_k1} k2={api_k2} lat={api['cutoffs']['1']['AVG_LATENCY_SEC']}s")
    print(f"Δ k1={summary['delta_k1']:+.4f}  Δ k2={summary['delta_k2']:+.4f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
