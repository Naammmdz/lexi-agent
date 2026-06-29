#!/usr/bin/env python3
"""Benchmark P3 conditional cap_docs=2 on cross-doc train subset."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("USE_MERGED_CORPUS", "1")
os.environ.setdefault("USE_WIDE_RETRIEVAL_POOL", "1")
os.environ.setdefault("HYBRID_FUSION", "rrf")

from _paths import REPO_ROOT
from config import Config
from main.chatbot import VietnameseLegalRAG
from run_retrieval_tune_benchmark import load_rows_for_filter
from submission_benchmark import gold_sets, macro_average, prf, pred_sets, refs_from_docs, retrieve
from utils.cross_doc_cap import apply_conditional_second_doc, candidates_from_retrieved_docs, should_add_second_doc
from utils.submission_formatter import load_law_title_mapping

OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"


def apply_winning_config() -> None:
    Config.RERANKER_FUSION_ALPHA = 0.92
    Config.RERANK_SCORE_THRESHOLD_RATIO = 0.78
    Config.ENABLE_WITHIN_LAW_RESCORE = False
    Config.HYBRID_FUSION = "rrf"
    Config.RERANK_BEFORE_RETRIEVAL_TOP_K = Config.RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE
    Config.RERANKER_TOP_K = Config.RERANKER_TOP_K_WIDE


def evaluate_variant(
    rag: VietnameseLegalRAG,
    rows: list[dict],
    mapping: dict[str, str],
    label: str,
    conditional_cap_docs_2: bool,
    min_score: float,
    max_gap: float,
) -> dict:
    doc_scores = []
    article_scores = []
    latencies = []
    triggered = 0

    for idx, row in enumerate(rows, 1):
        start = time.time()
        docs = retrieve(rag, row["question"], "hybrid_rerank", verbose_retrieval=False)
        latency = time.time() - start
        latencies.append(latency)

        doc_refs, article_refs = refs_from_docs(docs[:1], mapping)
        pred_row = {
            "relevant_docs": list(doc_refs),
            "relevant_articles": list(article_refs),
        }

        candidates = candidates_from_retrieved_docs(docs, mapping)
        if conditional_cap_docs_2:
            if should_add_second_doc(candidates, mapping, min_score, max_gap):
                triggered += 1
            apply_conditional_second_doc(
                pred_row,
                candidates,
                mapping,
                min_score=min_score,
                max_gap=max_gap,
                cap_docs=2,
            )

        gold_docs, gold_articles = gold_sets(row["relevant_articles"])
        pred_docs, pred_articles = pred_sets(pred_row["relevant_docs"], pred_row["relevant_articles"])
        doc_scores.append(prf(pred_docs, gold_docs))
        article_scores.append(prf(pred_articles, gold_articles))

        if idx % 20 == 0 or idx == len(rows):
            print(f"  [{label}] {idx}/{len(rows)}", flush=True)

    dp, dr, df = macro_average(doc_scores)
    ap, ar, af = macro_average(article_scores)
    return {
        "label": label,
        "conditional_cap_docs_2": conditional_cap_docs_2,
        "cross_doc_min_score": min_score,
        "cross_doc_max_gap": max_gap,
        "questions": len(rows),
        "triggered": triggered,
        "ARTICLES_F2MACRO": round(af, 4),
        "ARTICLES_PRECISION": round(ap, 4),
        "ARTICLES_RECALL": round(ar, 4),
        "DOCS_F2MACRO": round(df, 4),
        "DOCS_PRECISION": round(dp, 4),
        "DOCS_RECALL": round(dr, 4),
        "AVG_LATENCY_SEC": round(sum(latencies) / len(latencies), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=67)
    parser.add_argument("--min-score", type=float, default=0.9)
    parser.add_argument("--max-gap", type=float, default=0.03)
    parser.add_argument("--output", default=str(OUT_DIR / "p3_cross_doc67.json"))
    args = parser.parse_args()

    apply_winning_config()
    rows = load_rows_for_filter(args.max_questions, "cross_doc")
    mapping = load_law_title_mapping(Path(Config.LAW_TITLE_MAPPING_PATH))

    print(
        f"P3 benchmark | rows={len(rows)} | min_score={args.min_score} | "
        f"max_gap={args.max_gap} | collection={Config.COLLECTION_NAME}"
    )

    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    results = [
        evaluate_variant(rag, rows, mapping, "baseline_cap1", False, args.min_score, args.max_gap),
        evaluate_variant(
            rag,
            rows,
            mapping,
            "conditional_cap2",
            True,
            args.min_score,
            args.max_gap,
        ),
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    base, p3 = results
    delta_art = round(p3["ARTICLES_F2MACRO"] - base["ARTICLES_F2MACRO"], 4)
    delta_doc = round(p3["DOCS_F2MACRO"] - base["DOCS_F2MACRO"], 4)

    print("\n=== RESULTS ===")
    for r in results:
        print(
            f"{r['label']:<20} art_F2={r['ARTICLES_F2MACRO']:.4f} "
            f"doc_F2={r['DOCS_F2MACRO']:.4f} "
            f"doc_R={r['DOCS_RECALL']:.4f} triggered={r['triggered']}"
        )
    print(f"\nΔ art_F2={delta_art}  Δ doc_F2={delta_doc}  (triggered {p3['triggered']}/{len(rows)})")
    print(f"Wrote {out}")

    gate = max(delta_art, delta_doc)
    if gate >= 0.01:
        print("\n>>> SIGNAL: apply conditional cap_docs=2 on submission build")
    else:
        print("\n>>> NO SIGNAL: keep cap_docs=1")


if __name__ == "__main__":
    main()
