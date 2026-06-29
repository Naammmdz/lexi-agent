#!/usr/bin/env python3
"""Sweep retrieval fusion/threshold on train_qna (hybrid_rerank, k=1,2)."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

# Merged corpus is the active production stack.
os.environ.setdefault("USE_MERGED_CORPUS", "1")

from _paths import REPO_ROOT
from config import Config
from main.chatbot import VietnameseLegalRAG
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


def evaluate_config(
    rag: VietnameseLegalRAG,
    rows: list[dict[str, Any]],
    mapping: dict[str, str],
    label: str,
    cutoffs: list[int],
    question_filter: str = "all",
) -> list[dict[str, Any]]:
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
            print(f"  [{label}] {idx}/{len(rows)}", flush=True)

    summaries = []
    for cutoff in cutoffs:
        bucket = per_cutoff[cutoff]
        dp, dr, df = macro_average(bucket["doc_scores"])
        ap, ar, af = macro_average(bucket["article_scores"])
        summaries.append(
            {
                "label": label,
                "fusion_alpha": Config.RERANKER_FUSION_ALPHA,
                "threshold_ratio": Config.RERANK_SCORE_THRESHOLD_RATIO,
                "within_law_rescore": Config.ENABLE_WITHIN_LAW_RESCORE,
                "hybrid_fusion": Config.HYBRID_FUSION,
                "two_stage_within_law": Config.ENABLE_TWO_STAGE_WITHIN_LAW_RERANK,
                "pre_retrieval_top_k": Config.RERANK_BEFORE_RETRIEVAL_TOP_K,
                "reranker_top_k": Config.RERANKER_TOP_K,
                "query_decomposition": Config.ENABLE_QUERY_DECOMPOSITION,
                "question_filter": question_filter,
                "law_shortlist": Config.ENABLE_LAW_SHORTLIST,
                "law_shortlist_top_k": Config.LAW_SHORTLIST_TOP_K,
                "cutoff": cutoff,
                "questions": len(rows),
                "ARTICLES_F2MACRO": round(af, 4),
                "ARTICLES_PRECISION": round(ap, 4),
                "ARTICLES_RECALL": round(ar, 4),
                "DOCS_F2MACRO": round(df, 4),
                "AVG_LATENCY_SEC": round(sum(bucket["latencies"]) / len(bucket["latencies"]), 3),
            }
        )
    return summaries


def apply_config(
    fusion: float,
    threshold: float,
    within_law: bool,
    law_shortlist: bool = False,
    shortlist_k: int = 5,
    hybrid_fusion: str = "max",
    two_stage: bool = False,
    pre_k: int = 15,
    rerank_k: int = 15,
    query_decompose: bool = False,
) -> str:
    Config.RERANKER_FUSION_ALPHA = fusion
    Config.RERANK_SCORE_THRESHOLD_RATIO = threshold
    Config.ENABLE_WITHIN_LAW_RESCORE = within_law
    Config.ENABLE_LAW_SHORTLIST = law_shortlist
    Config.LAW_SHORTLIST_TOP_K = shortlist_k
    Config.HYBRID_FUSION = hybrid_fusion
    Config.ENABLE_TWO_STAGE_WITHIN_LAW_RERANK = two_stage
    Config.RERANK_BEFORE_RETRIEVAL_TOP_K = pre_k
    Config.RERANKER_TOP_K = rerank_k
    Config.ENABLE_QUERY_DECOMPOSITION = query_decompose
    sl = f"_sl{shortlist_k}" if law_shortlist else ""
    hf = "_rrf" if hybrid_fusion == "rrf" else ""
    ts = "_ts" if two_stage else ""
    pool = f"_p{pre_k}r{rerank_k}" if pre_k != 15 or rerank_k != 15 else ""
    dq = "_dq" if query_decompose else ""
    return f"a{fusion:.2f}_t{threshold:.2f}_wl{int(within_law)}{sl}{hf}{ts}{pool}{dq}"


def load_rows_for_filter(max_questions: int, question_filter: str) -> list[dict[str, Any]]:
    train_path = REPO_ROOT / "data" / "train" / "train_qna.csv"
    all_rows = load_train_rows(train_path, 0)
    if question_filter == "all":
        return all_rows[:max_questions] if max_questions else all_rows

    tags_path = OUT_DIR / "train_subset_tags.json"
    if not tags_path.exists():
        raise SystemExit(f"Missing {tags_path} — run tools/submission/tag_train_subsets.py first")

    payload = json.loads(tags_path.read_text(encoding="utf-8"))
    tag_rows = payload["rows"]
    filtered: list[dict[str, Any]] = []
    for tag in tag_rows:
        if not tag.get(question_filter):
            continue
        idx = tag["index"]
        if idx >= len(all_rows):
            continue
        filtered.append(all_rows[idx])
        if max_questions and len(filtered) >= max_questions:
            break
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--cutoffs", default="1,2")
    parser.add_argument(
        "--configs",
        default="baseline",
        help="baseline | no_wl | compare_v2 | compare_v3 | compare_p2 | compare_wl | compare_shortlist | sweep | comma list like 0.88:0.78",
    )
    parser.add_argument(
        "--question-filter",
        default="all",
        choices=["all", "multi_hop", "cross_doc", "easy"],
        help="Filter train rows using train_subset_tags.json",
    )
    parser.add_argument("--output", default=str(OUT_DIR / "retrieval_tune_results.json"))
    args = parser.parse_args()

    cutoffs = [int(x) for x in args.cutoffs.split(",") if x.strip()]
    rows = load_rows_for_filter(args.max_questions, args.question_filter)
    mapping = load_law_title_mapping(Path(Config.LAW_TITLE_MAPPING_PATH))

    print(
        f"Train rows: {len(rows)} | filter: {args.question_filter} | cutoffs: {cutoffs} | "
        f"merged corpus: {Config.USE_MERGED_CORPUS}"
    )
    print(f"Collection: {Config.COLLECTION_NAME}")

    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    _NO_WL = (0.92, 0.78, False, False, 5, "max", False, 15, 15)
    _WIDE = Config.RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE, Config.RERANKER_TOP_K_WIDE

    if args.configs == "baseline":
        configs = [(0.92, 0.78, True, False, 5, "max", False, 15, 15)]
    elif args.configs == "compare_shortlist":
        configs = [
            (0.92, 0.78, False, False, 5, "max", False, 15, 15),
            (0.92, 0.78, False, True, 3, "max", False, 15, 15),
            (0.92, 0.78, False, True, 5, "max", False, 15, 15),
            (0.92, 0.78, False, True, 7, "max", False, 15, 15),
        ]
    elif args.configs == "compare_wl":
        configs = [
            (0.92, 0.78, True, False, 5, "max", False, 15, 15),
            (0.92, 0.78, False, False, 5, "max", False, 15, 15),
        ]
    elif args.configs == "no_wl":
        configs = [_NO_WL]
    elif args.configs == "compare_v2":
        base = (0.92, 0.78, False, False, 5)
        configs = [
            (*base, "max", False, 15, 15),
            (*base, "rrf", False, 15, 15),
            (*base, "max", True, 15, 15),
            (*base, "rrf", True, 15, 15),
        ]
    elif args.configs == "compare_v3":
        base = (0.92, 0.78, False, False, 5)
        configs = [
            (*base, "max", False, 15, 15),
            (*base, "rrf", False, 15, 15),
            (*base, "rrf", False, *_WIDE),
        ]
    elif args.configs == "compare_p2":
        base = (0.92, 0.78, False, False, 5)
        wide = _WIDE
        configs = [
            (*base, "rrf", False, *wide, False),
            (*base, "rrf", False, *wide, True),
        ]
    elif args.configs == "sweep":
        configs = [
            (0.85, 0.78, True, False, 5, "max", False, 15, 15),
            (0.88, 0.78, True, False, 5, "max", False, 15, 15),
            (0.92, 0.78, True, False, 5, "max", False, 15, 15),
            (0.95, 0.78, True, False, 5, "max", False, 15, 15),
            (0.92, 0.75, True, False, 5, "max", False, 15, 15),
            (0.92, 0.82, True, False, 5, "max", False, 15, 15),
            (0.88, 0.75, True, False, 5, "max", False, 15, 15),
        ]
    else:
        configs = []
        for part in args.configs.split(","):
            fusion_s, thresh_s = part.split(":")
            configs.append((float(fusion_s), float(thresh_s), True, False, 5, "max", False, 15, 15))

    all_results: list[dict[str, Any]] = []
    for cfg in configs:
        query_decompose = False
        if len(cfg) == 10:
            (
                fusion,
                threshold,
                within_law,
                law_shortlist,
                shortlist_k,
                hybrid_fusion,
                two_stage,
                pre_k,
                rerank_k,
                query_decompose,
            ) = cfg
        else:
            (
                fusion,
                threshold,
                within_law,
                law_shortlist,
                shortlist_k,
                hybrid_fusion,
                two_stage,
                pre_k,
                rerank_k,
            ) = cfg
        label = apply_config(
            fusion,
            threshold,
            within_law,
            law_shortlist,
            shortlist_k,
            hybrid_fusion,
            two_stage,
            pre_k,
            rerank_k,
            query_decompose=query_decompose,
        )
        print(f"\n=== {label} ===", flush=True)
        all_results.extend(
            evaluate_config(rag, rows, mapping, label, cutoffs, args.question_filter)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== RANKED BY ARTICLES_F2 @ k=1 ===")
    k1 = sorted([r for r in all_results if r["cutoff"] == 1], key=lambda x: -x["ARTICLES_F2MACRO"])
    for r in k1:
        print(
            f"{r['label']:<22} art_F2={r['ARTICLES_F2MACRO']:.4f} "
            f"P={r['ARTICLES_PRECISION']:.4f} R={r['ARTICLES_RECALL']:.4f} "
            f"doc_F2={r['DOCS_F2MACRO']:.4f} lat={r['AVG_LATENCY_SEC']:.2f}s"
        )

    print("\n=== @ k=2 ===")
    k2 = sorted([r for r in all_results if r["cutoff"] == 2], key=lambda x: -x["ARTICLES_F2MACRO"])
    for r in k2:
        print(
            f"{r['label']:<22} art_F2={r['ARTICLES_F2MACRO']:.4f} "
            f"P={r['ARTICLES_PRECISION']:.4f} R={r['ARTICLES_RECALL']:.4f}"
        )

    # Prior baseline (100q, no_wl, no shortlist) from retrieval_no_wl100.json
    STORED_BASELINE_K1 = 0.6967
    STORED_BASELINE_K2 = 0.7261

    ref_row = next(
        (
            r
            for r in k1
            if not r.get("law_shortlist")
            and not r.get("two_stage_within_law")
            and not r.get("query_decomposition")
            and (
                args.configs != "compare_p2"
                or r.get("hybrid_fusion") == "rrf"
            )
            and (
                args.configs != "compare_p2"
                or r.get("pre_retrieval_top_k") == Config.RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE
            )
        ),
        k1[0] if k1 else None,
    )
    ref_k1 = ref_row["ARTICLES_F2MACRO"] if ref_row else STORED_BASELINE_K1
    ref_k2_row = next(
        (
            r
            for r in k2
            if not r.get("law_shortlist")
            and not r.get("two_stage_within_law")
            and not r.get("query_decomposition")
            and (
                args.configs != "compare_p2"
                or r.get("hybrid_fusion") == "rrf"
            )
            and (
                args.configs != "compare_p2"
                or r.get("pre_retrieval_top_k") == Config.RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE
            )
        ),
        k2[0] if k2 else None,
    )
    ref_k2 = ref_k2_row["ARTICLES_F2MACRO"] if ref_k2_row else STORED_BASELINE_K2

    best_k1 = k1[0] if k1 else None
    best_k2 = k2[0] if k2 else None
    delta_k1 = round(best_k1["ARTICLES_F2MACRO"] - ref_k1, 4) if best_k1 else 0.0
    delta_k2 = round(best_k2["ARTICLES_F2MACRO"] - ref_k2, 4) if best_k2 else 0.0

    print(f"\nBest vs no-shortlist ref (k1={ref_k1} k2={ref_k2}): k1 Δ={delta_k1}  k2 Δ={delta_k2}")
    print(f"Wrote {out}")

    gate = max(delta_k1, delta_k2)
    if gate >= 0.01:
        print(f"\n>>> SIGNAL: best config beats ref by >=0.01 — OK to re-cache + build submission")
        if best_k1:
            print(f"    best k=1: {best_k1['label']}")
    else:
        print("\n>>> NO SIGNAL: keep no_wl_tight_v1, do not re-cache yet")


if __name__ == "__main__":
    main()
