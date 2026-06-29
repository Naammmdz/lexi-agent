#!/usr/bin/env python3
"""Pilot benchmark: hybrid RRF pool -> top-N candidates -> LLM rerank vs cross-encoder."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_MERGED_CORPUS", "1")

from _paths import REPO_ROOT

os.chdir(REPO_ROOT)

from config import Config
from main.chatbot import VietnameseLegalRAG
from run_retrieval_tune_benchmark import apply_config, load_rows_for_filter
from submission_benchmark import gold_sets, macro_average, prf, pred_sets, refs_from_docs, retrieve
from utils.llm_reranker import LLMReranker
from utils.submission_formatter import load_law_title_mapping


OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"
DEFAULT_OUT = OUT_DIR / "llm_rerank_pilot100.json"
DEFAULT_CACHE = OUT_DIR / "llm_rerank_pilot_cache.jsonl"
BASELINE_K1 = 0.7522  # rrf_p50r20 on train 100q (p4_corpus_v2_100q.json)


def apply_rrf_wide_baseline() -> str:
    os.environ["USE_WIDE_RETRIEVAL_POOL"] = "1"
    os.environ["HYBRID_FUSION"] = "rrf"
    Config.HYBRID_FUSION = "rrf"
    Config.RERANK_BEFORE_RETRIEVAL_TOP_K = Config.RERANK_BEFORE_RETRIEVAL_TOP_K_WIDE
    Config.RERANKER_TOP_K = Config.RERANKER_TOP_K_WIDE
    return apply_config(0.92, 0.78, False, hybrid_fusion="rrf", pre_k=50, rerank_k=20)


def retrieve_hybrid_pool(rag: VietnameseLegalRAG, question: str, pool_size: int) -> list[dict[str, Any]]:
    with redirect_stdout(io.StringIO()):
        return rag._hybrid_retrieve_pool(question, pool_size, pool_size)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cache[row["cache_key"]] = row
    return cache


def append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_key(question: str, backend: str, candidate_k: int) -> str:
    return f"{backend}|k{candidate_k}|{question.strip()}"


def evaluate_method(
    *,
    label: str,
    rows: list[dict[str, Any]],
    mapping: dict[str, str],
    cutoffs: list[int],
    docs_fn,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_cutoff = {c: {"doc_scores": [], "article_scores": [], "latencies": []} for c in cutoffs}
    details: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, 1):
        start = time.time()
        docs, meta = docs_fn(row)
        latency = time.time() - start
        gold_docs, gold_articles = gold_sets(row["relevant_articles"])

        for cutoff in cutoffs:
            doc_refs, article_refs = refs_from_docs(docs[:cutoff], mapping)
            pred_docs, pred_articles = pred_sets(doc_refs, article_refs)
            bucket = per_cutoff[cutoff]
            bucket["doc_scores"].append(prf(pred_docs, gold_docs))
            bucket["article_scores"].append(prf(pred_articles, gold_articles))
            bucket["latencies"].append(latency)

        details.append(
            {
                "index": idx,
                "question": row["question"],
                "gold_docs": sorted(gold_docs),
                "gold_articles": sorted([f"{a[0]}|{a[1]}" for a in gold_articles]),
                "pred_top1": refs_from_docs(docs[:1], mapping)[1][:1],
                "meta": meta,
            }
        )

        if idx % 10 == 0 or idx == len(rows):
            print(f"  [{label}] {idx}/{len(rows)}", flush=True)

    summaries = []
    for cutoff in cutoffs:
        bucket = per_cutoff[cutoff]
        dp, dr, df = macro_average(bucket["doc_scores"])
        ap, ar, af = macro_average(bucket["article_scores"])
        summary = {
            "label": label,
            "cutoff": cutoff,
            "questions": len(rows),
            "ARTICLES_F2MACRO": round(af, 4),
            "ARTICLES_PRECISION": round(ap, 4),
            "ARTICLES_RECALL": round(ar, 4),
            "DOCS_F2MACRO": round(df, 4),
            "AVG_LATENCY_SEC": round(sum(bucket["latencies"]) / len(bucket["latencies"]), 3),
        }
        if extra_meta:
            summary.update(extra_meta)
        summaries.append(summary)
    return summaries, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--cutoffs", default="1,2")
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--llm-backend",
        default="local",
        choices=["local", "gemini", "mock", "passthrough"],
        help="local=Qwen3-4B (BTC-compliant), gemini=API (not for final submit)",
    )
    parser.add_argument("--llm-model", default="", help="Override gemini model name")
    parser.add_argument("--question-filter", default="all", choices=["all", "multi_hop", "cross_doc", "easy"])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--details-output", default="")
    parser.add_argument(
        "--disable-reranker-for-memory",
        action="store_true",
        help="Skip loading cross-encoder when only benchmarking LLM over hybrid pool",
    )
    args = parser.parse_args()

    cutoffs = [int(x) for x in args.cutoffs.split(",") if x.strip()]
    rows = load_rows_for_filter(args.max_questions, args.question_filter)
    mapping = load_law_title_mapping(Path(Config.LAW_TITLE_MAPPING_PATH))
    cache_path = Path(args.cache_file)
    cache = load_cache(cache_path)

    baseline_label = apply_rrf_wide_baseline()
    print(
        f"Rows={len(rows)} filter={args.question_filter} pool={args.pool_size} "
        f"candidates={args.candidate_k} llm={args.llm_backend} baseline={baseline_label}"
    )
    print(f"Collection={Config.COLLECTION_NAME}")

    if args.disable_reranker_for_memory or (args.skip_baseline and args.llm_backend == "local"):
        Config.ENABLE_RERANKING = False
        print("Reranker disabled to save RAM for local LLM pilot", flush=True)

    if args.llm_backend == "local":
        from utils.local_legal_llm import competition_model_profile

        profile = competition_model_profile(args.llm_model or None)
        print(f"BTC model profile: {json.dumps(profile, ensure_ascii=False)}", flush=True)

    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    llm = LLMReranker(
        backend=args.llm_backend,
        model=args.llm_model or None,
    )

    all_summaries: list[dict[str, Any]] = []
    all_details: dict[str, list[dict[str, Any]]] = {}

    if not args.skip_baseline:
        print("\n=== baseline cross-encoder ===", flush=True)

        def baseline_fn(row: dict[str, Any]):
            docs = retrieve(rag, row["question"], "hybrid_rerank", verbose_retrieval=False)
            return docs, {"method": "cross_encoder_rrf_wide"}

        baseline_summaries, baseline_details = evaluate_method(
            label=f"baseline_{baseline_label}",
            rows=rows,
            mapping=mapping,
            cutoffs=cutoffs,
            docs_fn=baseline_fn,
            extra_meta={
                "method": "cross_encoder",
                "pool_size": args.pool_size,
                "candidate_k": args.candidate_k,
            },
        )
        all_summaries.extend(baseline_summaries)
        all_details["baseline"] = baseline_details

    print("\n=== llm rerank over hybrid pool ===", flush=True)

    def llm_fn(row: dict[str, Any]):
        question = row["question"]
        key = cache_key(question, args.llm_backend, args.candidate_k)
        if key in cache:
            payload = cache[key]
            docs = payload["docs"]
            meta = payload.get("meta", {})
            meta["cache_hit"] = True
            return docs, meta

        pool = retrieve_hybrid_pool(rag, question, args.pool_size)
        candidates = pool[: args.candidate_k]
        result = llm.rerank(question, candidates, mapping, top_k=args.candidate_k)
        docs = result.docs
        meta = {
            "method": "llm_rerank",
            "backend": result.backend,
            "model": llm.model,
            "parse_ok": result.parse_ok,
            "llm_latency_sec": round(result.latency_sec, 3),
            "ranking": result.ranking,
            "cache_hit": False,
        }
        append_cache(
            cache_path,
            {
                "cache_key": key,
                "question": question,
                "docs": docs,
                "meta": meta,
                "raw_response": result.raw_response,
            },
        )
        cache[key] = {"docs": docs, "meta": meta}
        return docs, meta

    llm_label = f"llm_{args.llm_backend}_p{args.pool_size}c{args.candidate_k}"
    llm_summaries, llm_details = evaluate_method(
        label=llm_label,
        rows=rows,
        mapping=mapping,
        cutoffs=cutoffs,
        docs_fn=llm_fn,
        extra_meta={
            "method": "llm_rerank",
            "llm_backend": args.llm_backend,
            "llm_model": llm.model,
            "pool_size": args.pool_size,
            "candidate_k": args.candidate_k,
        },
    )
    all_summaries.extend(llm_summaries)
    all_details["llm"] = llm_details

    out = Path(args.output)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "max_questions": len(rows),
            "question_filter": args.question_filter,
            "pool_size": args.pool_size,
            "candidate_k": args.candidate_k,
            "llm_backend": args.llm_backend,
            "llm_model": llm.model,
            "baseline_label": baseline_label,
            "stored_baseline_k1": BASELINE_K1,
        },
        "summaries": all_summaries,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.details_output:
        Path(args.details_output).write_text(
            json.dumps(all_details, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n=== RESULTS @ k=1 ===")
    k1_rows = sorted([r for r in all_summaries if r["cutoff"] == 1], key=lambda x: -x["ARTICLES_F2MACRO"])
    for r in k1_rows:
        print(
            f"{r['label']:<34} art_F2={r['ARTICLES_F2MACRO']:.4f} "
            f"P={r['ARTICLES_PRECISION']:.4f} R={r['ARTICLES_RECALL']:.4f} "
            f"lat={r['AVG_LATENCY_SEC']:.2f}s"
        )

    llm_k1 = next((r for r in k1_rows if r.get("method") == "llm_rerank"), None)
    base_k1 = next((r for r in k1_rows if r.get("method") == "cross_encoder"), None)
    ref_k1 = base_k1["ARTICLES_F2MACRO"] if base_k1 else BASELINE_K1
    llm_score = llm_k1["ARTICLES_F2MACRO"] if llm_k1 else 0.0
    delta = round(llm_score - ref_k1, 4)
    print(f"\nLLM vs baseline k1: Δ={delta} (ref={ref_k1}, llm={llm_score})")
    print(f"Wrote {out}")
    print(f"Cache: {cache_path}")

    if delta >= 0.01:
        print("\n>>> SIGNAL: LLM rerank beats baseline by >=0.01 — consider cache_live_retrieval integration")
    else:
        print("\n>>> NO SIGNAL: keep cross-encoder baseline")


if __name__ == "__main__":
    main()
