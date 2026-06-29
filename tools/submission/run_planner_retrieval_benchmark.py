#!/usr/bin/env python3
"""Benchmark: baseline hybrid_rerank vs LLM legal-keyword planner + multi-query RRF."""

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

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "submission"))
os.chdir(REPO_ROOT)

from config import Config
from main.chatbot import VietnameseLegalRAG
from run_retrieval_tune_benchmark import apply_config, load_rows_for_filter
from run_llm_rerank_pilot100 import apply_rrf_wide_baseline, evaluate_method
from submission_benchmark import retrieve
from utils.llm_query_planner import LegalQueryPlanner, retrieve_with_plan
from utils.submission_formatter import load_law_title_mapping


OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"
DEFAULT_OUT = OUT_DIR / "planner_retrieval_benchmark.json"
DEFAULT_CACHE = OUT_DIR / "planner_keyword_cache.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner keyword retrieval benchmark")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--cutoffs", default="1,2")
    parser.add_argument(
        "--question-filter",
        default="all",
        choices=["all", "multi_hop", "cross_doc", "easy"],
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("OLLAMA_PLANNER_MODEL", "qwen3-vl:2b"),
        help="Ollama planner model (<14B). Default qwen3-vl:2b for speed.",
    )
    parser.add_argument(
        "--planner-cache",
        default=str(DEFAULT_CACHE),
        help="JSONL cache; use model-specific file when switching models",
    )
    parser.add_argument(
        "--plan-workers",
        type=int,
        default=int(os.getenv("OLLAMA_PLANNER_WORKERS", "32")),
        help="Parallel Ollama planner calls (set OLLAMA_NUM_PARALLEL similarly)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--config",
        default="rrf_wide",
        choices=["rrf_wide", "baseline"],
        help="rrf_wide = same pool as rrf_swap_g008 tuning",
    )
    args = parser.parse_args()

    cutoffs = [int(x) for x in args.cutoffs.split(",") if x.strip()]
    rows = load_rows_for_filter(args.max_questions, args.question_filter)
    mapping = load_law_title_mapping(Path(Config.LAW_TITLE_MAPPING_PATH))

    if args.config == "rrf_wide":
        baseline_label = apply_rrf_wide_baseline()
    else:
        baseline_label = apply_config(0.92, 0.78, True, False, 5, "max", False, 15, 15)

    planner = LegalQueryPlanner(model=args.planner_model, cache_path=args.planner_cache)

    print(
        f"Rows={len(rows)} filter={args.question_filter} config={baseline_label} "
        f"planner={args.planner_model}",
        flush=True,
    )
    print(f"Collection={Config.COLLECTION_NAME}", flush=True)

    rag = VietnameseLegalRAG()
    if rag.bm25_retriever:
        rag.bm25_retriever.load_index()

    all_summaries: dict[str, list[dict[str, Any]]] = {}
    all_details: dict[str, list[dict[str, Any]]] = {}

    if not args.skip_baseline:
        print("\n=== baseline hybrid_rerank ===", flush=True)

        def baseline_fn(row: dict[str, Any]):
            docs = retrieve(rag, row["question"], "hybrid_rerank", verbose_retrieval=False)
            return docs, {"method": "baseline"}

        summaries, details = evaluate_method(
            label=f"baseline_{baseline_label}",
            rows=rows,
            mapping=mapping,
            cutoffs=cutoffs,
            docs_fn=baseline_fn,
            extra_meta={"method": "baseline", "config": baseline_label},
        )
        all_summaries["baseline"] = summaries
        all_details["baseline"] = details

    print("\n=== planner keywords + multi-query RRF ===", flush=True)
    questions = [row["question"] for row in rows]
    cached = sum(1 for q in questions if q.strip() in planner._cache)
    print(
        f"Pre-planning {len(questions) - cached} questions "
        f"({cached} cached) workers={args.plan_workers}",
        flush=True,
    )
    t_plan = time.time()
    planner.plan_batch(questions, workers=args.plan_workers)
    print(f"Planning done in {time.time() - t_plan:.1f}s", flush=True)

    def planner_fn(row: dict[str, Any]):
        question = row["question"]
        t0 = time.time()
        plan = planner.plan(question)
        plan_ms = time.time() - t0
        with redirect_stdout(io.StringIO()):
            docs = retrieve_with_plan(rag, question, plan, use_reranking=True)
        return docs, {
            "method": "planner",
            "intent": plan.intent,
            "keywords": plan.keywords,
            "sub_queries": plan.sub_queries,
            "parse_ok": plan.parse_ok,
            "plan_latency_sec": round(plan_ms, 3),
        }

    planner_summaries, planner_details = evaluate_method(
        label=f"planner_{baseline_label}",
        rows=rows,
        mapping=mapping,
        cutoffs=cutoffs,
        docs_fn=planner_fn,
        extra_meta={"method": "planner", "planner_model": args.planner_model},
    )
    all_summaries["planner"] = planner_summaries
    all_details["planner"] = planner_details

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": len(rows),
        "question_filter": args.question_filter,
        "planner_model": args.planner_model,
        "config": baseline_label,
        "summaries": all_summaries,
        "details": all_details,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {out_path}", flush=True)
    for name, summaries in all_summaries.items():
        k1 = next((s for s in summaries if s["cutoff"] == 1), None)
        if k1:
            print(
                f"  {name}: ART_F2@1={k1['ARTICLES_F2MACRO']:.4f} "
                f"P={k1['ARTICLES_PRECISION']:.4f} R={k1['ARTICLES_RECALL']:.4f} "
                f"lat={k1['AVG_LATENCY_SEC']:.2f}s",
                flush=True,
            )

    if "baseline" in all_summaries and "planner" in all_summaries:
        b = next(s for s in all_summaries["baseline"] if s["cutoff"] == 1)
        p = next(s for s in all_summaries["planner"] if s["cutoff"] == 1)
        delta = p["ARTICLES_F2MACRO"] - b["ARTICLES_F2MACRO"]
        print(f"\nDelta ART_F2@1 (planner - baseline): {delta:+.4f}", flush=True)


if __name__ == "__main__":
    main()
