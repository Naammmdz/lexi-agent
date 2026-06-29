#!/usr/bin/env python3
"""Build a QA-optimized submission from a strong IR base (e.g. rrf_swap_g008).

Keeps relevant_docs / relevant_articles from the IR zip and rewrites answer using
grounded generation from the legal corpus (vLLM batch, Qwen3-4B, Ollama, or extractive).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from utils.qa_answer_generator import (
    build_corpus_lookup,
    generate_grounded_answer,
    generate_ollama_batch_answers,
    generate_vllm_batch_answers,
)

OUTPUT_DIR = REPO_ROOT / "submission_variants"
DEFAULT_BASE = OUTPUT_DIR / "rrf_swap_g008.json"
DEFAULT_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "qa_promote_g008.zip"
G008_PLACEHOLDER = "áp dụng căn cứ nêu trên"


def is_regenerated_answer(answer: str) -> bool:
    text = str(answer or "").strip()
    if not text or G008_PLACEHOLDER in text.lower():
        return False
    return "Lưu ý:" in text and text.startswith("Căn cứ pháp luật:")


def load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def write_submission(rows: list[dict[str, Any]], output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BTC QA promote submission")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default="")
    parser.add_argument(
        "--backend",
        choices=("extractive", "local", "ollama", "vllm"),
        default="extractive",
        help="vllm=batched GPU inference; local=transformers; extractive=no LLM",
    )
    parser.add_argument("--model", default="", help="HF model id or Ollama model name")
    parser.add_argument("--max-articles", type=int, default=3)
    parser.add_argument("--max-chars-per-article", type=int, default=1200)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8, help="Ollama/vLLM parallel batch size")
    parser.add_argument("--no-disclaimer", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all answers even if checkpoint exists",
    )
    args = parser.parse_args()

    if args.backend == "vllm":
        args.max_articles = min(args.max_articles, 2)
        if args.max_chars_per_article > 500:
            args.max_chars_per_article = 400
        if args.max_new_tokens > 256:
            args.max_new_tokens = 200
    elif args.backend == "ollama":
        args.max_articles = min(args.max_articles, 3)
        workers_default = min(args.batch_size, 10)
        os.environ.setdefault("OLLAMA_WORKERS", str(workers_default))

    base_path = Path(args.base)
    output_path = Path(args.output)
    corpus_path = Path(args.corpus)
    debug_path = Path(args.debug) if args.debug else output_path.parent / f"{output_path.stem}_debug.csv"

    all_rows = load_rows(base_path)
    by_id = {str(r["id"]): dict(r) for r in all_rows}
    row_ids = [str(r["id"]) for r in all_rows]
    if args.start:
        row_ids = row_ids[args.start :]
    if args.limit:
        row_ids = row_ids[: args.limit]

    answers_cache: dict[str, str] = {}
    if args.resume and not args.force and output_path.with_suffix(".json").exists():
        cached_rows = load_rows(output_path.with_suffix(".json"))
        answers_cache = {
            str(r["id"]): str(r.get("answer", ""))
            for r in cached_rows
            if is_regenerated_answer(str(r.get("answer", "")))
        }
        print(f"Resume: {len(answers_cache)} answers loaded", flush=True)

    lookup = build_corpus_lookup(corpus_path)
    model_name = args.model or None
    debug_rows: list[dict[str, str]] = []
    processed = 0

    pending_ids: list[str] = []
    for row_id in row_ids:
        if args.resume and not args.force and row_id in answers_cache:
            by_id[row_id]["answer"] = answers_cache[row_id]
            continue
        pending_ids.append(row_id)

    if args.backend == "vllm":
        for offset in range(0, len(pending_ids), args.batch_size):
            chunk_ids = pending_ids[offset : offset + args.batch_size]
            items = [
                {
                    "question": str(by_id[rid].get("question", "")),
                    "article_refs": list(by_id[rid].get("relevant_articles") or []),
                }
                for rid in chunk_ids
            ]
            answers = generate_vllm_batch_answers(
                items,
                lookup,
                model_name=model_name,
                max_articles=args.max_articles,
                max_chars_per_article=args.max_chars_per_article,
                max_new_tokens=args.max_new_tokens,
                include_disclaimer=not args.no_disclaimer,
            )
            for rid, answer in zip(chunk_ids, answers):
                by_id[rid]["answer"] = answer
                answers_cache[rid] = answer
                processed += 1
                debug_rows.append(
                    {
                        "id": rid,
                        "backend": args.backend,
                        "n_articles": str(len(by_id[rid].get("relevant_articles") or [])),
                        "answer_len": str(len(answer)),
                        "question": str(by_id[rid].get("question", ""))[:120],
                        "answer_preview": answer[:200],
                    }
                )
            print(
                f"generated {processed}/{len(pending_ids)} answers (batch {len(chunk_ids)})",
                flush=True,
            )
            if args.resume or args.backend == "ollama":
                write_submission(list(by_id.values()), output_path)
    elif args.backend == "ollama":
        workers = int(os.getenv("OLLAMA_WORKERS", str(min(args.batch_size, 10))))
        for offset in range(0, len(pending_ids), args.batch_size):
            chunk_ids = pending_ids[offset : offset + args.batch_size]
            items = [
                {
                    "question": str(by_id[rid].get("question", "")),
                    "article_refs": list(by_id[rid].get("relevant_articles") or []),
                }
                for rid in chunk_ids
            ]
            answers = generate_ollama_batch_answers(
                items,
                lookup,
                model_name=model_name,
                max_articles=args.max_articles,
                max_chars_per_article=args.max_chars_per_article,
                max_new_tokens=args.max_new_tokens,
                include_disclaimer=not args.no_disclaimer,
                workers=workers,
            )
            for rid, answer in zip(chunk_ids, answers):
                by_id[rid]["answer"] = answer
                answers_cache[rid] = answer
                processed += 1
                debug_rows.append(
                    {
                        "id": rid,
                        "backend": args.backend,
                        "n_articles": str(len(by_id[rid].get("relevant_articles") or [])),
                        "answer_len": str(len(answer)),
                        "question": str(by_id[rid].get("question", ""))[:120],
                        "answer_preview": answer[:200],
                    }
                )
            print(
                f"generated {processed}/{len(pending_ids)} answers (ollama batch {len(chunk_ids)})",
                flush=True,
            )
            if args.resume or args.backend == "ollama":
                write_submission(list(by_id.values()), output_path)
    else:
        for row_id in pending_ids:
            row = by_id[row_id]
            refs = list(row.get("relevant_articles") or [])
            answer = generate_grounded_answer(
                question=str(row.get("question", "")),
                article_refs=refs,
                lookup=lookup,
                backend=args.backend,
                max_articles=args.max_articles,
                max_chars_per_article=args.max_chars_per_article,
                max_new_tokens=args.max_new_tokens,
                model_name=model_name,
                include_disclaimer=not args.no_disclaimer,
            )
            row["answer"] = answer
            by_id[row_id] = row
            answers_cache[row_id] = answer
            processed += 1

            debug_rows.append(
                {
                    "id": row_id,
                    "backend": args.backend,
                    "n_articles": str(len(refs)),
                    "answer_len": str(len(answer)),
                    "question": str(row.get("question", ""))[:120],
                    "answer_preview": answer[:200],
                }
            )

            if processed % 10 == 0:
                print(f"generated {processed}/{len(pending_ids)} answers", flush=True)
                if args.resume:
                    write_submission(list(by_id.values()), output_path)

    out_rows = list(by_id.values())
    write_submission(out_rows, output_path)
    if debug_rows:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(debug_rows[0].keys()))
            writer.writeheader()
            writer.writerows(debug_rows)

    print(
        f"DONE -> {output_path} ({len(out_rows)} rows, "
        f"generated={processed}, backend={args.backend})",
        flush=True,
    )


if __name__ == "__main__":
    main()
