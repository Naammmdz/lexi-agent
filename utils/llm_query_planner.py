"""LLM legal keyword planner for retrieval query expansion (BTC <14B via Ollama)."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from config import Config
from utils.retrieval_scoring import fuse_multi_query_rrf

PLANNER_SYSTEM = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: phân tích câu hỏi pháp lý và trích xuất từ khóa tra cứu.
QUY TẮC:
1. Chỉ dùng thông tin có trong câu hỏi; không bịa luật, số điều, văn bản.
2. keywords: 4-10 cụm từ pháp lý tiếng Việt (danh từ/cụm danh từ), không có động từ chung.
3. intent: một trong penalty | condition | procedure | rights | definition | other
4. sub_queries: tối đa 2 câu hỏi phụ ngắn (nếu câu hỏi gốc có nhiều ý); nếu không cần thì []
5. Trả lời DUY NHẤT JSON hợp lệ, không markdown:
{"intent":"...","keywords":["..."],"sub_queries":["..."]}"""


@dataclass
class QueryPlan:
    intent: str = "other"
    keywords: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    raw: str = ""
    parse_ok: bool = False

    def cache_key(self, question: str) -> str:
        return question.strip()


def _extract_json(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _normalize_keywords(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        kw = re.sub(r"\s+", " ", str(item or "").strip().lower())
        if len(kw) < 3 or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out[:10]


def parse_plan_response(raw: str) -> QueryPlan:
    data = _extract_json(raw)
    intent = str(data.get("intent", "other") or "other").strip().lower()
    keywords = _normalize_keywords(data.get("keywords"))
    sub_queries: list[str] = []
    for item in data.get("sub_queries") or []:
        q = re.sub(r"\s+", " ", str(item or "").strip())
        if len(q) >= 12:
            sub_queries.append(q)
    return QueryPlan(
        intent=intent,
        keywords=keywords,
        sub_queries=sub_queries[:2],
        raw=raw,
        parse_ok=bool(keywords or sub_queries),
    )


def build_planned_queries(question: str, plan: QueryPlan) -> list[str]:
    """Return deduped retrieval queries: original + keyword-augmented + sub-queries."""
    base = question.strip()
    queries: list[str] = [base]
    if plan.keywords:
        kw_query = f"{base} {' '.join(plan.keywords[:8])}"
        queries.append(kw_query.strip())
    for sub in plan.sub_queries:
        if sub and sub.lower() != base.lower():
            queries.append(sub)
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(q)
    return unique


def _extract_ollama_planner_text(data: dict[str, Any]) -> str:
    """Pull JSON answer from Ollama chat response (content, thinking, think-tags)."""
    from utils.qa_answer_generator import extract_ollama_text, strip_r1_think_tags

    msg = data.get("message") or {}
    candidates = [
        extract_ollama_text(data),
        strip_r1_think_tags(str(msg.get("thinking", "") or "")),
        strip_r1_think_tags(str(msg.get("content", "") or data.get("response", "") or "")),
    ]
    for text in candidates:
        if not text:
            continue
        if _extract_json(text):
            return text
        match = re.search(
            r'\{[^{}]*"intent"\s*:\s*"[^"]+"[^{}]*"keywords"\s*:\s*\[[^\]]*\][^{}]*\}',
            text,
            flags=re.DOTALL,
        )
        if match:
            return match.group(0)
    return candidates[0] or candidates[1] or ""


def plan_via_ollama(
    question: str,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 512,
) -> QueryPlan:
    model = model or os.getenv("OLLAMA_PLANNER_MODEL", "qwen3-vl:2b")
    base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    user_content = f"Câu hỏi: {question.strip()}\nTrả lời chỉ một dòng JSON hợp lệ, không giải thích."
    if "qwen3" in model.lower():
        user_content = "/no_think\n" + user_content
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.05},
    }
    if "r1" in model.lower() or "qwen3" in model.lower():
        payload["think"] = False
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return QueryPlan(raw=str(exc), parse_ok=False)
    content = _extract_ollama_planner_text(data)
    return parse_plan_response(content)


class LegalQueryPlanner:
    """Cached LLM planner (Ollama)."""

    def __init__(self, model: str | None = None, cache_path: str | None = None) -> None:
        self.model = model
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path and os.path.isfile(cache_path):
            for line in open(cache_path, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    if not row.get("parse_ok"):
                        continue
                    if self.model and row.get("model") and row.get("model") != self.model:
                        continue
                    self._cache[row["question"]] = row

    def _cached_plan(self, key: str) -> QueryPlan | None:
        row = self._cache.get(key)
        if not row or not row.get("parse_ok"):
            return None
        return QueryPlan(
            intent=row.get("intent", "other"),
            keywords=list(row.get("keywords") or []),
            sub_queries=list(row.get("sub_queries") or []),
            raw=str(row.get("raw", "")),
            parse_ok=True,
        )

    def plan(self, question: str) -> QueryPlan:
        key = question.strip()
        cached = self._cached_plan(key)
        if cached:
            return cached
        result = plan_via_ollama(key, model=self.model)
        row = {
            "question": key,
            "intent": result.intent,
            "keywords": result.keywords,
            "sub_queries": result.sub_queries,
            "raw": result.raw,
            "parse_ok": result.parse_ok,
            "model": self.model or os.getenv("OLLAMA_PLANNER_MODEL", "qwen3:14b"),
        }
        self._cache[key] = row
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result

    def plan_batch(self, questions: list[str], workers: int = 8) -> int:
        """Plan uncached questions in parallel (Ollama NUM_PARALLEL should match workers)."""
        pending = [
            q.strip()
            for q in questions
            if q.strip() and self._cached_plan(q.strip()) is None
        ]
        if not pending:
            return 0

        lock = threading.Lock()
        model_name = self.model or os.getenv("OLLAMA_PLANNER_MODEL", "qwen3:14b")
        done = 0

        def _one(question: str) -> None:
            nonlocal done
            result = plan_via_ollama(question, model=self.model)
            row = {
                "question": question,
                "intent": result.intent,
                "keywords": result.keywords,
                "sub_queries": result.sub_queries,
                "raw": result.raw,
                "parse_ok": result.parse_ok,
                "model": model_name,
            }
            with lock:
                self._cache[question] = row
                if self.cache_path:
                    os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1
                if done % 10 == 0 or done == len(pending):
                    print(f"  [planner_cache] {done}/{len(pending)}", flush=True)

        n_workers = max(1, min(workers, len(pending)))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_one, pending))
        return len(pending)


def retrieve_with_plan(
    rag: Any,
    question: str,
    plan: QueryPlan,
    *,
    use_reranking: bool = True,
) -> list[dict[str, Any]]:
    """Hybrid retrieve over planned queries, then rerank with the original question."""
    from utils.retrieval_scoring import (
        apply_law_shortlist_filter,
        apply_two_stage_within_law_rerank,
        apply_within_law_rescore,
    )

    queries = build_planned_queries(question, plan) if plan.parse_ok else [question]
    bm25_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else Config.BM25_TOP_K
    vector_top_k = Config.RERANK_BEFORE_RETRIEVAL_TOP_K if use_reranking else Config.TOP_K_RETRIEVAL

    if len(queries) > 1:
        pools = [
            rag._hybrid_retrieve_pool(sub_q, bm25_top_k, vector_top_k) for sub_q in queries
        ]
        retrieved_docs = fuse_multi_query_rrf(pools, rrf_k=Config.HYBRID_RRF_K)
    else:
        retrieved_docs = rag._hybrid_retrieve_pool(question, bm25_top_k, vector_top_k)

    if not retrieved_docs:
        return []

    if Config.ENABLE_LAW_SHORTLIST and use_reranking:
        retrieved_docs = apply_law_shortlist_filter(
            retrieved_docs,
            top_laws=Config.LAW_SHORTLIST_TOP_K,
            min_docs=Config.LAW_SHORTLIST_MIN_DOCS,
        )

    if use_reranking and rag.reranker:
        if Config.USE_SCORE_FUSION:
            retrieved_docs = rag.reranker.rerank_with_fusion(
                question,
                retrieved_docs,
                alpha=Config.RERANKER_FUSION_ALPHA,
                top_k=Config.RERANKER_TOP_K,
            )
        else:
            retrieved_docs = rag.reranker.rerank_documents(
                question,
                retrieved_docs,
                top_k=Config.RERANKER_TOP_K,
            )
        if Config.ENABLE_WITHIN_LAW_RESCORE:
            retrieved_docs = apply_within_law_rescore(question, retrieved_docs)
        if Config.ENABLE_TWO_STAGE_WITHIN_LAW_RERANK:
            retrieved_docs = apply_two_stage_within_law_rerank(
                rag.reranker,
                question,
                retrieved_docs,
                top_laws=Config.WITHIN_LAW_RERANK_TOP_LAWS,
                within_weight=Config.WITHIN_LAW_RERANK_WEIGHT,
            )

    return retrieved_docs
