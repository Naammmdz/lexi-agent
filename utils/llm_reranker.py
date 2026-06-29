"""LLM-based reranking over a fixed hybrid retrieval candidate pool."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from utils.submission_formatter import article_label, canonical_law_id, format_law_title, get_mapping_title


SYSTEM_PROMPT = """Bạn là chuyên gia tra cứu văn bản pháp luật Việt Nam.
Nhiệm vụ: xếp hạng các điều luật ứng viên theo mức độ liên quan trực tiếp tới câu hỏi.
Chỉ chọn từ danh sách ứng viên đã cho. Không bịa văn bản hoặc số điều mới.
Trả lời DUY NHẤT bằng JSON hợp lệ, không markdown, dạng:
{"ranking":[1,3,2],"reason":"ngắn gọn"}
Trong đó ranking là chỉ số ứng viên (1-based), sắp xếp từ liên quan nhất tới ít liên quan hơn."""


@dataclass
class LLMRerankResult:
    docs: list[dict[str, Any]]
    ranking: list[int]
    raw_response: str
    backend: str
    latency_sec: float
    parse_ok: bool


def _truncate(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def candidate_line(doc: dict[str, Any], index: int, mapping: dict[str, str]) -> str:
    metadata = doc.get("metadata", {})
    law_id = canonical_law_id(metadata.get("law_id", ""))
    title = format_law_title(law_id, get_mapping_title(mapping, law_id))
    article = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
    snippet = _truncate(doc.get("content") or metadata.get("text") or "")
    return f"{index}. {law_id}|{title}|{article}\n   {snippet}"


def build_prompt(question: str, candidates: Sequence[dict[str, Any]], mapping: dict[str, str]) -> str:
    lines = [f"Câu hỏi: {question.strip()}", "", "Ứng viên:"]
    for idx, doc in enumerate(candidates, 1):
        lines.append(candidate_line(doc, idx, mapping))
    lines.append("")
    lines.append("Trả JSON ranking 1-based cho tất cả ứng viên liên quan, ưu tiên điều trả lời trực tiếp câu hỏi.")
    return "\n".join(lines)


def parse_ranking_response(text: str, n_candidates: int) -> tuple[list[int], bool]:
    raw = str(text or "").strip()
    if not raw:
        return list(range(1, n_candidates + 1)), False

    payload = raw
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S)
    if fence:
        payload = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", raw, flags=re.S)
        if brace:
            payload = brace.group(0)

    try:
        data = json.loads(payload)
        ranking = [int(x) for x in data.get("ranking", [])]
    except Exception:
        nums = [int(x) for x in re.findall(r"\b(\d+)\b", raw)]
        ranking = []
        for num in nums:
            if 1 <= num <= n_candidates and num not in ranking:
                ranking.append(num)

    cleaned: list[int] = []
    for num in ranking:
        if 1 <= num <= n_candidates and num not in cleaned:
            cleaned.append(num)
    for num in range(1, n_candidates + 1):
        if num not in cleaned:
            cleaned.append(num)
    return cleaned, bool(ranking)


def reorder_candidates(candidates: Sequence[dict[str, Any]], ranking: list[int]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for num in ranking:
        doc = candidates[num - 1]
        enriched = dict(doc)
        enriched["llm_rank"] = num
        ordered.append(enriched)
    return ordered


class LLMReranker:
    def __init__(
        self,
        backend: str = "local",
        model: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 384,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._llm = None
        if backend == "local":
            self._init_local()
        elif backend == "gemini":
            self._init_gemini()
        elif backend not in {"mock", "passthrough"}:
            raise ValueError(f"Unknown LLM backend: {backend}")

    def _init_local(self) -> None:
        from utils.local_legal_llm import competition_model_profile, get_local_legal_llm

        profile = competition_model_profile(self.model)
        if not profile["btc_compliant"]:
            raise RuntimeError(f"Model not BTC-compliant: {profile}")
        self._llm = get_local_legal_llm(model_name=self.model, max_new_tokens=self.max_new_tokens)
        self.model = profile["model_id"]

    def _init_gemini(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for --llm-backend gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = self.model or os.getenv("LLM_RERANK_MODEL", "gemini-2.0-flash")
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self.temperature,
        )
        self.model = model

    def rerank(
        self,
        question: str,
        candidates: Sequence[dict[str, Any]],
        mapping: dict[str, str],
        top_k: int | None = None,
    ) -> LLMRerankResult:
        pool = list(candidates)
        if not pool:
            return LLMRerankResult([], [], "", self.backend, 0.0, True)

        start = time.time()
        if self.backend == "mock":
            ranking = list(range(1, len(pool) + 1))
            raw = json.dumps({"ranking": ranking, "reason": "mock"})
            parse_ok = True
        elif self.backend == "passthrough":
            ranking = list(range(1, len(pool) + 1))
            raw = ""
            parse_ok = True
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = build_prompt(question, pool, mapping)
            response = self._llm.invoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            raw = getattr(response, "content", str(response))
            ranking, parse_ok = parse_ranking_response(raw, len(pool))

        ordered = reorder_candidates(pool, ranking)
        if top_k is not None:
            ordered = ordered[:top_k]
        return LLMRerankResult(
            docs=ordered,
            ranking=ranking,
            raw_response=raw,
            backend=self.backend,
            latency_sec=time.time() - start,
            parse_ok=parse_ok,
        )
