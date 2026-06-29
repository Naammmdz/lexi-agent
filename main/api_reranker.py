"""Remote reranker via FPT Cloud (OpenAI-compatible /v1/rerank)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from config import Config


class ApiDocumentReranker:
    """Drop-in reranker using bge-reranker-v2-m3 (or similar) over HTTP."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("FPT_RERANK_API_KEY", "")
        self.base_url = (base_url or os.getenv("FPT_RERANK_BASE_URL", "https://mkp-api.fptcloud.jp")).rstrip("/")
        self.model_name = model_name or os.getenv("FPT_RERANK_MODEL", "bge-reranker-v2-m3")
        self.timeout = timeout
        # Self-hosted endpoint may omit API key.
        self.model = self if self.base_url else None

    def _doc_text(self, doc: dict[str, Any], max_chars: int = 3000) -> str:
        content = doc.get("content", "") or ""
        title = doc.get("title", "") or ""
        text = f"{title}. {content}" if title else content
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def _call_api(self, query: str, documents: list[str]) -> list[float]:
        if not self.model or not documents:
            return [0.0] * len(documents)

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/rerank",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        scores = [0.0] * len(documents)
        for item in data.get("results", []):
            idx = int(item.get("index", -1))
            if 0 <= idx < len(scores):
                scores[idx] = float(item.get("relevance_score", 0.0))
        return scores

    def rerank_documents(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.model or not documents:
            return documents

        try:
            texts = [self._doc_text(doc) for doc in documents]
            scores = self._call_api(query, texts)

            reranked_docs: list[dict[str, Any]] = []
            for doc, score in zip(documents, scores):
                reranked_doc = doc.copy()
                reranked_doc["reranker_score"] = float(score)
                reranked_doc["original_score"] = doc.get("score", 0.0)
                reranked_doc["score"] = float(score)
                if "retrieval_method" in reranked_doc:
                    reranked_doc["retrieval_method"] += "_api_reranked"
                else:
                    reranked_doc["retrieval_method"] = "api_reranked"
                reranked_docs.append(reranked_doc)

            reranked_docs.sort(key=lambda x: x["reranker_score"], reverse=True)
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            return reranked_docs
        except Exception as e:
            print(f"API rerank error: {e}")
            return documents

    def rerank_with_fusion(
        self,
        query: str,
        documents: list[dict[str, Any]],
        alpha: float = 0.7,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.model or not documents:
            return documents

        try:
            reranked_docs = self.rerank_documents(query, documents, top_k=None)
            if not reranked_docs:
                return documents

            original_scores = [doc.get("original_score", 0.0) for doc in reranked_docs]
            max_orig = max(original_scores) if original_scores else 0.0
            original_scores_norm = [s / max_orig if max_orig > 0 else 0.0 for s in original_scores]

            reranker_scores = [doc.get("reranker_score", 0.0) for doc in reranked_docs]
            min_r, max_r = min(reranker_scores), max(reranker_scores)
            if max_r > min_r:
                reranker_scores_norm = [(s - min_r) / (max_r - min_r) for s in reranker_scores]
            else:
                reranker_scores_norm = [0.5] * len(reranker_scores)

            for i, doc in enumerate(reranked_docs):
                fused = alpha * reranker_scores_norm[i] + (1 - alpha) * original_scores_norm[i]
                doc["fused_score"] = fused
                doc["score"] = fused

            reranked_docs.sort(key=lambda x: x["fused_score"], reverse=True)
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            return reranked_docs
        except Exception as e:
            print(f"API fusion rerank error: {e}")
            return documents

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "model_type": "api-rerank",
            "base_url": self.base_url,
        }
