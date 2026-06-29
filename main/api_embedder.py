"""Remote bi-encoder embeddings over HTTP (self-hosted or compatible API)."""

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np

from config import Config


class ApiEmbeddingEncoder:
    """Drop-in encode() for SentenceTransformer when EMBEDDING_BACKEND=api."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ):
        self.base_url = (
            base_url or os.getenv("EMBED_API_BASE_URL", "http://localhost:8001")
        ).rstrip("/")
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.api_key = api_key or os.getenv("EMBED_API_KEY", "")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {"model": self.model_name, "input": texts}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/embed",
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"embed API returned {len(embeddings)} vectors for {len(texts)} texts"
            )
        return embeddings

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 64,
        convert_to_tensor: bool = False,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = False,
        **_: Any,
    ) -> np.ndarray | list[float]:
        del show_progress_bar, normalize_embeddings

        if isinstance(sentences, str):
            vectors = self._call_embed([sentences])
            result: np.ndarray | list[float] = vectors[0]
        else:
            texts = list(sentences)
            all_vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                all_vectors.extend(self._call_embed(chunk))
            result = np.array(all_vectors, dtype=np.float32)

        if convert_to_tensor:
            import torch

            return torch.tensor(result)
        return result
