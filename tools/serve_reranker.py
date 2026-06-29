#!/usr/bin/env python3
"""Self-hosted Vietnamese reranker API (FPT /v1/rerank compatible)."""

from __future__ import annotations

import argparse
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = os.getenv("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
DEFAULT_MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "768"))
DEFAULT_BATCH = int(os.getenv("RERANKER_BATCH_SIZE", "16"))


class RerankRequest(BaseModel):
    model: str = DEFAULT_MODEL
    query: str
    documents: list[str]
    top_n: int | None = None


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_app(model_name: str, max_length: int, batch_size: int) -> FastAPI:
    device = pick_device()
    print(f"Loading {model_name} on {device} (max_length={max_length})", flush=True)
    model = CrossEncoder(model_name, max_length=max_length, device=device)

    app = FastAPI(title="Vietnamese Reranker API")

    def score_batch(query: str, documents: list[str]) -> list[float]:
        pairs = [[query, doc] for doc in documents]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name, "device": device}

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> dict:
        docs = req.documents or []
        if not docs:
            return {"object": "rerank", "model": req.model, "results": []}

        raw_scores = score_batch(req.query, docs)
        ranked = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)
        top_n = req.top_n or len(docs)
        results = [
            {"index": idx, "relevance_score": score, "document": None}
            for idx, score in ranked[:top_n]
        ]
        return {"object": "rerank", "model": req.model, "results": results}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    app = create_app(args.model, args.max_length, args.batch_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
