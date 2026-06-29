#!/usr/bin/env python3
"""Self-hosted Vietnamese bi-encoder embedding API."""

from __future__ import annotations

import argparse
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv(
    "EMBED_MODEL", "bkai-foundation-models/vietnamese-bi-encoder"
)
DEFAULT_BATCH = int(os.getenv("EMBED_BATCH_SIZE", "256"))


class EmbedRequest(BaseModel):
    model: str = DEFAULT_MODEL
    input: list[str]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_app(model_name: str, batch_size: int) -> FastAPI:
    device = pick_device()
    print(f"Loading {model_name} on {device} (batch_size={batch_size})", flush=True)
    model = SentenceTransformer(model_name, device=device)

    app = FastAPI(title="Vietnamese Embedder API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name, "device": device}

    @app.post("/v1/embed")
    def embed(req: EmbedRequest) -> dict:
        texts = [t for t in (req.input or []) if t]
        if not texts:
            return {"model": req.model, "embeddings": [], "dimension": 0}

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        embeddings = [
            v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors
        ]
        dim = len(embeddings[0]) if embeddings else 0
        return {"model": req.model, "embeddings": embeddings, "dimension": dim}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()

    app = create_app(args.model, args.batch_size)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
