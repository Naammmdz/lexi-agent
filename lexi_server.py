"""FastAPI server for Lexi-style web UI (ported from LawShield frontend-landing)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_UI_DIR = Path(__file__).resolve().parent / "ui" / "lexi"


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    session_id: str = "default"


def create_lexi_app(chatbot: Any) -> FastAPI:
    app = FastAPI(title="Lexi Legal Agent", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        return FileResponse(_UI_DIR / "index.html")

    @app.get("/api/status")
    async def status():
        return chatbot.get_system_status()

    @app.get("/api/samples")
    async def samples():
        return chatbot.get_sample_questions()

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        history, docs_info, processing_status, sources, meta = chatbot.process_message(
            req.message,
            list(req.history),
            session_id=req.session_id,
        )
        return {
            "history": history,
            "sources_markdown": docs_info,
            "sources": sources,
            "status": processing_status,
            "meta": meta,
        }

    app.mount("/static", StaticFiles(directory=_UI_DIR), name="static")
    return app


def run_lexi_server(chatbot: Any, host: Optional[str] = None, port: Optional[int] = None) -> None:
    import uvicorn

    host = host or os.getenv("LEXI_HOST", "127.0.0.1")
    port = int(port or os.getenv("LEXI_PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    app = create_lexi_app(chatbot)
    print(f"⚖️ Lexi UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
