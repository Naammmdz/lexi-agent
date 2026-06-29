"""Single Ollama conversation for Lexi SME legal assistant (history + optional RAG)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from utils.qa_answer_generator import extract_ollama_text, format_article_label


def build_unified_sme_system_prompt() -> str:
    return (
        "Bạn là Lexi, trợ lý AI pháp lý cho doanh nghiệp vừa và nhỏ (SME) tại Việt Nam.\n\n"
        "Cách trả lời:\n"
        "- Đọc toàn bộ lịch sử chat; bám ngữ cảnh (vd. user hỏi lại «thật không», «hỏi về SME» phải liên hệ câu trước).\n"
        "- Trả lời tự nhiên như hội thoại thật — KHÔNG dùng template cố định, KHÔNG lặp y hệt cùng một đoạn chào.\n"
        "- Chào hỏi / xã giao: ngắn, thân thiện; giới thiệu Lexi hỗ trợ pháp lý doanh nghiệp (thuế, lao động, hợp đồng, thủ tục…).\n"
        "- Câu hỏi pháp lý: nếu có CĂN CỨ PHÁP LUẬT, chỉ dùng nội dung đó; nêu Điều/luật có trong căn cứ; tư vấn thực dụng cho chủ DN / kế toán / HR.\n"
        "- Chưa có căn cứ điều luật: giải thích khái niệm (vd. SME là gì, DN nhỏ/vừa) trong phạm vi pháp luật VN; không bịa số Điều; gợi hỏi cụ thể hơn.\n"
        "- Câu ngoài pháp luật: nhẹ nhàng chuyển hướng về pháp lý doanh nghiệp; KHÔNG từ chối bằng mẫu «tôi chỉ trả lời pháp luật».\n\n"
        "Quy tắc: tiếng Việt; 2–8 câu; không tiêu đề Phân tích/Kết luận; không disclaimer dài."
    )


def _format_blocks_section(blocks: list[dict[str, str]]) -> str:
    parts = ["CĂN CỨ PHÁP LUẬT (chỉ dùng khi liên quan câu hỏi hiện tại):"]
    for i, block in enumerate(blocks, 1):
        label = format_article_label(block)
        body = block.get("text") or "(Không có nội dung chi tiết.)"
        parts.append(f"[{i}] {block['law_id']} — {label}\n{body}")
    return "\n\n".join(parts)


def _ollama_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.35,
    timeout: int = 120,
) -> str:
    model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    if "deepseek-r1" in model.lower() or (
        "qwen3" in model.lower() and "instruct" not in model.lower()
    ):
        payload["think"] = False
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return extract_ollama_text(data)


def generate_unified_sme_reply(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    blocks: list[dict[str, str]] | None = None,
    extra_context: str = "",
    model: str | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """One LLM call: conversation history + optional retrieved law blocks."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_unified_sme_system_prompt()},
    ]
    for msg in (history or [])[-16:]:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(msg.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content[:2500]})

    user_parts: list[str] = []
    if blocks:
        user_parts.append(_format_blocks_section(blocks))
    extra = (extra_context or "").strip()
    if extra:
        user_parts.append(f"THÔNG TIN BỔ SUNG:\n{extra[:4000]}")
    user_parts.append(f"Câu hỏi hiện tại: {question.strip()}")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    tokens = max_new_tokens or int(os.getenv("CHAT_MAX_NEW_TOKENS", "512"))
    return _ollama_chat(
        messages,
        model=model,
        max_new_tokens=tokens,
        temperature=0.35,
    )
