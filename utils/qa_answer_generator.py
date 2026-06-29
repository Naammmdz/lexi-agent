"""Grounded QA answer generation for BTC promote submissions."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from utils.submission_formatter import canonical_law_id

DISCLAIMER = (
    "Lưu ý: Đây là tư vấn pháp lý sơ bộ dựa trên căn cứ được trích dẫn; "
    "trường hợp cụ thể nên được đối chiếu thêm văn bản hiện hành và tham vấn chuyên gia."
)

ARTICLE_RE = re.compile(r"điều\s+([0-9]+[a-z]?)", re.IGNORECASE)
LAW_ID_RE = re.compile(r"\b\d{2,3}/\d{4}/[A-ZĐ]+(?:-[A-Z]+)?\b")
GARBAGE_RE = re.compile(r"(?:\b(?:Có|Không|0)\b\s*){8,}")
META_EN_RE = re.compile(
    r"\b(?:wait|however|the user|legal references|instructions say|paraphrase|provided legal)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "công", "ty", "thì", "được", "như", "nào", "gì", "khi", "các", "theo", "phải",
    "có", "không", "nếu", "và", "cho", "về", "trong", "này", "để", "sẽ", "hay", "hoặc",
    "muốn", "biết", "làm", "sao", "cần", "với", "từ", "đến", "của", "một", "những",
}


def parse_article_ref(ref: str) -> tuple[str, str, str]:
    parts = str(ref).split("|")
    if len(parts) >= 3:
        return parts[0].strip(), parts[1].strip(), parts[-1].strip()
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), ""
    return str(ref).strip(), "", ""


def article_id_from_label(label: str) -> str:
    match = ARTICLE_RE.search(label or "")
    return match.group(1).lower() if match else ""


def build_corpus_lookup(corpus_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for doc in corpus:
        law_id = canonical_law_id(doc.get("law_id", ""))
        doc_title = str(doc.get("title", "") or "")
        for article in doc.get("articles", []):
            article_id = str(article.get("article_id", "")).strip()
            label = str(article.get("title", "") or f"Điều {article_id}").strip()
            text = str(article.get("text", "") or article.get("content", "") or "").strip()
            lookup[(law_id, article_id)] = {
                "law_id": law_id,
                "article_id": article_id,
                "label": label,
                "doc_title": doc_title,
                "text": text,
            }
            aid = article_id_from_label(label)
            if aid and (law_id, aid) not in lookup:
                lookup[(law_id, aid)] = lookup[(law_id, article_id)]
    return lookup


def resolve_article_text(
    ref: str,
    lookup: dict[tuple[str, str], dict[str, str]],
) -> dict[str, str] | None:
    law_id, doc_title, label = parse_article_ref(ref)
    law_id = canonical_law_id(law_id)
    article_id = article_id_from_label(label)
    if not law_id:
        return None
    item = lookup.get((law_id, article_id)) if article_id else None
    if item:
        return {**item, "doc_title": doc_title or item.get("doc_title", "")}
    for (lid, _), candidate in lookup.items():
        if lid == law_id and label and label.lower() in candidate.get("label", "").lower():
            return {**candidate, "doc_title": doc_title or candidate.get("doc_title", "")}
    return {
        "law_id": law_id,
        "article_id": article_id,
        "label": label,
        "doc_title": doc_title,
        "text": "",
    }


def normalize_corpus_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "").strip())
    text = re.sub(r"([.!?;:])(?=[A-Za-z0-9])", r"\1 ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])(\d)", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_text(text: str, max_chars: int) -> str:
    text = normalize_corpus_text(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.strip() + "…"


def split_sentences(text: str) -> list[str]:
    text = normalize_corpus_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+", text)
    cleaned: list[str] = []
    for part in parts:
        part = re.sub(r"^\d+\.\s*", "", part.strip(" -"))
        part = re.sub(r"^[a-zđ]\)\s*", "", part, flags=re.IGNORECASE)
        if len(part) < 25:
            continue
        if re.match(r"^[^.]{0,80}\s+\d+\.\s*$", part):
            continue
        if part.endswith(" 1.") or part.endswith(" 2."):
            continue
        cleaned.append(part)
    return cleaned


def content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Zà-ỹ0-9]+", text.lower())
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def query_phrases(question: str) -> list[str]:
    words = [w for w in re.findall(r"[a-zà-ỹ0-9]+", question.lower()) if len(w) >= 3 and w not in STOPWORDS]
    phrases: list[str] = []
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i : i + n]))
    return phrases


def score_sentence(question: str, sentence: str) -> float:
    q_tokens = content_tokens(question)
    s_tokens = content_tokens(sentence)
    if not q_tokens or not s_tokens:
        return 0.0
    overlap = len(q_tokens & s_tokens) / len(q_tokens)
    s_lower = sentence.lower()
    q_lower = question.lower()
    bonus = 0.0
    for phrase in query_phrases(question):
        if phrase in s_lower:
            bonus += 0.2
    for kw in ("phạt", "xử phạt", "vi phạm", "điều kiện", "thủ tục", "thời hạn", "mức", "khắc phục", "giữ bản chính", "văn bằng"):
        if kw in q_lower and kw in s_lower:
            bonus += 0.12
    if re.search(r"điều\s+\d+", s_lower):
        bonus += 0.03
    if re.search(r"\b000\s*đồng\b", s_lower):
        bonus -= 0.6
    if re.search(r"\d{1,3}(?:\.\d{3})+\s*đồng|\d+\s*triệu\s*đồng|\d+\s*tỷ\s*đồng", s_lower):
        bonus += 0.3
    if len(sentence) < 40:
        bonus -= 0.15
    return overlap + bonus


def select_relevant_sentences(question: str, text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return normalize_corpus_text(text)[:800]
    ranked = sorted(
        ((score_sentence(question, s), s) for s in sentences),
        key=lambda x: -x[0],
    )
    chosen: list[str] = []
    seen: set[str] = set()
    for score, sentence in ranked:
        if score <= 0 and chosen:
            continue
        key = sentence[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        chosen = [sentences[0]]
    return " ".join(chosen)


def block_relevance_score(question: str, block: dict[str, str]) -> float:
    sentences = split_sentences(block.get("text", ""))
    if not sentences:
        return 0.0
    return max(score_sentence(question, s) for s in sentences)


def build_context_blocks(
    article_refs: list[str],
    lookup: dict[tuple[str, str], dict[str, str]],
    max_articles: int,
    max_chars_per_article: int,
    question: str = "",
) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref in article_refs[:max_articles]:
        item = resolve_article_text(ref, lookup)
        if not item:
            continue
        key = (item["law_id"], item.get("article_id", ""))
        if key in seen:
            continue
        seen.add(key)
        raw = item.get("text", "")
        body = select_relevant_sentences(question, raw) if question else raw
        blocks.append(
            {
                "ref": ref,
                "law_id": item["law_id"],
                "article_id": item.get("article_id", ""),
                "label": item.get("label", ""),
                "doc_title": item.get("doc_title", ""),
                "text": trim_text(body, max_chars_per_article),
                "relevance": block_relevance_score(question, {"text": raw}),
            }
        )
    if question:
        blocks.sort(key=lambda b: -float(b.get("relevance", 0.0)))
    return blocks


def citation_prefix(blocks: list[dict[str, str]]) -> str:
    cites: list[str] = []
    for block in blocks:
        law_id, _, label = parse_article_ref(block.get("ref", ""))
        law_id = law_id or block.get("law_id", "")
        if not label:
            label = block.get("label") or f"Điều {block.get('article_id', '')}"
        if not label.lower().startswith("điều"):
            label = f"Điều {block.get('article_id', '')}".strip()
        cites.append(f"{law_id}|{label}")
    return "; ".join(cites)


def allowed_law_ids(blocks: list[dict[str, str]]) -> set[str]:
    ids: set[str] = set()
    for block in blocks:
        ids.add(block["law_id"].lower())
        law_id, _, _ = parse_article_ref(block.get("ref", ""))
        if law_id:
            ids.add(law_id.lower())
    return ids


def build_system_prompt() -> str:
    return (
        "Bạn là Lexi, trợ lý AI pháp lý cho doanh nghiệp vừa và nhỏ (SME) tại Việt Nam. "
        "Giọng văn thực dụng, dễ hiểu cho chủ DN / kế toán / nhân sự.\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Chỉ dùng thông tin trong phần CĂN CỨ PHÁP LUẬT.\n"
        "2. Câu đầu trả lời TRỰC TIẾP câu hỏi (có/không, điều kiện, mức phạt, thủ tục…).\n"
        "3. Tổng hợp bằng lời văn tư vấn; KHÔNG chép nguyên khoản a/b/c hay liệt kê dài.\n"
        "4. Nêu đủ điều kiện, thủ tục, mức phạt hoặc hậu quả nếu căn cứ có.\n"
        "5. Chỉ nhắc các Điều đã cho trong căn cứ; không trích dẫn Điều/văn bản khác.\n"
        "6. Viết 4–8 câu hoàn chỉnh, đủ ý; kết thúc bằng dấu chấm.\n"
        "7. Không dùng tiêu đề Phân tích/Kết luận; không lặp cụm Theo Điều… nhiều lần.\n"
        "8. Nếu căn cứ không nêu số tiền phạt cụ thể, ghi mức phạt theo quy định pháp luật, không ghi 000 đồng."
    )


def build_user_prompt(question: str, blocks: list[dict[str, str]], compact: bool = False) -> str:
    use_blocks = blocks
    parts = ["CĂN CỨ PHÁP LUẬT (chỉ dùng nội dung này):"]
    for i, block in enumerate(use_blocks, 1):
        label = format_article_label(block) if compact else (block.get("label") or f"Điều {block.get('article_id', '')}")
        header = f"[{i}] {block['law_id']} — {label}"
        body = block.get("text") or "(Không có nội dung chi tiết trong corpus.)"
        parts.append(f"{header}\n{body}")
    parts.extend(
        [
            "",
            f"CÂU HỎI: {question}",
            "",
            "Hãy viết câu trả lời tư vấn pháp lý (chỉ phần nội dung, không ghi Căn cứ/Trả lời/Lưu ý):",
        ]
    )
    return "\n".join(parts)


def format_article_label(block: dict[str, str]) -> str:
    label = str(block.get("label", "") or "").strip()
    aid = str(block.get("article_id", "") or "").strip() or article_id_from_label(label)
    if aid:
        return f"Điều {aid}"
    if label.lower().startswith("điều"):
        return label.split(".", 1)[0].strip()
    return label or "căn cứ pháp luật"


def synthesize_professional_extractive(question: str, blocks: list[dict[str, str]]) -> str:
    q_lower = question.lower()
    ranked_blocks = sorted(
        blocks,
        key=lambda b: -float(b.get("relevance", block_relevance_score(question, b))),
    )
    best_rel = float(ranked_blocks[0].get("relevance", 0.0)) if ranked_blocks else 0.0
    active_blocks = [
        b
        for b in ranked_blocks
        if float(b.get("relevance", 0.0)) >= max(0.12, best_rel * 0.45)
    ]
    if not active_blocks:
        active_blocks = ranked_blocks[:2] or ranked_blocks

    parts: list[str] = []
    for block in active_blocks[:2]:
        label = format_article_label(block)
        law_id = block.get("law_id", "")
        sentences = split_sentences(block.get("text", ""))
        if not sentences:
            continue
        ranked = sorted(
            ((score_sentence(question, s), s) for s in sentences),
            key=lambda x: -x[0],
        )
        best = [s for sc, s in ranked if sc > 0.05][:2] or [ranked[0][1]]
        lead = best[0].strip()
        if lead[0].isupper():
            lead = lead[0] + lead[1:]
        parts.append(f"Theo {label} của {law_id}, {lead}")
        if any(w in q_lower for w in ("phạt", "xử lý", "vi phạm", "bị", "khắc phục")) and len(best) > 1:
            parts.append(best[1])

    body = " ".join(parts[:3])
    body = trim_text(body, 1200)
    body = re.sub(r"\s+Theo\s*$", "", body, flags=re.IGNORECASE).strip()
    if not body and blocks:
        block = active_blocks[0]
        body = (
            f"Theo {block.get('label', 'căn cứ pháp luật')} của {block.get('law_id', '')}, "
            "cần đối chiếu quyền, nghĩa vụ, điều kiện và thủ tục áp dụng cho tình huống trong câu hỏi."
        )
    return body


def generate_extractive_answer(question: str, blocks: list[dict[str, str]]) -> str:
    return synthesize_professional_extractive(question, blocks)


def generate_local_answer(
    question: str,
    blocks: list[dict[str, str]],
    model_name: str | None = None,
    max_new_tokens: int = 384,
) -> str:
    from utils.local_legal_llm import get_local_legal_llm

    llm = get_local_legal_llm(model_name=model_name, max_new_tokens=max_new_tokens)
    messages = [
        SystemMessage(content=build_system_prompt()),
        HumanMessage(content=build_user_prompt(question, blocks)),
    ]
    return llm.invoke_plain(messages).strip()


THINK_OPEN = "\x3cthink\x3e"
THINK_CLOSE = "\x3c/think\x3e"


def strip_r1_think_tags(text: str) -> str:
    text = str(text or "")
    if THINK_CLOSE.lower() in text.lower():
        text = re.split(re.escape(THINK_CLOSE), text, flags=re.IGNORECASE)[-1]
    text = re.sub(
        re.escape(THINK_OPEN) + r".*?" + re.escape(THINK_CLOSE),
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if text.lower().startswith(THINK_OPEN):
        text = text[len(THINK_OPEN) :]
    return text.strip()


def extract_ollama_text(data: dict[str, Any]) -> str:
    msg = data.get("message") or {}
    content = strip_r1_think_tags(str(msg.get("content", "") or data.get("response", "") or ""))
    thinking = strip_r1_think_tags(str(msg.get("thinking", "") or ""))
    if content and not looks_like_meta_reasoning(content):
        return content
    if thinking and not looks_like_meta_reasoning(thinking):
        for para in reversed(re.split(r"\n\s*\n", thinking)):
            para = para.strip()
            if len(para) >= 80 and not looks_like_meta_reasoning(para):
                return para
    return content


def looks_like_meta_reasoning(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True
    if META_EN_RE.search(text):
        return True
    ascii_words = re.findall(r"[A-Za-z]{3,}", text)
    vi_words = re.findall(r"[à-ỹÀ-Ỹ]{2,}", text)
    if len(ascii_words) >= 8 and len(ascii_words) > len(vi_words):
        return True
    if text.lower().startswith("okay,") or text.lower().startswith("the "):
        return True
    return False


def looks_truncated(body: str) -> bool:
    body = body.strip()
    if len(body) < 40:
        return True
    if body[-1] in ".!?":
        if re.search(r"\b(?:tri|từ|đối|và|hoặc|theo)$", body, re.IGNORECASE):
            return True
        return False
    tail = body.rsplit(maxsplit=1)[-1] if body.split() else ""
    if tail.lower() in {"theo", "và", "hoặc", "đối", "với", "từ", "tri"}:
        return True
    return len(tail) <= 3 or (bool(re.search(r"[a-zà-ỹ]$", tail, re.IGNORECASE)) and len(tail) < 8)


def generate_ollama_answer(
    question: str,
    blocks: list[dict[str, str]],
    model: str | None = None,
    max_new_tokens: int = 1200,
    compact: bool = False,
) -> str:
    model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    user_prompt = build_user_prompt(question, blocks, compact=compact)
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": 0.15,
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
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return extract_ollama_text(data)


def generate_ollama_batch_answers(
    items: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, str]],
    model_name: str | None = None,
    max_articles: int = 3,
    max_chars_per_article: int = 1200,
    max_new_tokens: int = 1200,
    include_disclaimer: bool = True,
    workers: int = 4,
) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor

    def _one(item: dict[str, Any]) -> str:
        question = str(item["question"])
        refs = list(item.get("article_refs") or [])
        blocks = build_context_blocks(
            refs, lookup, max_articles, max_chars_per_article, question=question
        )
        if not blocks:
            return (
                "Chưa xác định được căn cứ pháp luật phù hợp từ dữ liệu được cung cấp. "
                + (DISCLAIMER if include_disclaimer else "")
            )
        try:
            raw = generate_ollama_answer(
                question, blocks, model_name, max_new_tokens, compact=True
            )
            if looks_like_meta_reasoning(raw) or looks_truncated(raw):
                body = synthesize_professional_extractive(question, blocks)
            else:
                body = polish_answer_body(raw, blocks, question)
            if (
                not body.strip()
                or looks_like_meta_reasoning(body)
                or answer_quality_score(question, body, blocks) < 0.12
            ):
                body = synthesize_professional_extractive(question, blocks)
                body = polish_answer_body(body, blocks, question)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"ollama failed: {exc}", flush=True)
            body = synthesize_professional_extractive(question, blocks)
        return finalize_answer(body, blocks, include_disclaimer, refs, lookup)

    workers = max(1, min(workers, len(items) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, items))


def mentions_citations(answer: str, blocks: list[dict[str, str]]) -> bool:
    lower = answer.lower()
    for block in blocks:
        label = block.get("label", "")
        article_id = block.get("article_id") or article_id_from_label(label)
        if article_id and re.search(rf"điều\s+{re.escape(article_id)}\b", lower):
            return True
    return False


def strip_generation_garbage(text: str) -> str:
    if GARBAGE_RE.search(text[-250:]):
        text = GARBAGE_RE.split(text)[0]
    text = re.sub(r"\s+(?:Có Không){2,}.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:Có Có){5,}.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:0 Có){2,}.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+Không(?:\s+Không){5,}.*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def strip_meta_prefixes(body: str) -> str:
    body = re.sub(r"^(?:Căn cứ pháp luật|Trả lời)\s*:\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^Lưu ý\s*:\s*.*$", "", body, flags=re.IGNORECASE)
    return body.strip()


def ensure_complete_sentences(body: str) -> str:
    body = body.strip()
    if not body:
        return body
    if body[-1] in ".!?":
        return body
    if re.search(r"\s[a-zà-ỹ]{1,3}$", body, re.IGNORECASE):
        for end in (". ", "; ", "! ", "? ", ", "):
            idx = body.rfind(end)
            if idx > 60:
                return body[: idx + 1].strip()
    for end in (". ", "; ", "! ", "? "):
        idx = body.rfind(end)
        if idx > 80:
            return body[: idx + 1].strip()
    if not body.endswith("."):
        body = body.rstrip(" ,;:") + "."
    return body


def remove_extra_law_mentions(body: str, blocks: list[dict[str, str]]) -> str:
    allowed = allowed_law_ids(blocks)
    for law_id in LAW_ID_RE.findall(body):
        if law_id.lower() not in allowed:
            body = body.replace(law_id, "căn cứ pháp luật được nêu")
    return body


def allowed_article_ids(blocks: list[dict[str, str]]) -> set[str]:
    ids: set[str] = set()
    for block in blocks:
        aid = str(block.get("article_id", "") or article_id_from_label(block.get("label", ""))).strip()
        if aid:
            ids.add(aid.lower())
    return ids


def keep_relevant_theo_paragraphs(body: str, blocks: list[dict[str, str]]) -> str:
    allowed = allowed_article_ids(blocks)
    if not allowed:
        return body
    parts = re.split(r"(?=(?:(?:^)|(?:\.\s+))Theo\s+Điều\s+\d)", body, flags=re.IGNORECASE)
    kept: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        match = re.search(r"điều\s+(\d+[a-z]?)", chunk, re.IGNORECASE)
        if match and match.group(1).lower() not in allowed:
            continue
        kept.append(chunk)
    if not kept:
        return body
    merged = " ".join(kept)
    return re.sub(r"\s+", " ", merged).strip()


def fix_corrupted_penalty_amounts(body: str, question: str) -> str:
    if re.search(r"\b000\s*đồng\b", body, re.IGNORECASE):
        body = re.sub(
            r"\b\d{3}\s*đồng\b",
            "mức phạt theo quy định của pháp luật",
            body,
            flags=re.IGNORECASE,
        )
    if any(w in question.lower() for w in ("phạt", "xử lý", "vi phạm", "bị")):
        body = re.sub(
            r"(?:^|[\s;])\d{3}\s*đồng(?=\s+(?:đối với|khi|với))",
            " mức phạt theo quy định",
            body,
            flags=re.IGNORECASE,
        )
    return body.strip()


def dedupe_theo_clauses(body: str) -> str:
    seen: set[str] = set()
    parts = re.split(r"(?=(?:^|\.\s+)Theo\s+Điều\s+\d)", body, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        key = chunk[:100].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return " ".join(out).strip() if out else body


def answer_quality_score(question: str, body: str, blocks: list[dict[str, str]]) -> float:
    q_tokens = content_tokens(question)
    b_tokens = content_tokens(body)
    if not q_tokens or not b_tokens:
        return 0.0
    overlap = len(q_tokens & b_tokens) / len(q_tokens)
    score = overlap
    if len(body) < 120:
        score -= 0.25
    if len(body) > 1200:
        score -= 0.1
    if GARBAGE_RE.search(body):
        score -= 1.0
    if re.search(r"\b[abcđ]\)\s", body) and body.count(";") >= 2:
        score -= 0.2
    if not mentions_citations(body, blocks):
        score -= 0.1
    allowed = allowed_law_ids(blocks)
    extra = [lid for lid in LAW_ID_RE.findall(body) if lid.lower() not in allowed]
    if extra:
        score -= 0.15 * min(len(extra), 3)
    return score


def polish_answer_body(body: str, blocks: list[dict[str, str]], question: str) -> str:
    body = strip_meta_prefixes(body)
    body = strip_generation_garbage(body)
    body = normalize_corpus_text(body)
    body = keep_relevant_theo_paragraphs(body, blocks)
    body = dedupe_theo_clauses(body)
    body = remove_extra_law_mentions(body, blocks)
    body = fix_corrupted_penalty_amounts(body, question)
    body = re.sub(r"\s+\d+\.\s+[A-ZÀ-Ỹ][^.]{10,}$", "", body)
    body = ensure_complete_sentences(body)
    body = re.sub(r"\s+Theo\s*$", "", body, flags=re.IGNORECASE).strip()
    if body and body[-1] not in ".!?":
        body += "."
    return body.strip()


def finalize_answer(
    body: str,
    blocks: list[dict[str, str]],
    include_disclaimer: bool,
    article_refs: list[str] | None = None,
    lookup: dict[tuple[str, str], dict[str, str]] | None = None,
) -> str:
    body = polish_answer_body(body, blocks, "")
    if not body:
        body = generate_extractive_answer("", blocks)
    if not mentions_citations(body, blocks) and blocks:
        _, _, label = parse_article_ref(blocks[0].get("ref", ""))
        label = label or blocks[0].get("label", "căn cứ pháp luật")
        lead = f"Theo {label} của {blocks[0].get('law_id', '')}, "
        body = lead + body[0].lower() + body[1:] if body else lead.strip()
    if article_refs and lookup is not None:
        prefix_blocks = build_context_blocks(article_refs, lookup, len(article_refs), 100, question="")
        prefix = citation_prefix(prefix_blocks)
    else:
        prefix = citation_prefix(blocks)
    answer = f"Căn cứ pháp luật: {prefix}. Trả lời: {body}"
    if include_disclaimer:
        answer = f"{answer} {DISCLAIMER}"
    return answer


def generate_grounded_answer(
    question: str,
    article_refs: list[str],
    lookup: dict[tuple[str, str], dict[str, str]],
    backend: str = "extractive",
    max_articles: int = 3,
    max_chars_per_article: int = 900,
    max_new_tokens: int = 384,
    model_name: str | None = None,
    include_disclaimer: bool = True,
) -> str:
    blocks = build_context_blocks(
        article_refs, lookup, max_articles, max_chars_per_article, question=question
    )
    if not blocks:
        return (
            "Chưa xác định được căn cứ pháp luật phù hợp từ dữ liệu được cung cấp. "
            + (DISCLAIMER if include_disclaimer else "")
        )

    body = ""
    if backend == "local":
        try:
            body = generate_local_answer(question, blocks, model_name, max_new_tokens)
            body = polish_answer_body(body, blocks, question)
            if not body.strip():
                body = synthesize_professional_extractive(question, blocks)
        except Exception as exc:
            print(f"local LLM failed: {exc}", flush=True)
            body = synthesize_professional_extractive(question, blocks)
    elif backend == "ollama":
        try:
            body = generate_ollama_answer(question, blocks, model_name, max_new_tokens)
            body = polish_answer_body(body, blocks, question)
            if not body.strip():
                body = synthesize_professional_extractive(question, blocks)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"ollama failed: {exc}", flush=True)
            body = synthesize_professional_extractive(question, blocks)
    else:
        body = synthesize_professional_extractive(question, blocks)

    return finalize_answer(body, blocks, include_disclaimer, article_refs, lookup)


def generate_vllm_batch_answers(
    items: list[dict[str, Any]],
    lookup: dict[tuple[str, str], dict[str, str]],
    model_name: str | None = None,
    max_articles: int = 2,
    max_chars_per_article: int = 400,
    max_new_tokens: int = 200,
    include_disclaimer: bool = True,
) -> list[str]:
    from utils.vllm_legal_llm import generate_batch, messages_to_prompt, get_vllm_engine

    model_name = model_name or os.getenv(
        "VLLM_MODEL", os.getenv("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
    )
    _, tokenizer = get_vllm_engine(model_name)

    prepared: list[tuple[list[dict[str, str]], str, list[str]]] = []
    prompts: list[str] = []
    for item in items:
        question = str(item["question"])
        refs = list(item.get("article_refs") or [])
        blocks = build_context_blocks(
            refs, lookup, max_articles, max_chars_per_article, question=question
        )
        if not blocks:
            prepared.append(([], "", refs))
            prompts.append("")
            continue
        prompt = messages_to_prompt(
            tokenizer,
            build_system_prompt(),
            build_user_prompt(question, blocks, compact=True),
        )
        prepared.append((blocks, prompt, refs))
        prompts.append(prompt)

    valid_idx = [i for i, p in enumerate(prompts) if p]
    valid_prompts = [prompts[i] for i in valid_idx]
    generated_map: dict[int, str] = {}
    if valid_prompts:
        bodies = generate_batch(
            valid_prompts,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        for idx, body in zip(valid_idx, bodies):
            generated_map[idx] = polish_answer_body(body, prepared[idx][0], items[idx]["question"])

    answers: list[str] = []
    for i, item in enumerate(items):
        blocks, _, refs = prepared[i]
        if not blocks:
            answers.append(
                "Chưa xác định được căn cứ pháp luật phù hợp từ dữ liệu được cung cấp. "
                + (DISCLAIMER if include_disclaimer else "")
            )
            continue
        body = generated_map.get(i, "")
        if not body.strip():
            body = synthesize_professional_extractive(str(item["question"]), blocks)
        answers.append(
            finalize_answer(body, blocks, include_disclaimer, refs, lookup)
        )
    return answers
