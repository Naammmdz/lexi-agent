"""Rule-based query decomposition for multi-hop / long legal questions."""

from __future__ import annotations

import re

from config import Config

SPLIT_PATTERNS = (
    r"\s+và\s+(?=(?:nếu|khi|việc|trường hợp|có|được|phải|thì|liệu|nhưng|tuy))",
    r"\s+đồng thời\s+",
    r"\s+ngoài ra\s+",
    r"\s+;\s+",
)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def should_decompose(query: str) -> bool:
    if not Config.ENABLE_QUERY_DECOMPOSITION:
        return False
    text = (query or "").strip()
    if not text:
        return False
    wc = word_count(text)
    if text.count("?") >= 2 or ";" in text:
        return True
    if wc >= Config.QUERY_DECOMPOSE_MIN_WORDS:
        return True
    return False


def _clean_part(part: str, *, add_qmark: bool = False) -> str:
    text = re.sub(r"\s+", " ", part).strip(" ,;")
    if not text:
        return ""
    if add_qmark and not text.endswith("?"):
        text = f"{text}?"
    return text


def decompose_query(query: str) -> list[str]:
    """Return 1..N sub-queries; always non-empty."""
    text = (query or "").strip()
    if not text:
        return [""]

    parts: list[str] = []

    if text.count("?") >= 2:
        chunks = re.split(r"\?+", text)
        parts = [_clean_part(c, add_qmark=True) for c in chunks if c.strip()]
    elif ";" in text:
        parts = [_clean_part(c) for c in text.split(";") if c.strip()]
    else:
        for pattern in SPLIT_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                split = re.split(pattern, text, maxsplit=1, flags=re.IGNORECASE)
                if len(split) == 2 and split[0].strip() and split[1].strip():
                    parts = [_clean_part(split[0]), _clean_part(split[1])]
                    break

    parts = [p for p in parts if p and word_count(p) >= 4]
    if not parts:
        return [text]

    max_q = max(1, Config.QUERY_DECOMPOSE_MAX_SUBQUERIES)
    if len(parts) > max_q:
        head = parts[: max_q - 1]
        tail = " ".join(parts[max_q - 1 :])
        parts = head + [_clean_part(tail)]

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)

    if len(unique) <= 1:
        return [text]
    return unique
