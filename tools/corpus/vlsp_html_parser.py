"""Parse VLSP legal-pretrain HTML bodies into Zalo-style article records."""

from __future__ import annotations

import re
from html import unescape
from typing import Any


DIEU_HEAD_RE = re.compile(
    r"(?:^|\n)\s*(Điều\s+(\d+[A-Za-z]?))\s*[\.\:\-]",
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html: str) -> str:
    text = html or ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def split_articles(text: str) -> list[dict[str, str]]:
    matches = list(DIEU_HEAD_RE.finditer(text))
    if not matches:
        return []

    articles: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        article_id = match.group(2)
        start = match.start(1)
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        title_line, _, body = chunk.partition("\n")
        title_line = title_line.strip()
        if not title_line.lower().startswith("điều"):
            title_line = f"Điều {article_id}."
        body = body.strip() or chunk
        articles.append(
            {
                "article_id": article_id,
                "title": title_line,
                "text": body,
            }
        )
    return articles


def parse_vlsp_document(doc_content: str) -> list[dict[str, str]]:
    text = html_to_text(doc_content)
    articles = split_articles(text)
    if articles:
        return articles
    if not text:
        return []
    return [{"article_id": "1", "title": "Điều 1.", "text": text}]


def doc_name_to_title(doc_name: str, law_id: str) -> str:
    name = (doc_name or "").strip()
    if not name:
        return law_id
    # Drop trailing ", số XX/YYYY/..." when the subject is already present.
    name = re.sub(r",\s*số\s+[\d/A-Za-zÀ-ỹĐđ\-]+.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^\s*(Luật|Nghị định|Nghị quyết|Thông tư liên tịch|Thông tư|Quyết định|Pháp lệnh|Chỉ thị|Văn bản hợp nhất)\s+",
                  "", name, count=1, flags=re.IGNORECASE)
    return name.strip() or doc_name.strip()


def pick_best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def score(row: dict[str, Any]) -> tuple:
        meta = row.get("metadata") or {}
        issue = meta.get("IssueDate")
        issue_key = issue.isoformat() if issue is not None else ""
        content_len = len(row.get("doc_content") or "")
        article_count = len(parse_vlsp_document(row.get("doc_content") or ""))
        return (article_count, content_len, issue_key)

    return max(rows, key=score)
