"""Helpers for formatting R2AI submission legal references."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ARTICLE_RE = re.compile(r"(Điều\s*\d+[a-zA-Z]?)", re.IGNORECASE)

DOC_TYPE_PREFIXES = (
    "Thông tư liên tịch",
    "Văn bản hợp nhất",
    "Nghị quyết",
    "Nghị định",
    "Quyết định",
    "Thông tư",
    "Chỉ thị",
    "Pháp lệnh",
    "Luật",
)


def load_law_title_mapping(mapping_path: Path) -> Dict[str, str]:
    if not mapping_path.exists():
        return {}

    with mapping_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}

    return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip()}


def get_mapping_title(mapping: Dict[str, str], law_id: str) -> str:
    law_id = normalize_law_id(law_id)
    return (
        mapping.get(law_id)
        or mapping.get(law_id.lower())
        or mapping.get(law_id.upper())
        or law_id
    )


def normalize_law_id(law_id: str) -> str:
    return str(law_id).strip().replace("ð", "đ").replace("Ð", "Đ")


def canonical_law_id(law_id: str) -> str:
    canonical = normalize_law_id(law_id).upper()
    return canonical.replace("-TTG", "-TTg").replace("/TTG", "/TTg")


def infer_doc_type(law_id: str, raw_title: str = "") -> str:
    law_id_upper = canonical_law_id(law_id)
    title_lower = raw_title.strip().lower()

    if title_lower.startswith("luật"):
        return "Luật"
    if title_lower.startswith("nghị định"):
        return "Nghị định"
    if title_lower.startswith("nghị quyết"):
        return "Nghị quyết"
    if title_lower.startswith("quyết định"):
        return "Quyết định"
    if title_lower.startswith("thông tư liên tịch"):
        return "Thông tư liên tịch"
    if title_lower.startswith("thông tư"):
        return "Thông tư"
    if title_lower.startswith("văn bản hợp nhất"):
        return "Văn bản hợp nhất"
    if title_lower.startswith("chỉ thị"):
        return "Chỉ thị"
    if title_lower.startswith("pháp lệnh"):
        return "Pháp lệnh"

    if "TTLT-" in law_id_upper:
        return "Thông tư liên tịch"
    if "VBHN" in law_id_upper:
        return "Văn bản hợp nhất"
    if "NĐ-CP" in law_id_upper or "ND-CP" in law_id_upper:
        return "Nghị định"
    if "/TT-" in law_id_upper:
        return "Thông tư"
    if "/QĐ-" in law_id_upper or "/QD-" in law_id_upper:
        return "Quyết định"
    if "/NQ-" in law_id_upper or law_id_upper.endswith("/NQ"):
        return "Nghị quyết"
    if "/CT-" in law_id_upper:
        return "Chỉ thị"
    if "QH" in law_id_upper:
        return "Luật"
    return ""


def _starts_with_doc_type(title: str) -> str:
    title_lower = title.strip().lower()
    for prefix in DOC_TYPE_PREFIXES:
        if title_lower.startswith(prefix.lower()):
            return prefix
    return ""


def format_law_title(law_id: str, raw_title: str) -> str:
    """Return "<Loại văn bản> <Mã văn bản> <Trích yếu>" where possible."""
    law_id_canonical = canonical_law_id(law_id)
    clean_title = str(raw_title or "").replace("đ|nh", "định").replace("|", " ")
    clean_title = re.sub(r"\s+", " ", clean_title.strip())
    if not clean_title or clean_title.lower() == str(law_id).strip().lower():
        clean_title = ""

    if law_id_canonical and law_id_canonical in clean_title.upper():
        return clean_title

    existing_type = _starts_with_doc_type(clean_title)
    if existing_type:
        remainder = clean_title[len(existing_type) :].strip()
        return f"{existing_type} {law_id_canonical} {remainder}".strip()

    doc_type = infer_doc_type(law_id, clean_title)
    if doc_type:
        return f"{doc_type} {law_id_canonical} {clean_title}".strip()

    return f"{law_id_canonical} {clean_title}".strip()


def article_label(article_title: str, article_id: Any = "") -> str:
    match = ARTICLE_RE.search(str(article_title or ""))
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()

    article_id = str(article_id or "").strip()
    if article_id:
        return f"Điều {article_id}"

    return ""


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def normalize_submission_rows(
    rows: List[Dict[str, Any]], mapping: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    normalized: List[Dict[str, Any]] = []
    stats = {
        "rows": len(rows),
        "doc_refs": 0,
        "article_refs": 0,
        "changed_doc_refs": 0,
        "changed_article_refs": 0,
    }

    for row in rows:
        new_row = dict(row)
        docs = []
        for ref in row.get("relevant_docs", []):
            parts = str(ref).split("|", 1)
            if len(parts) != 2:
                docs.append(str(ref))
                continue
            law_id, _old_title = parts
            clean_law_id = canonical_law_id(law_id)
            title = format_law_title(clean_law_id, get_mapping_title(mapping, clean_law_id))
            new_ref = f"{clean_law_id}|{title}"
            stats["doc_refs"] += 1
            if new_ref != ref:
                stats["changed_doc_refs"] += 1
            docs.append(new_ref)

        articles = []
        for ref in row.get("relevant_articles", []):
            ref_text = str(ref)
            first_split = ref_text.split("|", 1)
            if len(first_split) != 2:
                articles.append(ref_text)
                continue
            law_id, rest = first_split
            clean_law_id = canonical_law_id(law_id)
            last_split = rest.rsplit("|", 1)
            if len(last_split) != 2:
                articles.append(str(ref))
                continue
            _old_title, old_article = last_split
            title = format_law_title(clean_law_id, get_mapping_title(mapping, clean_law_id))
            label = article_label(old_article)
            if not label:
                label = old_article.strip()
            new_ref = f"{clean_law_id}|{title}|{label}"
            stats["article_refs"] += 1
            if new_ref != ref:
                stats["changed_article_refs"] += 1
            articles.append(new_ref)

        new_row["relevant_docs"] = dedupe_keep_order(docs)
        new_row["relevant_articles"] = dedupe_keep_order(articles)
        normalized.append(new_row)

    return normalized, stats
