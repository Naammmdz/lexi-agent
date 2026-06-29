"""P3: conditional cap_docs=2 when retrieval signals cross-document relevance."""

from __future__ import annotations

from typing import Any

from utils.submission_formatter import article_label, canonical_law_id, format_law_title, get_mapping_title


def _law_id_from_candidate(cand: dict[str, Any]) -> str:
    law_id = cand.get("law_id", "")
    if not law_id:
        ref = str(cand.get("article_ref", "") or cand.get("doc_ref", ""))
        law_id = ref.split("|", 1)[0] if ref else ""
    return canonical_law_id(law_id)


def _score(cand: dict[str, Any]) -> float:
    return float(cand.get("score", 0.0) or 0.0)


def doc_ref_from_candidate(cand: dict[str, Any], mapping: dict[str, str]) -> str:
    existing = str(cand.get("doc_ref", "")).strip()
    if existing:
        return existing
    law_id = _law_id_from_candidate(cand)
    title = format_law_title(law_id, get_mapping_title(mapping, law_id))
    return f"{law_id}|{title}"


def candidates_from_retrieved_docs(
    docs: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[dict[str, Any]]:
    """Normalize live retrieval docs into cache-style candidate dicts."""
    out: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc.get("metadata", {})
        law_id = canonical_law_id(metadata.get("law_id", ""))
        if not law_id:
            continue
        title = format_law_title(law_id, get_mapping_title(mapping, law_id))
        label = article_label(metadata.get("title", ""), metadata.get("article_id", ""))
        out.append(
            {
                "law_id": law_id,
                "label": label,
                "doc_ref": f"{law_id}|{title}",
                "article_ref": f"{law_id}|{title}|{label}" if label else "",
                "score": float(doc.get("score", 0.0) or 0.0),
            }
        )
    return out


def find_second_doc_ref(
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    min_score: float = 0.9,
    max_gap: float = 0.03,
) -> str | None:
    """Return a 2nd doc ref when top-2 distinct laws have close reranker scores."""
    if len(candidates) < 2:
        return None

    top = candidates[0]
    top_law = _law_id_from_candidate(top)
    top_score = _score(top)
    if not top_law or top_score < min_score:
        return None

    for cand in candidates[1:]:
        law = _law_id_from_candidate(cand)
        if not law or law == top_law:
            continue
        score = _score(cand)
        if top_score - score > max_gap:
            break
        if score < min_score:
            continue
        return doc_ref_from_candidate(cand, mapping)
    return None


def should_add_second_doc(
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    min_score: float = 0.9,
    max_gap: float = 0.03,
) -> bool:
    return find_second_doc_ref(candidates, mapping, min_score, max_gap) is not None


def apply_conditional_second_doc(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
    min_score: float = 0.9,
    max_gap: float = 0.03,
    cap_docs: int = 2,
) -> bool:
    """Append a 2nd doc when the cross-doc heuristic fires."""
    if cap_docs <= 1:
        return False

    docs = list(row.get("relevant_docs", []))
    if len(docs) >= cap_docs:
        return False

    second = find_second_doc_ref(candidates, mapping, min_score, max_gap)
    if not second:
        return False

    existing = {canonical_law_id(str(d).split("|", 1)[0]) for d in docs}
    law = canonical_law_id(second.split("|", 1)[0])
    if law in existing:
        return False

    row["relevant_docs"] = docs + [second]
    return True
