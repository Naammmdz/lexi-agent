"""Lightweight query-document relevance boosts for legal retrieval."""

from __future__ import annotations

import re
from typing import Any

from utils.submission_formatter import canonical_law_id

STOPWORDS = {
    "của",
    "và",
    "về",
    "cho",
    "các",
    "những",
    "được",
    "không",
    "như",
    "nào",
    "trong",
    "theo",
    "khi",
    "nếu",
    "phải",
    "cần",
    "quy",
    "định",
    "doanh",
    "nghiệp",
    "công",
    "ty",
    "trường",
    "hợp",
}

ARTICLE_RE = re.compile(r"điều\s+([0-9]+[a-z]?)", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w/.-]+", re.UNICODE)


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ð", "đ")).strip()


def query_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in TOKEN_RE.findall(normalize_query(text))
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def within_law_relevance_bonus(query: str, doc: dict[str, Any]) -> float:
    """Score bonus for article-level relevance within a law."""
    metadata = doc.get("metadata", {})
    title = normalize_query(doc.get("title", ""))
    doc_title = normalize_query(metadata.get("document_title", ""))
    content = normalize_query(str(doc.get("content", ""))[:1800])
    title_text = f"{title} {doc_title}".strip()
    full_text = f"{title_text} {content}"

    q_tokens = query_tokens(query)
    if not q_tokens:
        return 0.0

    title_tokens = query_tokens(title_text)
    content_tokens = query_tokens(content)
    bonus = 0.0
    bonus += min(len(q_tokens & title_tokens) / max(len(q_tokens), 1) * 0.18, 0.14)
    bonus += min(len(q_tokens & content_tokens) / max(len(q_tokens), 1) * 0.08, 0.08)

    for phrase in _query_phrases(query):
        if phrase in title_text:
            bonus += 0.06
        elif phrase in content:
            bonus += 0.02

    article_id = str(metadata.get("article_id", "")).strip().lower()
    explicit = ARTICLE_RE.search(normalize_query(query))
    if explicit and explicit.group(1).lower() == article_id:
        bonus += 0.22

    if article_id in {"1", "2", "3"} and not any(
        term in normalize_query(query) for term in ("phạm vi", "đối tượng", "giải thích", "hình thức xử phạt")
    ):
        bonus -= 0.07

    return bonus


def _query_phrases(query: str) -> list[str]:
    q = normalize_query(query)
    phrases: list[str] = []
    for size in (4, 3, 2):
        words = q.split()
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size])
            if len(phrase) >= 8:
                phrases.append(phrase)
    return phrases


def apply_within_law_rescore(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank retrieved docs with within-law article relevance boosts."""
    if not docs:
        return docs

    rescored: list[dict[str, Any]] = []
    for doc in docs:
        bonus = within_law_relevance_bonus(query, doc)
        updated = dict(doc)
        updated["within_law_bonus"] = bonus
        updated["score"] = float(doc.get("score", 0.0) or 0.0) + bonus
        rescored.append(updated)

    rescored.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)

    # Promote the strongest article from the top law when scores are close.
    top_law = canonical_law_id(rescored[0].get("metadata", {}).get("law_id", ""))
    if top_law:
        law_docs = [
            doc
            for doc in rescored
            if canonical_law_id(doc.get("metadata", {}).get("law_id", "")) == top_law
        ]
        if len(law_docs) > 1:
            best_in_law = max(law_docs, key=lambda item: float(item.get("score", 0.0) or 0.0))
            others = [doc for doc in rescored if doc is not best_in_law]
            rescored = [best_in_law] + others

    return rescored


def law_id_from_doc(doc: dict[str, Any]) -> str:
    return canonical_law_id(doc.get("metadata", {}).get("law_id", ""))


def rank_laws_by_hybrid_score(docs: list[dict[str, Any]]) -> list[tuple[str, float]]:
    from collections import defaultdict

    law_scores: dict[str, float] = defaultdict(float)
    for doc in docs:
        law = law_id_from_doc(doc)
        if law:
            law_scores[law] += float(doc.get("score", 0.0) or 0.0)
    return sorted(law_scores.items(), key=lambda item: item[1], reverse=True)


def apply_law_shortlist_filter(
    docs: list[dict[str, Any]],
    *,
    top_laws: int,
    min_docs: int,
) -> list[dict[str, Any]]:
    """Keep rerank candidates within the top-N laws from hybrid scores."""
    if not docs or top_laws <= 0:
        return docs

    ranked_laws = rank_laws_by_hybrid_score(docs)
    if not ranked_laws:
        return docs

    allowed: set[str] = {law for law, _ in ranked_laws[:top_laws]}
    filtered = [doc for doc in docs if law_id_from_doc(doc) in allowed]

    extra = top_laws
    while len(filtered) < min_docs and extra < len(ranked_laws):
        allowed.add(ranked_laws[extra][0])
        filtered = [doc for doc in docs if law_id_from_doc(doc) in allowed]
        extra += 1

    if not filtered:
        return docs

    filtered.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return filtered


def fuse_hybrid_rrf(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge BM25 + dense hits with reciprocal rank fusion."""
    fused: dict[str, dict[str, Any]] = {}

    def add_ranked(results: list[dict[str, Any]], source: str) -> None:
        for rank, doc in enumerate(results):
            doc_id = str(doc.get("id", "")).strip()
            if not doc_id:
                continue
            entry = fused.get(doc_id)
            if entry is None:
                entry = {"doc": dict(doc), "rrf": 0.0}
                fused[doc_id] = entry
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)
            if source == "bm25":
                entry["doc"]["bm25_score"] = float(doc.get("score", 0.0) or 0.0)
                entry["doc"].setdefault("retrieval_method", "bm25")
            else:
                entry["doc"]["vector_score"] = float(doc.get("score", 0.0) or 0.0)
                method = entry["doc"].get("retrieval_method")
                entry["doc"]["retrieval_method"] = "hybrid" if method == "bm25" else "vector"

    add_ranked(bm25_results, "bm25")
    add_ranked(vector_results, "vector")

    merged: list[dict[str, Any]] = []
    for entry in fused.values():
        doc = entry["doc"]
        doc["rrf_score"] = entry["rrf"]
        doc["score"] = entry["rrf"]
        merged.append(doc)
    merged.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return merged


def fuse_multi_query_rrf(
    ranked_pools: list[list[dict[str, Any]]],
    *,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge ranked doc lists from multiple sub-queries via RRF."""
    if not ranked_pools:
        return []
    if len(ranked_pools) == 1:
        return list(ranked_pools[0])

    fused: dict[str, dict[str, Any]] = {}
    for pool in ranked_pools:
        for rank, doc in enumerate(pool):
            doc_id = str(doc.get("id", "")).strip()
            if not doc_id:
                continue
            entry = fused.get(doc_id)
            if entry is None:
                entry = {"doc": dict(doc), "rrf": 0.0}
                fused[doc_id] = entry
            entry["rrf"] += 1.0 / (rrf_k + rank + 1)

    merged: list[dict[str, Any]] = []
    for entry in fused.values():
        doc = entry["doc"]
        doc["rrf_score"] = entry["rrf"]
        doc["score"] = entry["rrf"]
        doc["retrieval_method"] = doc.get("retrieval_method", "decomposed_hybrid")
        merged.append(doc)
    merged.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return merged


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def apply_two_stage_within_law_rerank(
    reranker: Any,
    query: str,
    docs: list[dict[str, Any]],
    *,
    top_laws: int,
    within_weight: float,
) -> list[dict[str, Any]]:
    """Re-rerank within top laws and blend with global scores (no forced #1 promote)."""
    if not docs or not reranker or top_laws <= 0 or within_weight <= 0:
        return docs

    from collections import defaultdict

    global_scores = {str(doc.get("id", "")): float(doc.get("score", 0.0) or 0.0) for doc in docs}
    by_law: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        law = law_id_from_doc(doc)
        if law:
            by_law[law].append(doc)

    if not by_law:
        return docs

    ranked_laws = sorted(
        by_law,
        key=lambda law: max(float(d.get("score", 0.0) or 0.0) for d in by_law[law]),
        reverse=True,
    )[:top_laws]

    blended = {str(doc.get("id", "")): dict(doc) for doc in docs}
    for law in ranked_laws:
        subset = by_law[law]
        reranked = reranker.rerank_documents(query, subset, top_k=None)
        if not reranked:
            continue
        within_norm = _normalize_scores([float(d.get("reranker_score", 0.0) or 0.0) for d in reranked])
        for doc, w_score in zip(reranked, within_norm):
            doc_id = str(doc.get("id", ""))
            if doc_id not in blended:
                continue
            g = global_scores.get(doc_id, 0.0)
            fused = (1.0 - within_weight) * g + within_weight * w_score
            blended[doc_id]["within_law_rerank_score"] = w_score
            blended[doc_id]["score"] = fused

    out = list(blended.values())
    out.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return out
