"""Create a domain-aware reranked submission from the current best repair run.

This is broader than the hand-written repair layer.  It keeps the current
submission as the high-precision anchor, then uses the DB-augmented article
corpus to add at most a few high-confidence companion documents/articles for
questions that clearly span multiple legal aspects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from _paths import REPO_ROOT
from create_augmented_submission import (
    article_relevance_boost,
    build_or_load_index,
    domain_boost,
    load_augmented_articles,
    normalize_text,
    tokenize,
)
from create_domain_repair_submission import (
    article_key,
    article_ref,
    dedupe_refs,
    doc_key,
    doc_ref,
    update_answer,
)
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


BASE_DIR = REPO_ROOT
OUTPUT_DIR = BASE_DIR / "submission_variants"
MAPPING_PATH = BASE_DIR / "data" / "law_id_to_title.json"
DEFAULT_BASE = OUTPUT_DIR / "submission_domain_repair_v47_tax_doc_order.zip"
DEFAULT_OUTPUT = OUTPUT_DIR / "submission_domain_rerank_balanced.zip"
DEFAULT_DEBUG = OUTPUT_DIR / "submission_domain_rerank_balanced_debug.csv"
DEFAULT_READY = BASE_DIR / "submission.zip"
DEFAULT_READY_VARIANT = OUTPUT_DIR / "submission.zip"


DOMAIN_LAWS: dict[str, set[str]] = {
    "tax": {
        "38/2019/QH14",
        "126/2020/NĐ-CP",
        "123/2020/NĐ-CP",
        "125/2020/NĐ-CP",
        "105/2020/TT-BTC",
        "78/2014/TT-BTC",
        "96/2015/TT-BTC",
        "218/2013/NĐ-CP",
        "85/2015/NĐ-CP",
        "117/2012/TT-BTC",
        "48/2024/QH15",
        "40/2021/TT-BTC",
    },
    "labor": {
        "45/2019/QH14",
        "145/2020/NĐ-CP",
        "12/2022/NĐ-CP",
        "84/2015/QH13",
        "85/2015/NĐ-CP",
        "41/2024/QH15",
        "152/2020/NĐ-CP",
        "143/2018/NĐ-CP",
    },
    "sme": {
        "04/2017/QH14",
        "80/2021/NĐ-CP",
        "39/2019/NĐ-CP",
        "34/2018/NĐ-CP",
        "38/2018/NĐ-CP",
        "05/2019/TT-BKHĐT",
        "54/2019/TT-BTC",
    },
    "ip": {
        "50/2005/QH11",
        "65/2023/NĐ-CP",
        "17/2023/NĐ-CP",
        "103/2006/NĐ-CP",
        "99/2013/NĐ-CP",
        "22/2018/NĐ-CP",
        "88/2010/NĐ-CP",
    },
    "consumer": {
        "19/2023/QH15",
        "55/2024/NĐ-CP",
        "98/2020/NĐ-CP",
        "52/2013/NĐ-CP",
        "85/2021/NĐ-CP",
    },
    "enterprise": {
        "59/2020/QH14",
        "168/2025/NĐ-CP",
        "38/2019/QH14",
        "105/2020/TT-BTC",
        "96/2015/NĐ-CP",
    },
    "commerce": {
        "36/2005/QH11",
        "91/2015/QH13",
        "54/2010/QH12",
        "81/2018/NĐ-CP",
        "16/2012/QH13",
        "38/2021/NĐ-CP",
        "40/2018/NĐ-CP",
    },
}

DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "tax": ("thuế", "hóa đơn", "hoá đơn", "mã số thuế", "khai thuế", "hoàn thuế", "nộp thừa"),
    "labor": ("lao động", "nhân viên", "hợp đồng lao động", "thử việc", "tiền lương", "bảo hiểm xã hội"),
    "sme": (
        "doanh nghiệp nhỏ và vừa",
        "nhỏ và vừa",
        "khởi nghiệp sáng tạo",
        "chuỗi giá trị",
        "quỹ phát triển doanh nghiệp nhỏ",
        "cơ sở ươm tạo",
        "khu làm việc chung",
    ),
    "ip": ("sở hữu trí tuệ", "sở hữu công nghiệp", "quyền tác giả", "nhãn hiệu", "sáng chế", "kiểu dáng"),
    "consumer": ("người tiêu dùng", "khách hàng", "dữ liệu khách hàng", "hợp đồng theo mẫu"),
    "enterprise": ("đăng ký doanh nghiệp", "hộ kinh doanh", "chi nhánh", "người đại diện", "giải thể", "tạm ngừng kinh doanh"),
    "commerce": ("thương mại", "hợp đồng mua bán", "khuyến mại", "quảng cáo", "trọng tài", "nhượng quyền"),
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def has_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def detect_domains(question: str) -> list[str]:
    q = normalize_text(question)
    domains = [name for name, terms in DOMAIN_TERMS.items() if has_any(q, terms)]
    if "khách hàng" in q and not has_any(q, ("người tiêu dùng", "hợp đồng theo mẫu", "khiếu nại", "bảo hành")):
        domains = [domain for domain in domains if domain != "consumer"]
    return domains


def allowed_laws(domains: Sequence[str]) -> set[str]:
    laws: set[str] = set()
    for domain in domains:
        laws.update(DOMAIN_LAWS.get(domain, set()))
    return laws


def is_complex_question(question: str, domains: Sequence[str]) -> bool:
    q = normalize_text(question)
    return (
        len(domains) >= 2
        or q.count(" và ") >= 2
        or "đồng thời" in q
        or re.search(r"\bvừa\b.+\bvừa\b", q) is not None
        or "cùng lúc" in q
        or "kèm theo" in q
    )


def skip_broad_append(question: str) -> bool:
    q = normalize_text(question)
    if (
        has_any(q, ("nền tảng thương mại điện tử", "sàn thương mại điện tử"))
        and "khấu trừ" in q
        and "nộp thuế thay" in q
        and "hộ kinh doanh" in q
    ):
        return True
    return False


def current_article_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    return {key for ref in row.get("relevant_articles", []) if (key := article_key(ref))}


def current_doc_keys(row: dict[str, Any]) -> set[str]:
    return {key for ref in row.get("relevant_docs", []) if (key := doc_key(ref))}


def score_candidates(question: str, articles: Sequence[dict[str, Any]], bm25, topn: int) -> list[dict[str, Any]]:
    scores = bm25.get_scores(tokenize(question))
    if len(scores) == 0:
        return []
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topn]
    max_score = max(float(scores[top_indices[0]]), 1e-9)
    candidates: list[dict[str, Any]] = []
    for idx in top_indices:
        article = articles[idx]
        law_id = canonical_law_id(article.get("law_id", ""))
        raw = float(scores[idx]) / max_score
        final = raw + domain_boost(question, article) + article_relevance_boost(question, article)
        candidates.append(
            {
                "law_id": law_id,
                "article_id": str(article.get("article_id", "")).strip(),
                "score": final,
                "raw": raw,
                "title": article.get("title", ""),
                "document_title": article.get("document_title", ""),
                "status": article.get("status", ""),
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def candidate_allowed(candidate: dict[str, Any], domains: Sequence[str], law_filter: set[str], question: str) -> bool:
    law_id = canonical_law_id(candidate["law_id"])
    q = normalize_text(question)
    if law_filter and law_id not in law_filter:
        return False
    if normalize_text(candidate.get("status", "")) == "expired":
        return False
    article_id = str(candidate.get("article_id", ""))
    if not article_id or article_id == "0":
        return False
    if law_id == "85/2015/NĐ-CP" and not has_any(q, ("lao động nữ", "lao động là nữ", "người lao động nữ", "doanh nghiệp sử dụng nhiều lao động nữ")):
        return False
    if law_id == "38/2019/QH14" and article_id == "27" and has_any(q, ("nền tảng thương mại điện tử", "sàn thương mại điện tử")):
        return False
    if law_id == "39/2019/NĐ-CP" and not has_any(q, ("quỹ", "cho vay", "vay vốn", "tài trợ vốn", "bảo lãnh tín dụng")):
        return False
    if law_id == "38/2018/NĐ-CP" and not has_any(q, ("lựa chọn", "quỹ đầu tư khởi nghiệp", "đầu tư khởi nghiệp sáng tạo", "đề án hỗ trợ")):
        return False
    if law_id == "88/2010/NĐ-CP" and "giống cây trồng" not in q:
        return False
    if law_id == "99/2013/NĐ-CP" and not has_any(q, ("xử phạt", "vi phạm", "xâm phạm", "cạnh tranh không lành mạnh", "biện pháp khắc phục")):
        return False
    if law_id == "91/2015/QH13" and any(domain in {"ip", "tax", "labor", "sme", "consumer", "enterprise"} for domain in domains):
        return False
    if article_id in {"1", "2", "3"} and not any(domain in {"tax", "labor", "consumer"} for domain in domains):
        return False
    return True


def add_candidate_refs(
    row: dict[str, Any],
    mapping: dict[str, str],
    candidates: Sequence[dict[str, Any]],
    domains: Sequence[str],
    min_score: float,
    max_docs: int,
    max_articles: int,
    question: str,
) -> list[str]:
    doc_keys = current_doc_keys(row)
    article_keys = current_article_keys(row)
    law_filter = allowed_laws(domains)
    added: list[str] = []

    for candidate in candidates:
        if len(row.get("relevant_docs", [])) >= max_docs and len(row.get("relevant_articles", [])) >= max_articles:
            break
        if candidate["score"] < min_score:
            continue
        if not candidate_allowed(candidate, domains, law_filter, question):
            continue
        law_id = canonical_law_id(candidate["law_id"])
        article_id = str(candidate["article_id"])
        akey = (law_id, article_id.lower())
        # This broad reranker is for companion-document recall.  Same-document
        # article expansion is handled by audited hard rules; doing it here
        # over-adds generic articles and hurts macro precision.
        if law_id in doc_keys:
            continue
        if akey in article_keys:
            continue
        if len(row.get("relevant_articles", [])) >= max_articles:
            continue

        if len(row.get("relevant_docs", [])) >= max_docs:
            continue
        row.setdefault("relevant_docs", []).append(doc_ref(law_id, mapping))
        doc_keys.add(law_id)
        row.setdefault("relevant_articles", []).append(article_ref(law_id, article_id, mapping))
        article_keys.add(akey)
        added.append(f"{law_id}|{article_id}|{candidate['score']:.3f}")
    if added:
        dedupe_refs(row)
        update_answer(row)
    return added


def create_domain_rerank_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    min_score: float,
    topn: int,
    max_docs: int,
    max_articles: int,
    copy_to_submission: bool = False,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    articles = load_augmented_articles(include_db_heuristic_codes=True)
    bm25, _ = build_or_load_index(articles)

    debug_rows: list[dict[str, Any]] = []
    changed = 0
    for row in rows:
        question = row.get("question", "")
        domains = detect_domains(question)
        if not domains or skip_broad_append(question) or not is_complex_question(question, domains):
            continue
        before_docs = list(row.get("relevant_docs", []))
        before_articles = list(row.get("relevant_articles", []))
        candidates = score_candidates(question, articles, bm25, topn=topn)
        added = add_candidate_refs(
            row,
            mapping,
            candidates,
            domains,
            min_score=min_score,
            max_docs=max_docs,
            max_articles=max_articles,
            question=question,
        )
        if added:
            changed += 1
            debug_rows.append(
                {
                    "id": row.get("id"),
                    "domains": ",".join(domains),
                    "question": question,
                    "added": " || ".join(added),
                    "before_docs": " || ".join(before_docs),
                    "after_docs": " || ".join(row.get("relevant_docs", [])),
                    "before_articles": " || ".join(before_articles),
                    "after_articles": " || ".join(row.get("relevant_articles", [])),
                }
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "domains", "question", "added", "before_docs", "after_docs", "before_articles", "after_articles"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    stats = {
        "rows": len(rows),
        "changed_rows": changed,
        "doc_refs": sum(len(row.get("relevant_docs", [])) for row in rows),
        "article_refs": sum(len(row.get("relevant_articles", [])) for row in rows),
        "multi_doc_rows": sum(1 for row in rows if len(row.get("relevant_docs", [])) > 1),
        "multi_article_rows": sum(1 for row in rows if len(row.get("relevant_articles", [])) > 1),
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
        "validation": validate_submission(output_zip),
    }
    if copy_to_submission:
        stats["ready_validation"] = validate_submission(DEFAULT_READY)
    return stats


def validate_submission(path: Path) -> dict[str, Any]:
    rows = load_rows(path)
    bad_rows = []
    for row in rows:
        if not row.get("relevant_docs") or not row.get("relevant_articles"):
            bad_rows.append({"id": row.get("id"), "reason": "empty_refs"})
        for ref in row.get("relevant_articles", []):
            if article_key(ref) is None:
                bad_rows.append({"id": row.get("id"), "reason": f"bad_article:{ref}"})
    with zipfile.ZipFile(path) as zf:
        entries = zf.namelist()
    return {
        "zip_entries": entries,
        "rows": len(rows),
        "doc_refs": sum(len(row.get("relevant_docs", [])) for row in rows),
        "article_refs": sum(len(row.get("relevant_articles", [])) for row in rows),
        "bad_rows": len(bad_rows),
        "bad_examples": bad_rows[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--min-score", type=float, default=1.85)
    parser.add_argument("--topn", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=3)
    parser.add_argument("--max-articles", type=int, default=3)
    parser.add_argument("--copy-to-submission", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = create_domain_rerank_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        min_score=args.min_score,
        topn=args.topn,
        max_docs=args.max_docs,
        max_articles=args.max_articles,
        copy_to_submission=args.copy_to_submission,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
