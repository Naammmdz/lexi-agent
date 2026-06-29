"""Print likely off-domain rows and rough corpus matches for manual repair.

This is an analysis helper only. It does not write a submission.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import article_key, doc_key
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


DOMAIN_TERMS: dict[str, tuple[str, ...]] = {
    "environment": ("môi trường", "nước thải", "khí nhà kính", "chất thải", "xả thải", "quan trắc"),
    "postal": ("bưu chính", "bưu gửi", "giấy phép bưu chính"),
    "customs": ("hải quan", "đại lý làm thủ tục hải quan", "khai hải quan"),
    "petroleum": ("xăng dầu", "khí dầu mỏ", "lpg"),
    "price": ("hiệp thương giá", "bình ổn giá", "thẩm định giá", "niêm yết giá"),
    "education": ("giáo dục", "trường", "cơ sở giáo dục", "đào tạo"),
    "alcohol": ("rượu", "bia", "thuốc lá"),
    "construction": ("xây dựng", "công trình", "giấy phép xây dựng", "phá dỡ"),
    "commerce": ("khuyến mại", "nhượng quyền", "đại lý thương mại", "logistics", "hội chợ", "triển lãm"),
    "enterprise": ("doanh nghiệp", "chi nhánh", "người đại diện", "vốn điều lệ", "đăng ký kinh doanh"),
    "labor": ("lao động", "hợp đồng lao động", "tiền lương", "người lao động", "bảo hiểm xã hội"),
    "ip": ("sở hữu trí tuệ", "quyền tác giả", "nhãn hiệu", "sáng chế", "kiểu dáng"),
    "tax": ("thuế", "hóa đơn", "hoá đơn", "mã số thuế", "khai thuế"),
}

LAW_DOMAIN_HINTS: dict[str, tuple[str, ...]] = {
    "environment": ("môi trường", "khí nhà kính", "nước thải", "chất thải"),
    "postal": ("bưu chính", "viễn thông", "giao dịch điện tử"),
    "customs": ("hải quan", "khai hải quan"),
    "petroleum": ("xăng dầu", "dầu khí", "khí dầu mỏ"),
    "price": ("giá", "thẩm định giá"),
    "education": ("giáo dục", "đào tạo"),
    "alcohol": ("rượu", "bia", "thuốc lá"),
    "construction": ("xây dựng", "công trình"),
    "commerce": ("thương mại", "khuyến mại", "nhượng quyền", "logistics", "hội chợ"),
    "enterprise": ("doanh nghiệp", "đầu tư", "kinh doanh"),
    "labor": ("lao động", "bảo hiểm xã hội", "việc làm"),
    "ip": ("sở hữu trí tuệ", "quyền tác giả", "sở hữu công nghiệp"),
    "tax": ("thuế", "hóa đơn", "hoá đơn"),
}

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
}


def normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("ð", "đ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w/.-]+", normalize_text(text), flags=re.UNICODE)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def domains_from_text(text: str, terms: dict[str, tuple[str, ...]]) -> set[str]:
    q = normalize_text(text)
    return {domain for domain, needles in terms.items() if any(needle in q for needle in needles)}


def article_label(ref: str) -> str:
    key = article_key(ref)
    if not key:
        return ""
    return f"{key[0]}|{key[1]}"


def build_articles(corpus_path: Path, mapping: dict[str, str]) -> list[dict[str, str]]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    articles: list[dict[str, str]] = []
    for doc in corpus:
        law_id = canonical_law_id(doc.get("law_id", ""))
        law_title = mapping.get(law_id, "")
        for article in doc.get("articles", []):
            text = " ".join(
                [
                    law_id,
                    law_title,
                    str(article.get("article_id", "")),
                    str(article.get("title", "")),
                    str(article.get("text", "")),
                ]
            )
            articles.append(
                {
                    "law_id": law_id,
                    "article_id": str(article.get("article_id", "")),
                    "title": str(article.get("title", "")),
                    "law_title": law_title,
                    "norm": normalize_text(text),
                }
            )
    return articles


def score_article(question: str, qdomains: set[str], article: dict[str, str]) -> float:
    qnorm = normalize_text(question)
    qtokens = [tok for tok in tokenize(qnorm) if len(tok) >= 3 and tok not in STOPWORDS]
    counts = Counter(qtokens)
    score = 0.0
    anorm = article["norm"]
    for token, count in counts.items():
        if token in anorm:
            score += min(count, 3)
    for domain in qdomains:
        for term in DOMAIN_TERMS.get(domain, ()):
            if term in qnorm and term in anorm:
                score += 8.0
    if any(domain in domains_from_text(article["law_title"], LAW_DOMAIN_HINTS) for domain in qdomains):
        score += 10.0
    if normalize_text(article["title"]).startswith("điều 1."):
        score -= 4.0
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=str(REPO_ROOT / "submission.zip"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()

    mapping = load_law_title_mapping(REPO_ROOT / "data" / "law_id_to_title.json")
    rows = load_rows(Path(args.submission))
    articles = build_articles(REPO_ROOT / "data" / "corpus" / "legal_corpus.json", mapping)

    printed = 0
    for row in rows:
        qdomains = domains_from_text(row.get("question", ""), DOMAIN_TERMS)
        if not qdomains:
            continue
        doc_domains: set[str] = set()
        doc_ids = []
        for ref in row.get("relevant_docs", []):
            key = doc_key(ref)
            if key:
                doc_ids.append(key)
            doc_domains.update(domains_from_text(ref, LAW_DOMAIN_HINTS))
        if qdomains & doc_domains:
            continue
        candidates = sorted(
            ((score_article(row["question"], qdomains, article), article) for article in articles),
            key=lambda item: item[0],
            reverse=True,
        )[: args.top]
        if not candidates or candidates[0][0] < 14:
            continue
        printed += 1
        print(f"\nID {row['id']} qdomains={sorted(qdomains)} current_docs={doc_ids}")
        print(row["question"])
        print("current_articles:", "; ".join(article_label(ref) for ref in row.get("relevant_articles", [])))
        for score, article in candidates:
            print(
                f"  {score:5.1f} {article['law_id']}|{article['article_id']} "
                f"{article['law_title']} :: {article['title']}"
            )
        if printed >= args.limit:
            break


if __name__ == "__main__":
    main()
