"""Extract SME-relevant articles for gap-fill (legacy dev path).

**Nguồn công khai để merge corpus (nghiệm thu / reproduce):**
https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents

Script này chỉ dùng khi có Postgres ``legal_db`` nội bộ (dev). Pipeline chuẩn
dùng ``tools/corpus/build_hf_vbpl_gap_corpus.py`` thay thế.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import psycopg2

from _paths import REPO_ROOT
from utils.submission_formatter import canonical_law_id


BASE_DIR = REPO_ROOT
OUTPUT_PATH = BASE_DIR / "data" / "augmented" / "db_seed_articles.json"

DB_CONFIG = {
    "host": "localhost",
    "port": 5435,
    "user": "legal_user",
    "password": "legal_password",
    "dbname": "legal_db",
}

SUBMISSION_SEED_FILES = [
    BASE_DIR / "submission.zip",
    BASE_DIR / "submission_variants" / "submission_hybrid_rerank_top3.zip",
    BASE_DIR / "submission_variants" / "submission_bm25_top2.zip",
    BASE_DIR / "submission_variants" / "submission_hybrid_top2.zip",
    BASE_DIR / "submission_variants" / "rrf_swap_v2.zip",
]

MANUAL_SEED_CODES = """
04/2017/QH14 80/2021/NĐ-CP 12/2022/NĐ-CP 123/2020/NĐ-CP 133/2016/TT-BTC
65/2023/NĐ-CP 132/2020/NĐ-CP 01/2021/TT-BKHĐT 41/2024/QH15 45/2019/QH14
59/2020/QH14 38/2019/QH14 50/2005/QH11 36/2005/QH11 91/2015/QH13
20/2023/QH15 52/2013/NĐ-CP 81/2018/NĐ-CP 152/2020/NĐ-CP 125/2020/NĐ-CP
70/2023/NĐ-CP 13/2023/NĐ-CP 168/2025/NĐ-CP 68/2025/TT-BTC 181/2025/NĐ-CP
48/2024/QH15 14/2008/QH12 100/2015/QH13 43/2013/QH13 63/2014/NĐ-CP
99/2013/NĐ-CP 103/2006/NĐ-CP 22/2018/NĐ-CP 119/2018/NĐ-CP 126/2020/NĐ-CP
28/2020/NĐ-CP 39/2018/NĐ-CP 39/2019/NĐ-CP 34/2018/NĐ-CP 38/2018/NĐ-CP
05/2019/TT-BKHĐT 54/2019/TT-BTC 68/2019/TT-BTC 105/2020/TT-BTC 69/2020/TT-BTC
88/2010/NĐ-CP 11/2015/TT-BKHCN 01/2007/TT-BKHCN 16/2016/TT-BKHCN
263/2016/TT-BTC 274/2016/TT-BTC 08/2015/NĐ-CP 134/2016/NĐ-CP
09/2018/NĐ-CP 15/2018/NĐ-CP 08/2021/TT-BTC 84/2015/QH13 85/2015/NĐ-CP
39/2016/NĐ-CP 44/2013/NĐ-CP 24/2018/NĐ-CP 09/2018/NĐ-CP
""".split()


def connect():
    return psycopg2.connect(**DB_CONFIG)


def submission_codes() -> set[str]:
    codes: set[str] = set()
    for path in SUBMISSION_SEED_FILES:
        if not path.exists():
            continue
        try:
            with zipfile.ZipFile(path) as zf:
                rows = json.loads(zf.read("results.json"))
        except Exception:
            continue
        for row in rows:
            for ref in row.get("relevant_docs", []) + row.get("relevant_articles", []):
                if "|" in str(ref):
                    codes.add(canonical_law_id(str(ref).split("|", 1)[0]))
    return codes


def heuristic_rule_codes(conn) -> set[str]:
    codes: set[str] = set()
    cur = conn.cursor()
    cur.execute("select target_document_code from heuristic_rules")
    for (code,) in cur.fetchall():
        if code and "/" in code and str(code)[0].isdigit():
            codes.add(canonical_law_id(str(code).strip()))
    return codes


def seed_codes(conn) -> list[str]:
    codes = {canonical_law_id(code) for code in MANUAL_SEED_CODES}
    codes.update(submission_codes())
    codes.update(heuristic_rule_codes(conn))
    return sorted(codes)


def find_documents(conn, codes: Sequence[str]) -> list[dict[str, Any]]:
    cur = conn.cursor()
    docs = []
    for code in codes:
        cur.execute(
            """
            select d.id, d.code, d.document_type, d.title, d.status,
                   d.effective_date, d.expiry_date, count(a.id) as article_count
            from documents d
            left join articles a on a.document_id = d.id
            where lower(d.code) = lower(%s)
            group by d.id
            order by article_count desc, d.effective_date desc nulls last
            limit 1
            """,
            (code,),
        )
        row = cur.fetchone()
        if not row:
            continue
        docs.append(
            {
                "document_id": row[0],
                "law_id": canonical_law_id(row[1]),
                "document_type": row[2] or "",
                "document_title": row[3] or "",
                "status": row[4] or "",
                "effective_date": row[5].isoformat() if row[5] else "",
                "expiry_date": row[6].isoformat() if row[6] else "",
                "article_count": int(row[7] or 0),
            }
        )
    return docs


def fetch_article_metadata(conn, document_ids: Sequence[int]) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        select a.id, a.document_id, a.article_number, a.title, a.sequence
        from articles a
        where a.document_id = any(%s)
        order by a.document_id, a.sequence, a.id
        """,
        (list(document_ids),),
    )
    return [
        {
            "article_db_id": row[0],
            "document_id": row[1],
            "article_id": str(row[2]).strip(),
            "article_title": str(row[3] or "").strip(),
            "sequence": int(row[4] or 0),
        }
        for row in cur.fetchall()
        if str(row[2]).strip() and str(row[2]).strip() != "0"
    ]


def fetch_content_batch(conn, article_ids: Sequence[int]) -> dict[int, str]:
    if not article_ids:
        return {}
    cur = conn.cursor()
    try:
        cur.execute(
            "select id, content from articles where id = any(%s)",
            (list(article_ids),),
        )
        return {int(row[0]): str(row[1] or "") for row in cur.fetchall()}
    except Exception:
        conn.rollback()
        if len(article_ids) == 1:
            return {int(article_ids[0]): ""}
        mid = len(article_ids) // 2
        left = fetch_content_batch(conn, article_ids[:mid])
        right = fetch_content_batch(conn, article_ids[mid:])
        left.update(right)
        return left


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def build_articles(docs: Sequence[dict[str, Any]], metadata: Sequence[dict[str, Any]], contents: dict[int, str]) -> list[dict[str, Any]]:
    doc_by_id = {doc["document_id"]: doc for doc in docs}
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for article in metadata:
        grouped[(article["document_id"], article["article_id"])].append(article)

    output = []
    for (document_id, article_id), chunks in grouped.items():
        doc = doc_by_id[document_id]
        chunks = sorted(chunks, key=lambda item: (item["sequence"], item["article_db_id"]))
        title = next((chunk["article_title"] for chunk in chunks if chunk["article_title"]), "")
        text_parts = [contents.get(chunk["article_db_id"], "") for chunk in chunks]
        content = normalize_spaces("\n".join(part for part in text_parts if part))
        if not content:
            content = title
        output.append(
            {
                "id": f"{doc['law_id']}_{article_id}",
                "law_id": doc["law_id"],
                "article_id": article_id,
                "title": title or f"Điều {article_id}",
                "content": content,
                "document_title": doc["document_title"],
                "document_type": doc["document_type"],
                "status": doc["status"],
                "effective_date": doc["effective_date"],
                "expiry_date": doc["expiry_date"],
                "source": "legal_db",
            }
        )
    output.sort(key=lambda item: (item["law_id"], int(item["article_id"]) if item["article_id"].isdigit() else 10**9, item["article_id"]))
    return output


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    codes = seed_codes(conn)
    docs = find_documents(conn, codes)
    metadata = fetch_article_metadata(conn, [doc["document_id"] for doc in docs])
    contents = fetch_content_batch(conn, [article["article_db_id"] for article in metadata])
    articles = build_articles(docs, metadata, contents)

    OUTPUT_PATH.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    found_codes = {doc["law_id"] for doc in docs}
    print(f"seed_codes={len(codes)} found_documents={len(docs)} articles={len(articles)}")
    print(f"missing_codes={len(set(codes) - found_codes)}")
    print(f"wrote={OUTPUT_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
