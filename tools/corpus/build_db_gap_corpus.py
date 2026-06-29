"""Build Zalo-format gap corpus from legal_db SME seed (legacy dev — prefer HF vbpl)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.submission_formatter import canonical_law_id, format_law_title

ZALO_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus.json"
DB_SEED = REPO_ROOT / "data" / "augmented" / "db_seed_articles.json"
OUTPUT_CORPUS = REPO_ROOT / "data" / "augmented" / "db_gap_corpus.json"
OUTPUT_TITLES = REPO_ROOT / "data" / "augmented" / "db_gap_titles.json"
OUTPUT_REPORT = REPO_ROOT / "data" / "augmented" / "db_gap_report.json"
MERGED_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
MERGED_TITLES = REPO_ROOT / "data" / "law_id_to_title_merged.json"


def group_db_articles(articles: list[dict]) -> tuple[list[dict], dict[str, str]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    titles: dict[str, str] = {}
    for item in articles:
        law_id = canonical_law_id(item["law_id"])
        grouped[law_id].append(item)
        raw = str(item.get("document_title") or "").strip()
        if raw and law_id not in titles:
            titles[law_id] = format_law_title(law_id, raw)

    documents = []
    for law_id, items in sorted(grouped.items()):
        items.sort(key=lambda x: (int(x["article_id"]) if str(x["article_id"]).isdigit() else 10**9, x["article_id"]))
        documents.append(
            {
                "law_id": law_id.lower(),
                "articles": [
                    {
                        "article_id": str(item["article_id"]),
                        "title": item.get("title") or f"Điều {item['article_id']}",
                        "text": item.get("content") or "",
                    }
                    for item in items
                    if (item.get("content") or item.get("title"))
                ],
                "source": "legal_db",
            }
        )
    return documents, titles


def main() -> None:
    zalo_ids = {canonical_law_id(d["law_id"]) for d in json.loads(ZALO_CORPUS.read_text(encoding="utf-8"))}
    db_articles = json.loads(DB_SEED.read_text(encoding="utf-8"))
    gap_articles = [a for a in db_articles if canonical_law_id(a["law_id"]) not in zalo_ids]

    documents, titles = group_db_articles(gap_articles)
    article_count = sum(len(d["articles"]) for d in documents)

    report = {
        "zalo_laws": len(zalo_ids),
        "db_seed_articles": len(db_articles),
        "gap_laws": len(documents),
        "gap_articles": article_count,
        "law_ids": [d["law_id"] for d in documents],
    }

    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CORPUS.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TITLES.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # merge zalo + db gap
    zalo = json.loads(ZALO_CORPUS.read_text(encoding="utf-8"))
    zalo_keys = {canonical_law_id(d["law_id"]) for d in zalo}
    merged = list(zalo) + [d for d in documents if canonical_law_id(d["law_id"]) not in zalo_keys]
    MERGED_CORPUS.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    base_titles = json.loads((REPO_ROOT / "data" / "law_id_to_title.json").read_text(encoding="utf-8"))
    for law_id, title in titles.items():
        base_titles.setdefault(law_id.lower(), title)
    MERGED_TITLES.write_text(json.dumps(base_titles, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {OUTPUT_CORPUS}")
    print(f"merged {len(merged)} laws -> {MERGED_CORPUS}")


if __name__ == "__main__":
    main()
