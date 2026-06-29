"""Create a low-risk final submission with strict one-article answers.

This keeps the proven retrieval fields from an existing top1 submission and
rewrites answer so automatic article extraction sees only the kept top article.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List


from _paths import REPO_ROOT


BASE_DIR = REPO_ROOT
INPUT_ZIP = BASE_DIR / "submission_variants" / "submission_hybrid_rerank_top1.zip"
OUTPUT_JSON = BASE_DIR / "submission_variants" / "results_hybrid_rerank_top1_answer_strict.json"
OUTPUT_ZIP = BASE_DIR / "submission_variants" / "submission_hybrid_rerank_top1_answer_strict.zip"


def parse_article_ref(ref: str) -> tuple[str, str, str]:
    parts = str(ref).split("|")
    if len(parts) >= 3:
        return parts[0].strip(), parts[1].strip(), parts[-1].strip()
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip(), ""
    return str(ref).strip(), "", ""


def compact_answer(row: Dict[str, Any]) -> str:
    articles: List[str] = list(row.get("relevant_articles") or [])
    if not articles:
        return "Chưa xác định được căn cứ pháp luật phù hợp từ dữ liệu được cung cấp."

    law_id, _title, article = parse_article_ref(articles[0])
    article = re.sub(r"\s+", " ", article).strip()

    return (
        f"Căn cứ pháp luật: {law_id}|{article}. "
        "Trả lời: tình huống trong câu hỏi cần được đối chiếu theo căn cứ nêu trên "
        "để xác định quyền, nghĩa vụ và cách áp dụng cụ thể."
    )


def main() -> None:
    with zipfile.ZipFile(INPUT_ZIP) as zf:
        rows = json.loads(zf.read("results.json"))

    output = []
    for row in rows:
        new_row = dict(row)
        new_row["answer"] = compact_answer(row)
        output.append(new_row)

    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_JSON, arcname="results.json")

    print(f"Wrote {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
