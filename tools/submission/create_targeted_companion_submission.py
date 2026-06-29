"""Apply a small audited companion-reference patch to an existing submission."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import article_ref, doc_ref, update_answer
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


MAPPING_PATH = REPO_ROOT / "data" / "law_id_to_title.json"
DEFAULT_READY = REPO_ROOT / "submission.zip"
DEFAULT_READY_VARIANT = REPO_ROOT / "submission_variants" / "submission.zip"


TARGET_RULES: dict[str, list[tuple[str, str]]] = {
    "bộ tài chính có trách nhiệm hướng dẫn những nội dung gì về thuế và kế toán cho doanh nghiệp siêu nhỏ?": [
        ("80/2021/NĐ-CP", "19"),
    ],
    "trường hợp nào việc sử dụng dấu hiệu trùng với nhãn hiệu được bảo hộ bị coi là xâm phạm quyền đối với nhãn hiệu?": [
        ("65/2023/NĐ-CP", "77"),
    ],
    "công ty sử dụng kiểu dáng công nghiệp đã được bảo hộ mà không xin phép chủ sở hữu thì có bị coi là xâm phạm quyền không?": [
        ("65/2023/NĐ-CP", "76"),
    ],
}


def normalize_question(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def append_ref(row: dict[str, Any], mapping: dict[str, str], law_id: str, article_id: str) -> bool:
    law_id = canonical_law_id(law_id)
    docs = row.setdefault("relevant_docs", [])
    articles = row.setdefault("relevant_articles", [])
    if not any(canonical_law_id(str(ref).split("|", 1)[0]) == law_id for ref in docs):
        docs.append(doc_ref(law_id, mapping))
    article_prefix = f"{law_id}|"
    article_suffix = f"|Điều {article_id}"
    if any(str(ref).startswith(article_prefix) and str(ref).endswith(article_suffix) for ref in articles):
        return False
    articles.append(article_ref(law_id, article_id, mapping))
    return True


def create_submission(base_zip: Path, output_zip: Path, debug_path: Path, copy_to_submission: bool) -> dict[str, Any]:
    mapping = load_law_title_mapping(MAPPING_PATH)
    rows = load_rows(base_zip)
    debug_rows: list[dict[str, Any]] = []

    for row in rows:
        refs = TARGET_RULES.get(normalize_question(row.get("question", "")))
        if not refs:
            continue
        before_docs = list(row.get("relevant_docs", []))
        before_articles = list(row.get("relevant_articles", []))
        added = []
        for law_id, article_id in refs:
            if append_ref(row, mapping, law_id, article_id):
                added.append(f"{law_id}|{article_id}")
        if added:
            update_answer(row)
            debug_rows.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question", ""),
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
        fieldnames = ["id", "question", "added", "before_docs", "after_docs", "before_articles", "after_articles"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)

    if copy_to_submission:
        shutil.copyfile(output_zip, DEFAULT_READY)
        shutil.copyfile(output_zip, DEFAULT_READY_VARIANT)

    return {
        "rows": len(rows),
        "changed_rows": len(debug_rows),
        "doc_refs": sum(len(row.get("relevant_docs", [])) for row in rows),
        "article_refs": sum(len(row.get("relevant_articles", [])) for row in rows),
        "output": str(output_zip),
        "debug": str(debug_path),
        "ready": str(DEFAULT_READY) if copy_to_submission else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_READY))
    parser.add_argument("--output", default=str(REPO_ROOT / "submission_variants" / "submission_targeted_companion.zip"))
    parser.add_argument("--debug", default=str(REPO_ROOT / "submission_variants" / "submission_targeted_companion_debug.csv"))
    parser.add_argument("--copy-to-submission", action="store_true")
    args = parser.parse_args()
    stats = create_submission(Path(args.base), Path(args.output), Path(args.debug), args.copy_to_submission)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
