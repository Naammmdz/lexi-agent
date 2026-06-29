"""Create a conservative recall submission by unioning changed references."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT


BASE_DIR = REPO_ROOT
DEFAULT_BASE = BASE_DIR / "submission_variants" / "submission_augmented_publicfix_top1.zip"
DEFAULT_CANDIDATE = BASE_DIR / "submission_variants" / "submission_augmented_hardrules_top1.zip"
DEFAULT_OUTPUT = BASE_DIR / "submission_variants" / "submission_augmented_hardrules_union_top2.zip"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    base_rows = load_rows(Path(args.base))
    candidate_rows = load_rows(Path(args.candidate))
    rows = []
    changed = 0
    extra_docs = 0
    extra_articles = 0
    for base, candidate in zip(base_rows, candidate_rows):
        row = dict(candidate)
        if base.get("relevant_docs") != candidate.get("relevant_docs") or base.get("relevant_articles") != candidate.get("relevant_articles"):
            changed += 1
            docs = dedupe(list(candidate.get("relevant_docs", [])) + list(base.get("relevant_docs", [])))
            articles = dedupe(list(candidate.get("relevant_articles", [])) + list(base.get("relevant_articles", [])))
            extra_docs += max(len(docs) - 1, 0)
            extra_articles += max(len(articles) - 1, 0)
            row["relevant_docs"] = docs
            row["relevant_articles"] = articles
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")
    print(f"rows={len(rows)} changed={changed} extra_docs={extra_docs} extra_articles={extra_articles}")
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
