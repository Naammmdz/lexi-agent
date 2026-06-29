"""Merge Zalo corpus with VLSP (or other) gap-fill documents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.submission_formatter import canonical_law_id

ZALO_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus.json"
GAP_CORPUS = REPO_ROOT / "data" / "augmented" / "vlsp_gap_corpus.json"
GAP_TITLES = REPO_ROOT / "data" / "augmented" / "vlsp_gap_titles.json"
TITLE_MAPPING = REPO_ROOT / "data" / "law_id_to_title.json"
OUTPUT_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
OUTPUT_MAPPING = REPO_ROOT / "data" / "law_id_to_title_merged.json"


def merge_corpus(zalo_path: Path, gap_path: Path) -> list[dict]:
    zalo = json.loads(zalo_path.read_text(encoding="utf-8"))
    gap = json.loads(gap_path.read_text(encoding="utf-8")) if gap_path.exists() else []

    by_id = {canonical_law_id(doc.get("law_id", "")): doc for doc in zalo if doc.get("law_id")}
    added = 0
    for doc in gap:
        law_id = canonical_law_id(doc.get("law_id", ""))
        if not law_id or law_id in by_id:
            continue
        by_id[law_id] = doc
        added += 1

    merged = list(zalo)
    merged.extend(doc for doc in gap if canonical_law_id(doc.get("law_id", "")) not in {
        canonical_law_id(item.get("law_id", "")) for item in zalo
    })
    print(f"zalo={len(zalo)} gap_added={added} merged={len(merged)}")
    return merged


def merge_titles(base_path: Path, gap_titles_path: Path) -> dict[str, str]:
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    gap = json.loads(gap_titles_path.read_text(encoding="utf-8")) if gap_titles_path.exists() else {}
    merged = dict(base)
    for law_id, title in gap.items():
        key = law_id.lower()
        if key not in merged:
            merged[key] = title
    print(f"title_mapping={len(merged)} (+{len(gap)} gap titles)")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Zalo + VLSP gap corpus.")
    parser.add_argument("--zalo", type=Path, default=ZALO_CORPUS)
    parser.add_argument("--gap", type=Path, default=GAP_CORPUS)
    parser.add_argument("--titles", type=Path, default=GAP_TITLES)
    parser.add_argument("--out-corpus", type=Path, default=OUTPUT_CORPUS)
    parser.add_argument("--out-titles", type=Path, default=OUTPUT_MAPPING)
    args = parser.parse_args()

    merged = merge_corpus(args.zalo, args.gap)
    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    args.out_corpus.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_titles = merge_titles(TITLE_MAPPING, args.titles)
    args.out_titles.write_text(json.dumps(merged_titles, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote_corpus={args.out_corpus}")
    print(f"wrote_titles={args.out_titles}")


if __name__ == "__main__":
    main()
