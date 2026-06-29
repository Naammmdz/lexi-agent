#!/usr/bin/env python3
"""Build legal_corpus_merged.json from Zalo + gap sources (Hugging Face vbpl.vn)."""

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
TITLE_MAPPING = REPO_ROOT / "data" / "law_id_to_title.json"
DEFAULT_GAP_SOURCES = [
    REPO_ROOT / "data" / "augmented" / "hf_vbpl_gap_corpus.json",
    # Legacy fallbacks (nội bộ / thử nghiệm trước khi chốt nguồn HF):
    REPO_ROOT / "data" / "augmented" / "db_gap_corpus.json",
    REPO_ROOT / "data" / "augmented" / "vlsp_bulk_gap_corpus.json",
]
OUTPUT_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus_merged.json"
OUTPUT_MAPPING = REPO_ROOT / "data" / "law_id_to_title_merged.json"
OUTPUT_REPORT = REPO_ROOT / "data" / "augmented" / "merged_v2_report.json"


def load_gap_titles(gap_path: Path) -> dict[str, str]:
    titles_path = gap_path.with_name(gap_path.stem.replace("_corpus", "_titles") + ".json")
    if not titles_path.exists():
        return {}
    return json.loads(titles_path.read_text(encoding="utf-8"))


def merge_sources(
    zalo_path: Path,
    gap_paths: list[Path],
) -> tuple[list[dict], dict[str, str], dict]:
    zalo = json.loads(zalo_path.read_text(encoding="utf-8"))
    by_id = {canonical_law_id(doc.get("law_id", "")): doc for doc in zalo if doc.get("law_id")}

    report: dict = {"zalo_laws": len(zalo), "sources": [], "added_by_source": {}}
    all_titles: dict[str, str] = {}

    for gap_path in gap_paths:
        if not gap_path.exists():
            report["sources"].append({"path": str(gap_path), "skipped": "missing"})
            continue
        gap = json.loads(gap_path.read_text(encoding="utf-8"))
        titles = load_gap_titles(gap_path)
        all_titles.update(titles)
        added = 0
        for doc in gap:
            law_id = canonical_law_id(doc.get("law_id", ""))
            if not law_id or law_id in by_id:
                continue
            by_id[law_id] = doc
            added += 1
        report["sources"].append({"path": str(gap_path), "gap_laws": len(gap), "added": added})
        report["added_by_source"][gap_path.name] = added

    merged = list(zalo)
    zalo_keys = {canonical_law_id(d["law_id"]) for d in zalo}
    for doc in by_id.values():
        law_id = canonical_law_id(doc.get("law_id", ""))
        if law_id not in zalo_keys:
            merged.append(doc)

    report["merged_laws"] = len(merged)
    report["merged_articles"] = sum(len(d.get("articles", [])) for d in merged)
    report["added_total"] = len(merged) - len(zalo)
    return merged, all_titles, report


def merge_title_mapping(base_path: Path, gap_titles: dict[str, str]) -> dict[str, str]:
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    merged = dict(base)
    for law_id, title in gap_titles.items():
        merged.setdefault(law_id.lower(), title)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Zalo + HF vbpl gap corpora.")
    parser.add_argument("--zalo", type=Path, default=ZALO_CORPUS)
    parser.add_argument(
        "--gap",
        type=Path,
        action="append",
        default=[],
        help="Gap corpus JSON (repeatable). Default: hf_vbpl + legacy gaps if present.",
    )
    parser.add_argument("--out-corpus", type=Path, default=OUTPUT_CORPUS)
    parser.add_argument("--out-titles", type=Path, default=OUTPUT_MAPPING)
    parser.add_argument("--report", type=Path, default=OUTPUT_REPORT)
    args = parser.parse_args()

    gap_paths = args.gap or DEFAULT_GAP_SOURCES
    merged, gap_titles, report = merge_sources(args.zalo, gap_paths)

    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    args.out_corpus.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_titles = merge_title_mapping(TITLE_MAPPING, gap_titles)
    args.out_titles.write_text(json.dumps(merged_titles, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote_corpus={args.out_corpus}")
    print(f"wrote_titles={args.out_titles}")


if __name__ == "__main__":
    main()
