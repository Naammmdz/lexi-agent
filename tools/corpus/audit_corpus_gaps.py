#!/usr/bin/env python3
"""Audit corpus coverage for P4 expansion (train, submission, RRF cache)."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "tools" / "submission") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "submission"))

from config import Config
from utils.submission_formatter import canonical_law_id

OUT_DIR = REPO_ROOT / "submission_variants" / "local_benchmark"
DEFAULT_OUTPUT = OUT_DIR / "p4_corpus_gap_audit.json"

PRIORITY_SUBMISSIONS = [
    REPO_ROOT / "submission_variants" / "rrf_swap_v2.zip",
    REPO_ROOT / "submission_variants" / "submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.json",
    REPO_ROOT / "submission_variants" / "submission_recall_boost_merged_vn_rerank_tight_v1.zip",
]
RRF_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_rrf_wide_merged.json"
GAP_SOURCES = {
    "db_gap": REPO_ROOT / "data" / "augmented" / "db_gap_corpus.json",
    "vlsp_gap": REPO_ROOT / "data" / "augmented" / "vlsp_gap_corpus.json",
}


def load_corpus_law_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {canonical_law_id(doc.get("law_id", "")) for doc in data if doc.get("law_id")}


def load_submission_law_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            rows = json.loads(zf.read("results.json"))
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    codes: set[str] = set()
    for row in rows:
        for ref in row.get("relevant_docs", []) + row.get("relevant_articles", []):
            if "|" in str(ref):
                codes.add(canonical_law_id(str(ref).split("|", 1)[0]))
    return codes


def load_train_gold_laws(train_path: Path) -> dict[str, set[str]]:
    all_laws: set[str] = set()
    cross_doc: set[str] = set()
    tags_path = OUT_DIR / "train_subset_tags.json"
    cross_idx: set[int] = set()
    if tags_path.exists():
        payload = json.loads(tags_path.read_text(encoding="utf-8"))
        cross_idx = {t["index"] for t in payload["rows"] if t.get("cross_doc")}

    with train_path.open("r", encoding="utf-8", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            articles = ast.literal_eval(row["relevant_articles"])
            laws = {canonical_law_id(a["law_id"]) for a in articles if a.get("law_id")}
            all_laws.update(laws)
            if idx in cross_idx:
                cross_doc.update(laws)
    return {"all": all_laws, "cross_doc": cross_doc}


def load_rrf_cache_laws(cache_path: Path) -> set[str]:
    if not cache_path.exists():
        return set()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    laws: set[str] = set()
    for candidates in cache.values():
        for cand in candidates[:5]:
            law = cand.get("law_id") or ""
            if not law and cand.get("article_ref"):
                law = str(cand["article_ref"]).split("|", 1)[0]
            if law:
                laws.add(canonical_law_id(law))
    return laws


def gap_source_status(gap_paths: dict[str, Path], corpus_ids: set[str], zalo_ids: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in gap_paths.items():
        if not path.exists():
            out[name] = {"exists": False}
            continue
        gap_ids = load_corpus_law_ids(path)
        out[name] = {
            "exists": True,
            "laws": len(gap_ids),
            "not_in_zalo": sorted(gap_ids - zalo_ids),
            "not_in_merged": sorted(gap_ids - corpus_ids),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 corpus gap audit")
    parser.add_argument("--corpus", default=str(Config.CORPUS_PATH))
    parser.add_argument("--zalo", default=str(REPO_ROOT / "data" / "corpus" / "legal_corpus.json"))
    parser.add_argument("--train", default=str(REPO_ROOT / "data" / "train" / "train_qna.csv"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    zalo_path = Path(args.zalo)
    corpus_ids = load_corpus_law_ids(corpus_path)
    zalo_ids = load_corpus_law_ids(zalo_path)
    corpus_data = json.loads(corpus_path.read_text(encoding="utf-8"))
    article_count = sum(len(doc.get("articles", [])) for doc in corpus_data)

    train_laws = load_train_gold_laws(Path(args.train))
    submission_laws: set[str] = set()
    for sub in PRIORITY_SUBMISSIONS:
        submission_laws.update(load_submission_law_ids(sub))
    rrf_laws = load_rrf_cache_laws(RRF_CACHE)

    def missing_from_corpus(laws: set[str]) -> list[str]:
        return sorted(law for law in laws if law and law not in corpus_ids)

    payload = {
        "corpus_path": str(corpus_path),
        "corpus_laws": len(corpus_ids),
        "corpus_articles": article_count,
        "zalo_laws": len(zalo_ids),
        "merged_added_vs_zalo": len(corpus_ids - zalo_ids),
        "mentor_target_laws": 8436,
        "gap_vs_mentor_laws": 8436 - len(corpus_ids),
        "train_gold_missing": missing_from_corpus(train_laws["all"]),
        "cross_doc_gold_missing": missing_from_corpus(train_laws["cross_doc"]),
        "submission_cited_missing": missing_from_corpus(submission_laws),
        "rrf_cache_missing": missing_from_corpus(rrf_laws),
        "priority_fill_targets": sorted(
            set(missing_from_corpus(train_laws["all"]))
            | set(missing_from_corpus(submission_laws))
            | set(missing_from_corpus(rrf_laws))
        ),
        "gap_sources": gap_source_status(GAP_SOURCES, corpus_ids, zalo_ids),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"corpus: {payload['corpus_laws']} laws, {payload['corpus_articles']} articles")
    print(f"vs mentor 8436: gap {payload['gap_vs_mentor_laws']} laws")
    print(f"train missing: {len(payload['train_gold_missing'])}")
    print(f"submission missing: {len(payload['submission_cited_missing'])}")
    print(f"rrf cache missing: {len(payload['rrf_cache_missing'])}")
    print(f"priority fill targets: {len(payload['priority_fill_targets'])}")
    vlsp = payload["gap_sources"].get("vlsp_gap", {})
    if vlsp.get("not_in_merged"):
        print("vlsp not merged:", ", ".join(vlsp["not_in_merged"]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
