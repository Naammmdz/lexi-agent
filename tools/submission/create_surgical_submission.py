"""Surgical top-1 swap: replace anchor only when cache reranker strongly disagrees."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Callable

from _paths import REPO_ROOT
from create_domain_repair_submission import load_rows, update_answer
from create_recall_boost_submission import article_ref_key, doc_ref, maybe_prefer_cache_top1
from utils.submission_formatter import canonical_law_id, load_law_title_mapping

OUTPUT_DIR = REPO_ROOT / "submission_variants"
DEFAULT_BASE = OUTPUT_DIR / "submission_recall_boost_merged_vn_rerank_tight_v1.zip"
DEFAULT_CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_vn_rerank_tuned_merged.json"
MAPPING = REPO_ROOT / "data" / "law_id_to_title_merged.json"

TAX_LAWS = {
    "38/2019/QH14", "123/2020/NĐ-CP", "126/2020/NĐ-CP", "105/2020/TT-BTC",
    "133/2016/TT-BTC", "48/2024/QH15", "59/2020/QH14", "219/2013/TT-BTC",
}
LABOR_LAWS = {
    "45/2019/QH14", "41/2024/QH15", "12/2022/NĐ-CP", "85/2015/NĐ-CP",
    "145/2020/NĐ-CP", "58/2015/QH13",
}

TAX_TERMS = ("thuế", "hóa đơn", "hoá đơn", "mã số thuế", "khai thuế", "gtgt", "kế toán")
LABOR_TERMS = ("lao động", "bảo hiểm xã hội", "bhxh", "hợp đồng lao động", "tiền lương", "nghỉ việc")


def normalize_text(text: str) -> str:
    return str(text or "").lower().replace("ð", "đ")


def question_hits(question: str, terms: tuple[str, ...]) -> bool:
    q = normalize_text(question)
    return any(t in q for t in terms)


def law_in_set(law_id: str, allowed: set[str]) -> bool:
    lid = canonical_law_id(law_id)
    return lid in allowed or any(lid.endswith(s.split("/")[-1]) for s in allowed if "/" in s)


def make_domain_gate(
    allowed_laws: set[str] | None,
    question_terms: tuple[str, ...] | None,
) -> Callable[[str, str], bool]:
    def gate(question: str, new_law: str) -> bool:
        if allowed_laws and not law_in_set(new_law, allowed_laws):
            return False
        if question_terms and not question_hits(question, question_terms):
            return False
        return True

    return gate


def create_surgical_submission(
    base_zip: Path,
    output_zip: Path,
    debug_path: Path,
    live_cache: Path,
    mapping_path: Path,
    replace_min_gap: float,
    article_min_score: float = 0.9,
    cap_docs: int = 1,
    domain_gate: Callable[[str, str], bool] | None = None,
    same_law_only: bool = False,
    relative_gap: bool = False,
) -> dict[str, Any]:
    mapping = load_law_title_mapping(mapping_path)
    rows = load_rows(base_zip)
    cache = json.loads(live_cache.read_text(encoding="utf-8"))
    debug_rows: list[dict[str, Any]] = []
    swapped = 0
    skipped_domain = 0
    skipped_gap = 0

    for row in rows:
        row_id = str(row["id"])
        candidates = cache.get(row_id, [])
        if not candidates:
            continue
        before_articles = list(row.get("relevant_articles", []))
        before_docs = list(row.get("relevant_docs", []))
        if not before_articles:
            continue

        top_cand = candidates[0]
        new_ref = str(top_cand.get("article_ref", "")).strip()
        new_law = article_ref_key(new_ref)[0] if new_ref else ""
        if domain_gate and new_law and not domain_gate(row.get("question", ""), new_law):
            skipped_domain += 1
            continue

        top_score = float(top_cand.get("score", 0.0) or 0.0)
        current_key = article_ref_key(before_articles[0])
        new_key = article_ref_key(new_ref) if new_ref else ("", "")
        if current_key == new_key:
            continue

        current_score = None
        for cand in candidates:
            cand_ref = str(cand.get("article_ref", "")).strip()
            if cand_ref and article_ref_key(cand_ref) == current_key:
                current_score = float(cand.get("score", 0.0) or 0.0)
                break

        if current_score is not None:
            gap = top_score - current_score
            if relative_gap and top_score > 0:
                gap_ratio = gap / top_score
                if gap_ratio < replace_min_gap:
                    skipped_gap += 1
                    continue
            elif gap < replace_min_gap:
                skipped_gap += 1
                continue

        if same_law_only and new_law and current_key[0] and canonical_law_id(new_law) != canonical_law_id(current_key[0]):
            skipped_domain += 1
            continue

        if maybe_prefer_cache_top1(
            row,
            candidates,
            mapping,
            article_min_score=article_min_score,
            replace_min_gap=replace_min_gap,
            cap_docs=cap_docs,
        ):
            update_answer(row)
            swapped += 1
            debug_rows.append(
                {
                    "id": row_id,
                    "question": row.get("question", ""),
                    "gap": round(top_score - (current_score or 0.0), 4),
                    "before_articles": " || ".join(before_articles),
                    "after_articles": " || ".join(row.get("relevant_articles", [])),
                    "before_docs": " || ".join(before_docs),
                    "after_docs": " || ".join(row.get("relevant_docs", [])),
                }
            )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_zip.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")

    with debug_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "gap", "question", "before_articles", "after_articles", "before_docs", "after_docs"],
        )
        writer.writeheader()
        writer.writerows(debug_rows)

    avg_articles = round(sum(len(r.get("relevant_articles", [])) for r in rows) / len(rows), 3)
    return {
        "rows": len(rows),
        "top1_swapped": swapped,
        "skipped_domain": skipped_domain,
        "skipped_gap": skipped_gap,
        "replace_min_gap": replace_min_gap,
        "avg_articles": avg_articles,
        "output": str(output_zip),
        "debug": str(debug_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", required=True)
    parser.add_argument("--live-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--replace-min-gap", type=float, default=0.08)
    parser.add_argument("--article-min-score", type=float, default=0.9)
    parser.add_argument("--domain", choices=("all", "tax", "labor"), default="all")
    parser.add_argument("--same-law-only", action="store_true")
    parser.add_argument("--relative-gap", action="store_true")
    args = parser.parse_args()

    gate = None
    if args.domain == "tax":
        gate = make_domain_gate(TAX_LAWS, TAX_TERMS)
    elif args.domain == "labor":
        gate = make_domain_gate(LABOR_LAWS, LABOR_TERMS)

    stats = create_surgical_submission(
        Path(args.base),
        Path(args.output),
        Path(args.debug),
        Path(args.live_cache),
        Path(args.mapping),
        replace_min_gap=args.replace_min_gap,
        article_min_score=args.article_min_score,
        domain_gate=gate,
        same_law_only=args.same_law_only,
        relative_gap=args.relative_gap,
    )
    stats["domain"] = args.domain
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
