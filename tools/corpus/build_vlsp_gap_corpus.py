"""Build Zalo-format gap corpus from VLSP2025 legal-pretrain.

Fills law documents that R2AI needs but the Zalo 2021 corpus lacks.
Outputs article-level JSON compatible with ``legal_corpus.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corpus.vlsp_html_parser import doc_name_to_title, parse_vlsp_document, pick_best_row
from utils.submission_formatter import canonical_law_id, format_law_title

ZALO_CORPUS = REPO_ROOT / "data" / "corpus" / "legal_corpus.json"
OUTPUT_CORPUS = REPO_ROOT / "data" / "augmented" / "vlsp_gap_corpus.json"
OUTPUT_TITLES = REPO_ROOT / "data" / "augmented" / "vlsp_gap_titles.json"
OUTPUT_REPORT = REPO_ROOT / "data" / "augmented" / "vlsp_gap_report.json"
TIGHT_V1 = REPO_ROOT / "submission_variants" / "submission_recall_boost_tight_v1.zip"

MANUAL_GAP_CODES = """
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
39/2016/NĐ-CP 44/2013/NĐ-CP 24/2018/NĐ-CP 16/2012/QH13 145/2020/NĐ-CP
17/2023/NĐ-CP 19/2023/QH15 78/2014/TT-BTC 37/2015/NĐ-CP 50/2014/QH13
52/2013/NĐ-CP 50/2019/QH14 35/2018/QH14 50/2019/QH14
""".split()


def load_zalo_law_ids() -> set[str]:
    corpus = json.loads(ZALO_CORPUS.read_text(encoding="utf-8"))
    return {canonical_law_id(doc.get("law_id", "")) for doc in corpus if doc.get("law_id")}


def submission_law_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with zipfile.ZipFile(path) as zf:
        rows = json.loads(zf.read("results.json"))
    codes: set[str] = set()
    for row in rows:
        for ref in row.get("relevant_docs", []) + row.get("relevant_articles", []):
            if "|" in str(ref):
                codes.add(canonical_law_id(str(ref).split("|", 1)[0]))
    return codes


def gap_targets(zalo_ids: set[str], extra_only: bool) -> list[str]:
    wanted = {canonical_law_id(code) for code in MANUAL_GAP_CODES if code.strip()}
    wanted.update(submission_law_ids(TIGHT_V1))
    if extra_only:
        wanted = {code for code in wanted if code not in zalo_ids}
    else:
        wanted = {code for code in wanted if code not in zalo_ids}
    return sorted(wanted)


def harvest_vlsp_rows(targets: set[str], streaming: bool) -> dict[str, list[dict[str, Any]]]:
    split = "train"
    if streaming:
        dataset = load_dataset("VLSP2025-LegalSML/legal-pretrain", split=split, streaming=True)
    else:
        dataset = load_dataset("VLSP2025-LegalSML/legal-pretrain", split=split)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        meta = row.get("metadata") or {}
        ident = canonical_law_id(str(meta.get("DocIdentity") or ""))
        if ident in targets:
            grouped[ident].append(row)
    return grouped


def build_gap_documents(grouped: dict[str, list[dict[str, Any]]], targets: list[str]) -> tuple[list[dict], dict[str, str], dict[str, Any]]:
    documents: list[dict] = []
    titles: dict[str, str] = {}
    report: dict[str, Any] = {"found": [], "missing": [], "empty": []}

    for law_id in targets:
        rows = grouped.get(law_id, [])
        if not rows:
            report["missing"].append(law_id)
            continue

        row = pick_best_row(rows)
        meta = row.get("metadata") or {}
        articles = parse_vlsp_document(row.get("doc_content") or "")
        if not articles:
            report["empty"].append(law_id)
            continue

        raw_title = doc_name_to_title(str(meta.get("DocName") or ""), law_id)
        titles[law_id] = format_law_title(law_id, raw_title)
        documents.append(
            {
                "law_id": law_id.lower(),
                "articles": articles,
                "source": "vlsp2025_legal_pretrain",
                "doc_name": meta.get("DocName"),
                "issue_date": str(meta.get("IssueDate") or ""),
                "organ_name": meta.get("OrganName"),
            }
        )
        report["found"].append(
            {
                "law_id": law_id,
                "articles": len(articles),
                "variants": len(rows),
                "doc_name": meta.get("DocName"),
            }
        )

    return documents, titles, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VLSP gap corpus for missing R2AI laws.")
    parser.add_argument("--streaming", action="store_true", help="Stream HF dataset (lower memory).")
    parser.add_argument("--all-targets", action="store_true", help="Keep all seed codes, not only Zalo gaps.")
    args = parser.parse_args()

    zalo_ids = load_zalo_law_ids()
    targets = gap_targets(zalo_ids, extra_only=not args.all_targets)
    print(f"zalo_laws={len(zalo_ids)} gap_targets={len(targets)}")

    grouped = harvest_vlsp_rows(set(targets), streaming=args.streaming)
    documents, titles, report = build_gap_documents(grouped, targets)

    OUTPUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CORPUS.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TITLES.write_text(json.dumps(titles, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    article_count = sum(len(doc.get("articles", [])) for doc in documents)
    print(f"found_documents={len(documents)} articles={article_count}")
    print(f"missing={len(report['missing'])} empty={len(report['empty'])}")
    if report["missing"]:
        print("missing_codes:", ", ".join(report["missing"]))
    print(f"wrote_corpus={OUTPUT_CORPUS}")
    print(f"wrote_titles={OUTPUT_TITLES}")
    print(f"wrote_report={OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
