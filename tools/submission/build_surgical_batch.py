#!/usr/bin/env python3
"""Audit + build surgical variants + offline companion benchmark."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from _paths import REPO_ROOT
from utils.submission_formatter import canonical_law_id

PY = sys.executable
OUT = REPO_ROOT / "submission_variants"
BENCH = OUT / "local_benchmark"
BASE = OUT / "submission_recall_boost_merged_vn_rerank_tight_v1.zip"
CACHE = REPO_ROOT / "data" / "augmented" / "live_retrieval_vn_rerank_tuned_merged.json"

VARIANTS = [
    ("surgical_tax_same", ["--replace-min-gap", "0.08", "--domain", "tax", "--same-law-only"]),
    ("surgical_labor_same", ["--replace-min-gap", "0.08", "--domain", "labor", "--same-law-only"]),
    ("surgical_tax_rel015", ["--replace-min-gap", "0.15", "--domain", "tax", "--relative-gap"]),
    ("surgical_labor_rel015", ["--replace-min-gap", "0.15", "--domain", "labor", "--relative-gap"]),
]


def article_key(ref: str) -> tuple[str, str]:
    parts = str(ref).split("|")
    law = canonical_law_id(parts[0]) if parts else ""
    label = parts[-1].strip().lower() if len(parts) >= 3 else ""
    return law, label


def audit_swap_candidates() -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    with zipfile.ZipFile(BASE) as zf:
        rows = {str(r["id"]): r for r in json.loads(zf.read("results.json"))}

    diffs = []
    for row_id, row in rows.items():
        arts = row.get("relevant_articles", [])
        if not arts:
            continue
        cands = cache.get(row_id, [])
        if not cands:
            continue
        cur = article_key(arts[0])
        top_ref = str(cands[0].get("article_ref", ""))
        if not top_ref:
            continue
        top = article_key(top_ref)
        if cur == top:
            continue
        top_score = float(cands[0].get("score", 0))
        cur_score = 0.0
        for c in cands:
            ref = str(c.get("article_ref", ""))
            if ref and article_key(ref) == cur:
                cur_score = float(c.get("score", 0))
                break
        gap = top_score - cur_score
        diffs.append({"id": row_id, "gap": gap, "cur_law": cur[0], "top_law": top[0]})

    gaps = [d["gap"] for d in diffs]
    report = {
        "base": str(BASE.name),
        "cache_rows": len(cache),
        "top1_mismatch": len(diffs),
        "gap_ge_008": sum(1 for g in gaps if g >= 0.08),
        "gap_ge_010": sum(1 for g in gaps if g >= 0.10),
        "gap_ge_015": sum(1 for g in gaps if g >= 0.15),
        "sample": sorted(diffs, key=lambda x: -x["gap"])[:15],
    }
    path = BENCH / "surgical_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_variants() -> list[dict]:
    results = []
    for name, extra in VARIANTS:
        out = OUT / f"submission_recall_boost_merged_vn_rerank_{name}.zip"
        dbg = OUT / f"submission_recall_boost_merged_vn_rerank_{name}_debug.csv"
        cmd = [
            PY,
            "tools/submission/create_surgical_submission.py",
            "--base",
            str(BASE),
            "--output",
            str(out),
            "--debug",
            str(dbg),
            "--live-cache",
            str(CACHE),
            *extra,
        ]
        print("$", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        stats = json.loads(proc.stdout)
        stats["name"] = name
        results.append(stats)
        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return results


def build_domain_repair() -> dict:
    out = OUT / "submission_recall_boost_merged_domain_repair_v1.zip"
    dbg = OUT / "submission_recall_boost_merged_domain_repair_v1_debug.csv"
    cmd = [
        PY,
        "tools/submission/create_domain_repair_submission.py",
        "--base",
        str(BASE),
        "--output",
        str(out),
        "--debug",
        str(dbg),
        "--mode",
        "append",
    ]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    stats = json.loads(proc.stdout)
    stats["name"] = "domain_repair_v1"
    return stats


def companion_benchmark(paths: list[Path]) -> list[dict]:
    cmd = [
        PY,
        "tools/submission/benchmark_companion_candidate.py",
        str(BASE),
        *[str(p) for p in paths],
        "--output",
        str(BENCH / "surgical_companion_audit.json"),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    return data["results"]


def main() -> None:
    print("=== AUDIT ===", flush=True)
    audit_swap_candidates()
    print("\n=== BUILD SURGICAL ===", flush=True)
    built = build_variants()
    print("\n=== BUILD DOMAIN REPAIR ===", flush=True)
    repair = build_domain_repair()
    built.append(repair)

    paths = [Path(s["output"]) for s in built if "output" in s]
    print("\n=== COMPANION BENCHMARK ===", flush=True)
    bench = companion_benchmark(paths)

    summary = []
    for s in built:
        row = {"name": s.get("name"), "top1_swapped": s.get("top1_swapped", s.get("changed_rows")), "output": s.get("output")}
        for b in bench:
            if s.get("output", "").endswith(Path(b["path"]).name):
                row["ART_F2"] = b["ARTICLES_F2MACRO"]
                row["ART_REC"] = b["ARTICLES_RECALL"]
                row["ART_PREC"] = b["ARTICLES_PRECISION"]
        summary.append(row)

    out = BENCH / "surgical_batch_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUBMIT ORDER (companion audit) ===", flush=True)
    ranked = sorted(
        [r for r in summary if "ART_F2" in r],
        key=lambda x: (x.get("ART_F2", 0), x.get("top1_swapped", 0)),
        reverse=True,
    )
    for i, r in enumerate(ranked, 1):
        print(
            f"{i}. {Path(r['output']).name}: ART_F2={r.get('ART_F2')} "
            f"swapped={r.get('top1_swapped')} REC={r.get('ART_REC')} PREC={r.get('ART_PREC')}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
