#!/usr/bin/env python3
"""Build precision-refine variants and run companion + public audit benchmarks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _paths import REPO_ROOT

PY = sys.executable
OUT = REPO_ROOT / "submission_variants"
BENCH = OUT / "local_benchmark"
BASE = OUT / "submission_recall_boost_merged_vn_rerank_tight_v1.zip"

VARIANTS = [
    ("precision_refine_v1", []),
    ("precision_trim_only", ["--no-reorder"]),
    ("precision_reorder_only", ["--no-trim"]),
]


def build_one(name: str, extra: list[str]) -> dict:
    out = OUT / f"submission_recall_boost_merged_vn_rerank_{name}.zip"
    dbg = OUT / f"submission_recall_boost_merged_vn_rerank_{name}_debug.csv"
    cmd = [
        PY,
        "tools/submission/create_precision_refine_submission.py",
        "--base",
        str(BASE),
        "--output",
        str(out),
        "--debug",
        str(dbg),
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    stats = json.loads(proc.stdout)
    stats["name"] = name
    return stats


def benchmark(paths: list[Path], script: str, out_name: str) -> list[dict]:
    out = BENCH / out_name
    cmd = [PY, f"tools/submission/{script}", str(BASE), *[str(p) for p in paths], "--output", str(out)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)["results"]


def main() -> None:
    built = [build_one(name, extra) for name, extra in VARIANTS]
    paths = [Path(s["output"]) for s in built]

    companion = benchmark(paths, "benchmark_companion_candidate.py", "precision_refine_companion.json")
    public = benchmark(paths, "benchmark_public_audit_candidate.py", "precision_refine_public.json")

    base_comp = next(r for r in companion if str(BASE) in r["path"])
    base_pub = next(r for r in public if str(BASE) in r["path"])

    summary = []
    for s in built:
        row = {
            "name": s["name"],
            "changed_rows": s["changed_rows"],
            "reordered": s["reordered"],
            "trimmed": s["trimmed"],
            "avg_articles": s["avg_articles"],
            "output": s["output"],
        }
        comp = next((r for r in companion if s["output"] in r["path"]), None)
        pub = next((r for r in public if s["output"] in r["path"]), None)
        if comp:
            row["companion_ART_F2"] = comp["ARTICLES_F2MACRO"]
            row["companion_delta"] = round(comp["ARTICLES_F2MACRO"] - base_comp["ARTICLES_F2MACRO"], 4)
            row["companion_REC"] = comp["ARTICLES_RECALL"]
            row["companion_PREC"] = comp["ARTICLES_PRECISION"]
            row["companion_weak"] = len(comp.get("weak_cases", []))
        if pub:
            row["public_ART_F2"] = pub["ARTICLES_F2MACRO"]
            row["public_delta"] = round(pub["ARTICLES_F2MACRO"] - base_pub["ARTICLES_F2MACRO"], 4)
        summary.append(row)

    out = BENCH / "precision_refine_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== PRECISION REFINE BENCHMARK ===\n")
    print(f"BASE companion ART_F2={base_comp['ARTICLES_F2MACRO']} public ART_F2={base_pub['ARTICLES_F2MACRO']}\n")
    for r in sorted(summary, key=lambda x: (-(x.get("companion_delta", 0)), -(x.get("public_delta", 0)))):
        print(
            f"{r['name']}: changed={r['changed_rows']} trim={r['trimmed']} reorder={r['reordered']} "
            f"avg_art={r['avg_articles']} "
            f"comp_F2={r.get('companion_ART_F2')} (Δ{r.get('companion_delta')}) "
            f"pub_F2={r.get('public_ART_F2')} (Δ{r.get('public_delta')}) weak={r.get('companion_weak')}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
