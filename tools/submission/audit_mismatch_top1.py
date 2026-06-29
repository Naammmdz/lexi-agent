"""Audit submission top-1 vs live cache; proxy-label via train neighbors."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT
from create_domain_repair_submission import article_key
from utils.submission_formatter import canonical_law_id, load_law_title_mapping


STOPWORDS = {
    "của", "và", "về", "cho", "các", "những", "được", "không", "như", "nào",
    "trong", "theo", "khi", "nếu", "phải", "cần", "quy", "định", "doanh",
    "nghiệp", "công", "ty", "trường", "hợp",
}

DEFAULT_BASE = REPO_ROOT / "submission_variants/submission_recall_boost_merged_vn_rerank_tight_v1.zip"
DEFAULT_CACHE = REPO_ROOT / "data/augmented/live_retrieval_vn_rerank_tuned_merged.json"
OUT_DIR = REPO_ROOT / "submission_variants/local_benchmark"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower().replace("ð", "đ")).strip()


def tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[\w/.-]+", norm(text), flags=re.UNICODE)
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def token_f1(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    p = inter / len(a)
    r = inter / len(b)
    return 2 * p * r / (p + r)


def load_submission(path: Path) -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        rows = json.loads(zf.read("results.json"))
    return {str(r["id"]): r for r in rows}


def load_train() -> list[tuple[str, set[str], set[tuple[str, str]], set[str]]]:
    rows = []
    with (REPO_ROOT / "data/train/train_qna.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            arts = set()
            docs = set()
            for item in ast.literal_eval(r["relevant_articles"]):
                law = canonical_law_id(item.get("law_id", ""))
                art = str(item.get("article_id", "")).lower()
                if law:
                    docs.add(law)
                if law and art:
                    arts.add((law, art))
            rows.append((r["question"], tokens(r["question"]), arts, docs))
    return rows


def top1_key(row: dict[str, Any]) -> tuple[str, str] | None:
    arts = row.get("relevant_articles", [])
    return article_key(arts[0]) if arts else None


def cache_top1(cands: list[dict[str, Any]]) -> tuple[str, str] | None:
    if not cands:
        return None
    ref = str(cands[0].get("article_ref", "")).strip()
    return article_key(ref) if ref else None


def score_gap(cands: list[dict[str, Any]], current: tuple[str, str] | None) -> tuple[float, float, float]:
    if not cands:
        return 0.0, 0.0, 0.0
    top = float(cands[0].get("score", 0) or 0)
    cur = 0.0
    if current:
        for c in cands:
            ref = str(c.get("article_ref", "")).strip()
            if ref and article_key(ref) == current:
                cur = float(c.get("score", 0) or 0)
                break
    gap = top - cur
    ratio = gap / top if top > 0 else 0.0
    return top, gap, ratio


def f2(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> float:
    if not gold:
        return 0.0
    if not pred:
        return 0.0
    prec = len(pred & gold) / len(pred)
    rec = len(pred & gold) / len(gold)
    if prec == 0 and rec == 0:
        return 0.0
    return 5 * prec * rec / (4 * prec + rec)


def neighbor_proxy(
    question: str,
    train: list[tuple[str, set[str], set[tuple[str, str]], set[str]]],
    threshold: float,
) -> tuple[float, set[tuple[str, str]], set[str]] | None:
    qtok = tokens(question)
    best = max(((token_f1(qtok, tt), arts, docs) for _q, tt, arts, docs in train), key=lambda x: x[0], default=(0.0, set(), set()))
    if best[0] < threshold:
        return None
    return best[0], best[1], best[2]


def pick_winner(
    sub_key: tuple[str, str] | None,
    cache_key: tuple[str, str] | None,
    gold_arts: set[tuple[str, str]],
) -> str:
    if not gold_arts:
        return "unknown"
    sub_hit = sub_key in gold_arts if sub_key else False
    cache_hit = cache_key in gold_arts if cache_key else False
    sub_law = sub_key[0] in {g[0] for g in gold_arts} if sub_key else False
    cache_law = cache_key[0] in {g[0] for g in gold_arts} if cache_key else False
    if sub_hit and not cache_hit:
        return "submission"
    if cache_hit and not sub_hit:
        return "cache"
    if sub_law and not cache_law:
        return "submission"
    if cache_law and not sub_law:
        return "cache"
    if sub_hit and cache_hit:
        return "tie"
    return "unknown"


def simulate_rule(
    mismatches: list[dict[str, Any]],
    *,
    same_law_only: bool = False,
    min_ratio: float = 0.0,
    proxy_winner: str | None = "cache",
    min_sim: float = 0.72,
) -> dict[str, Any]:
    swapped = 0
    proxy_hits = 0
    for m in mismatches:
        if m["sim"] is None or m["sim"] < min_sim:
            continue
        if proxy_winner and m["winner"] != proxy_winner:
            continue
        if same_law_only and not m["same_law"]:
            continue
        if m["gap_ratio"] < min_ratio:
            continue
        swapped += 1
        if m["winner"] == "cache":
            proxy_hits += 1
    return {"swapped": swapped, "proxy_cache_wins": proxy_hits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--neighbor-threshold", type=float, default=0.72)
    parser.add_argument("--output", default=str(OUT_DIR / "mismatch_top1_audit.json"))
    args = parser.parse_args()

    sub = load_submission(Path(args.base))
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    train = load_train()
    mapping = load_law_title_mapping(REPO_ROOT / "data/law_id_to_title_merged.json")

    mismatches: list[dict[str, Any]] = []
    winner_counts = Counter()
    same_law = 0
    cross_law = 0

    for rid, row in sub.items():
        sk = top1_key(row)
        cands = cache.get(rid, [])
        ck = cache_top1(cands)
        if sk == ck:
            continue
        top_s, gap, ratio = score_gap(cands, sk)
        sl = sk and ck and canonical_law_id(sk[0]) == canonical_law_id(ck[0])
        if sl:
            same_law += 1
        else:
            cross_law += 1

        proxy = neighbor_proxy(row.get("question", ""), train, args.neighbor_threshold)
        sim = proxy[0] if proxy else None
        gold_arts = proxy[1] if proxy else set()
        winner = pick_winner(sk, ck, gold_arts) if proxy else "no_neighbor"
        winner_counts[winner] += 1

        mismatches.append(
            {
                "id": rid,
                "same_law": sl,
                "sub_top1": sk,
                "cache_top1": ck,
                "gap": round(gap, 4),
                "gap_ratio": round(ratio, 4),
                "sim": round(sim, 4) if sim else None,
                "winner": winner,
                "gold_proxy": sorted(gold_arts)[:5],
            }
        )

    # Proxy F2 on neighbor-covered mismatches only
    sub_scores = []
    cache_scores = []
    for m in mismatches:
        if m["sim"] is None:
            continue
        gold = set(tuple(x) for x in m["gold_proxy"])
        if not gold:
            continue
        sub_pred = {m["sub_top1"]} if m["sub_top1"] else set()
        cache_pred = {m["cache_top1"]} if m["cache_top1"] else set()
        sub_scores.append(f2(sub_pred, gold))
        cache_scores.append(f2(cache_pred, gold))

    rules = []
    for name, kwargs in [
        ("same_law_proxy_cache_r015", dict(same_law_only=True, min_ratio=0.15, proxy_winner="cache")),
        ("same_law_proxy_cache_r025", dict(same_law_only=True, min_ratio=0.25, proxy_winner="cache")),
        ("same_law_all", dict(same_law_only=True, min_ratio=0.0, proxy_winner=None)),
        ("cross_law_proxy_cache_r015", dict(same_law_only=False, min_ratio=0.15, proxy_winner="cache")),
        ("proxy_cache_sim075", dict(same_law_only=False, min_ratio=0.0, proxy_winner="cache", min_sim=0.75)),
    ]:
        rules.append({"name": name, **simulate_rule(mismatches, **kwargs)})

    report = {
        "base": Path(args.base).name,
        "cache": Path(args.cache).name,
        "total_rows": len(sub),
        "top1_mismatch": len(mismatches),
        "same_law_mismatch": same_law,
        "cross_law_mismatch": cross_law,
        "neighbor_threshold": args.neighbor_threshold,
        "winner_on_mismatches": dict(winner_counts),
        "proxy_f2_neighbor_covered": {
            "rows": len(sub_scores),
            "submission_top1": round(sum(sub_scores) / len(sub_scores), 4) if sub_scores else None,
            "cache_top1": round(sum(cache_scores) / len(cache_scores), 4) if cache_scores else None,
        },
        "swap_rules": rules,
        "sample_cache_wins": [
            m for m in sorted(mismatches, key=lambda x: -(x["sim"] or 0)) if m["winner"] == "cache"
        ][:20],
        "sample_sub_wins": [
            m for m in sorted(mismatches, key=lambda x: -(x["sim"] or 0)) if m["winner"] == "submission"
        ][:20],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["id", "same_law", "winner", "sim", "gap_ratio", "sub_top1", "cache_top1"],
        )
        w.writeheader()
        for m in mismatches:
            w.writerow(
                {
                    "id": m["id"],
                    "same_law": m["same_law"],
                    "winner": m["winner"],
                    "sim": m["sim"],
                    "gap_ratio": m["gap_ratio"],
                    "sub_top1": m["sub_top1"],
                    "cache_top1": m["cache_top1"],
                }
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {out} and {csv_path}")


if __name__ == "__main__":
    main()
