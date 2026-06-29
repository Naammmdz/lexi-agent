#!/usr/bin/env python3
"""Merge partial primary cache with fallback cache (primary wins on overlap)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _paths import REPO_ROOT


def merge_caches(primary: Path, fallback: Path, output: Path) -> dict[str, int]:
    primary_data = json.loads(primary.read_text(encoding="utf-8"))
    fallback_data = json.loads(fallback.read_text(encoding="utf-8"))

    merged = dict(fallback_data)
    merged.update(primary_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")

    return {
        "primary": len(primary_data),
        "fallback": len(fallback_data),
        "merged": len(merged),
        "from_primary_only": len(set(primary_data) - set(fallback_data)),
        "overlap": len(set(primary_data) & set(fallback_data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, help="Higher-priority cache (e.g. partial rrf_wide)")
    parser.add_argument("--fallback", required=True, help="Fallback cache (e.g. no_wl full)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stats = merge_caches(Path(args.primary), Path(args.fallback), Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
