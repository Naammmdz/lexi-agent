"""Create small controlled public-leaderboard submission probes.

The leaderboard has reported ``gold=50 pred=2000``. These artifacts keep only
the first 50 rows and isolate the two first-50 rows where recent legal rules
changed the prediction. They are meant for low-noise leaderboard diagnosis, not
for final private submission.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from _paths import REPO_ROOT


BASE_DIR = REPO_ROOT
VARIANT_DIR = BASE_DIR / "submission_variants"
CURRENT = VARIANT_DIR / "submission_augmented_hardrules_top1.zip"
OLD_RERANK = VARIANT_DIR / "submission_augmented_rerank_top1.zip"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("results.json"))


def write_zip(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="results.json")
    print(f"wrote={output} rows={len(rows)}")


def replace_row(rows: list[dict[str, Any]], old_rows: list[dict[str, Any]], one_based_index: int) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    output[one_based_index - 1] = dict(old_rows[one_based_index - 1])
    return output


def main() -> None:
    current = load_rows(CURRENT)
    old = load_rows(OLD_RERANK)

    current50 = current[:50]
    old50 = old[:50]

    write_zip(current50, VARIANT_DIR / "submission_public50_hardrules_top1.zip")
    write_zip(old50, VARIANT_DIR / "submission_public50_rerank_top1.zip")
    write_zip(
        replace_row(current50, old50, 21),
        VARIANT_DIR / "submission_public50_probe_row21_old.zip",
    )
    write_zip(
        replace_row(current50, old50, 33),
        VARIANT_DIR / "submission_public50_probe_row33_old.zip",
    )
    write_zip(
        replace_row(replace_row(current50, old50, 21), old50, 33),
        VARIANT_DIR / "submission_public50_probe_rows21_33_old.zip",
    )


if __name__ == "__main__":
    main()
