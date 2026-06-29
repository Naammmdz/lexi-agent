"""Normalize an existing R2AI results.json and rebuild a flat submission zip."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from _paths import REPO_ROOT
from utils.submission_formatter import load_law_title_mapping, normalize_submission_rows


BASE_DIR = REPO_ROOT


def main() -> None:
    results_path = BASE_DIR / "results.json"
    mapping_path = BASE_DIR / "data" / "law_id_to_title.json"
    zip_path = BASE_DIR / "submission.zip"

    rows = json.loads(results_path.read_text(encoding="utf-8"))
    mapping = load_law_title_mapping(mapping_path)
    normalized, stats = normalize_submission_rows(rows, mapping)

    results_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {results_path}")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
