#!/usr/bin/env bash
# Đóng gói dữ liệu để upload lên Google Drive / OneDrive (nghiệm thu R2AI)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/r2ai-data-bundle"
STAMP="$(date +%Y%m%d)"

rm -rf "$OUT"
mkdir -p "$OUT/data/corpus" "$OUT/data/utils" "$OUT/index"

copy_if() {
  local src="$1" dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "  + $(basename "$src")"
  else
    echo "  ! MISSING: $src" >&2
  fi
}

echo "Packing data bundle → $OUT"
copy_if "$ROOT/data/corpus/legal_corpus_merged.json" "$OUT/data/corpus/"
copy_if "$ROOT/data/law_id_to_title_merged.json" "$OUT/data/"
copy_if "$ROOT/data/utils/stopwords.txt" "$OUT/data/utils/"
copy_if "$ROOT/index/bm25_index_merged.pkl" "$OUT/index/"
copy_if "$ROOT/R2AIStage1DATA.json" "$OUT/"
copy_if "$ROOT/submission.zip" "$OUT/"
copy_if "$ROOT/submission_qa.zip" "$OUT/"
cp "$ROOT/docs/01_MO_TA_DU_LIEU.md" "$OUT/README_DATA.md"

ARCHIVE="${ROOT}/r2ai-data-bundle-${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$ROOT" r2ai-data-bundle
echo ""
echo "Done: $ARCHIVE"
du -sh "$ARCHIVE"
