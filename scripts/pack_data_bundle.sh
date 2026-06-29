#!/usr/bin/env bash
# Đóng gói data/ + index/ để upload Google Drive / OneDrive (nghiệm thu R2AI)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/r2ai-data-bundle"
STAMP="$(date +%Y%m%d)"

rm -rf "$OUT"
mkdir -p "$OUT"

copy_tree() {
  local src="$1" dst="$2"
  if [[ -d "$src" ]]; then
    cp -R "$src" "$dst"
    echo "  + $(basename "$src")/ ($(du -sh "$src" | cut -f1))"
  else
    echo "  ! MISSING: $src" >&2
  fi
}

echo "Packing data + index → $OUT"
copy_tree "$ROOT/data" "$OUT/"
copy_tree "$ROOT/index" "$OUT/"

cp "$ROOT/docs/01_MO_TA_DU_LIEU.md" "$OUT/README_DATA.md"
cp "$ROOT/docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md" "$OUT/README_REPRODUCE.md"

ARCHIVE="${ROOT}/r2ai-data-bundle-${STAMP}.tar.gz"
echo ""
echo "Creating archive (may take a few minutes)..."
tar -czf "$ARCHIVE" -C "$ROOT" r2ai-data-bundle
echo "Done: $ARCHIVE"
du -sh "$ARCHIVE"
echo ""
echo "Upload to Drive: $ARCHIVE"
echo "Or upload folders directly: data/ (~1.6 GB) + index/ (~1.3 GB)"
