#!/usr/bin/env bash
# Build trial hybrid submission: partial rrf_wide cache + no_wl fallback.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=venv/bin/python

$PY tools/submission/merge_live_caches.py \
  --primary data/augmented/live_retrieval_rrf_wide_merged.json \
  --fallback data/augmented/live_retrieval_no_wl_merged.json \
  --output data/augmented/live_rrf_hybrid.json

$PY tools/submission/create_recall_boost_submission.py \
  --base submission.zip \
  --output submission_variants/rrf_hybrid_v1.zip \
  --debug submission_variants/rrf_hybrid_v1_debug.csv \
  --cap-articles 2 --cap-docs 1 --article-same-law-only \
  --article-min-score 0.9 --article-min-gap-from-top1 0.03 \
  --live-cache data/augmented/live_rrf_hybrid.json \
  --mapping data/law_id_to_title_merged.json

echo "DONE -> submission_variants/rrf_hybrid_v1.zip"
