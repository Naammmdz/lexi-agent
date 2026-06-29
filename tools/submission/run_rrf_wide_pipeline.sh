#!/usr/bin/env bash
# Re-cache + build tight submission with RRF hybrid fusion + wide retrieval pool.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=venv/bin/python
export USE_MERGED_CORPUS=1
export HYBRID_FUSION=rrf
export USE_WIDE_RETRIEVAL_POOL=1

CACHE=data/augmented/live_retrieval_rrf_wide_merged.json
NAME=rrf_wide_tight_v1

log() { echo "$1" | tee -a /tmp/rrf_wide_pipeline.log; }

log "=== cache start $(date) ==="
$PY tools/submission/cache_live_retrieval.py \
  --output "$CACHE" \
  --mapping data/law_id_to_title_merged.json \
  --resume \
  >> /tmp/rrf_wide_cache.log 2>&1
code=$?
if [ "$code" -ne 0 ]; then
  log "CACHE FAILED exit=$code $(date)"
  exit "$code"
fi
log "=== cache done $(date) ==="

log "=== submission $NAME start $(date) ==="
$PY tools/submission/create_recall_boost_submission.py \
  --base submission.zip \
  --output "submission_variants/submission_recall_boost_merged_vn_rerank_${NAME}.zip" \
  --debug "submission_variants/submission_recall_boost_merged_vn_rerank_${NAME}_debug.csv" \
  --cap-articles 2 --cap-docs 1 --article-same-law-only \
  --article-min-score 0.9 --article-min-gap-from-top1 0.03 \
  --live-cache "$CACHE" \
  --mapping data/law_id_to_title_merged.json \
  >> "/tmp/${NAME}.log" 2>&1
log "=== submission $NAME done $(date) ==="

$PY tools/submission/benchmark_companion_candidate.py \
  "submission_variants/submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.zip" \
  "submission_variants/submission_recall_boost_merged_vn_rerank_${NAME}.zip" \
  --output submission_variants/local_benchmark/rrf_wide_companion_audit.json

log "RRF_WIDE_PIPELINE_DONE $(date)"
