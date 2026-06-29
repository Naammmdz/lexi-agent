#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=venv/bin/python
export USE_MERGED_CORPUS=1

log() { echo "$1" | tee -a /tmp/vn_rerank_tuned_pipeline.log; }

log "=== cache start $(date) ==="
$PY tools/submission/cache_live_retrieval.py \
  --output data/augmented/live_retrieval_vn_rerank_tuned_merged.json \
  --mapping data/law_id_to_title_merged.json \
  >> /tmp/vn_rerank_tuned_cache.log 2>&1
code=$?
if [ "$code" -ne 0 ]; then
  log "CACHE FAILED exit=$code $(date)"
  exit "$code"
fi
log "=== cache done $(date) ==="

for name in top1_g005 top1_g003 s085_g003; do
  log "=== submission $name start $(date) ==="
  if [ "$name" = s085_g003 ]; then
    EXTRA=(--article-min-score 0.85 --article-min-gap-from-top1 0.03)
  elif [ "$name" = top1_g005 ]; then
    EXTRA=(--article-min-score 0.9 --article-min-gap-from-top1 0.03 --prefer-cache-top1 --replace-top1-min-gap 0.05)
  else
    EXTRA=(--article-min-score 0.9 --article-min-gap-from-top1 0.03 --prefer-cache-top1 --replace-top1-min-gap 0.03)
  fi
  $PY tools/submission/create_recall_boost_submission.py \
    --base submission.zip \
    --output "submission_variants/submission_recall_boost_merged_vn_rerank_tuned_${name}.zip" \
    --debug "submission_variants/submission_recall_boost_merged_vn_rerank_tuned_${name}_debug.csv" \
    --cap-articles 2 --cap-docs 1 --article-same-law-only \
    --live-cache data/augmented/live_retrieval_vn_rerank_tuned_merged.json \
    --mapping data/law_id_to_title_merged.json \
    "${EXTRA[@]}" \
    >> "/tmp/tuned_${name}.log" 2>&1
  log "=== submission $name done $(date) ==="
done

log "TUNED_PIPELINE_DONE $(date)"
