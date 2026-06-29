#!/usr/bin/env bash
# P4 corpus expansion pipeline: audit → gap fill → merge → re-index → benchmark gate.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-venv/bin/python}"

echo "=== P4 Step 1: Audit ==="
$PY tools/corpus/audit_corpus_gaps.py

echo ""
echo "=== P4 Step 2: HF vbpl gap (th1nhng0/vietnamese-legal-documents) ==="
if [[ ! -f data/augmented/hf_vbpl_gap_corpus.json ]] || [[ "${P4_HF_REBUILD:-0}" == "1" ]]; then
  $PY tools/corpus/build_hf_vbpl_gap_corpus.py
else
  echo "hf_vbpl_gap_corpus.json exists — skip (P4_HF_REBUILD=1 to rebuild)"
fi

echo ""
echo "=== P4 Step 3: Legacy DB gap (optional, Postgres dev) ==="
if nc -z localhost 5435 2>/dev/null; then
  $PY tools/submission/create_augmented_corpus_from_db.py
  $PY tools/corpus/build_db_gap_corpus.py
else
  echo "Postgres :5435 not available — skip legacy DB gap"
fi

echo ""
echo "=== P4 Step 4: Merge Zalo + HF (+ legacy gaps if present) ==="
GAP_ARGS=(--gap data/augmented/hf_vbpl_gap_corpus.json)
if [[ -f data/augmented/db_gap_corpus.json ]]; then
  GAP_ARGS+=(--gap data/augmented/db_gap_corpus.json)
fi
if [[ -f data/augmented/vlsp_bulk_gap_corpus.json ]]; then
  GAP_ARGS+=(--gap data/augmented/vlsp_bulk_gap_corpus.json)
elif [[ -f data/augmented/vlsp_gap_corpus.json ]]; then
  GAP_ARGS+=(--gap data/augmented/vlsp_gap_corpus.json)
fi
$PY tools/corpus/build_merged_corpus_v2.py "${GAP_ARGS[@]}"

echo ""
echo "=== P4 Step 5: Re-index Qdrant + BM25 ==="
USE_MERGED_CORPUS=1 $PY run_indexing.py --rebuild

echo ""
echo "=== P4 Step 6: Train benchmark gate (100q) ==="
USE_MERGED_CORPUS=1 USE_WIDE_RETRIEVAL_POOL=1 HYBRID_FUSION=rrf \
  $PY tools/submission/run_retrieval_tune_benchmark.py \
  --configs compare_v3 --max-questions 100 \
  --output submission_variants/local_benchmark/p4_corpus_v2_100q.json

echo ""
echo "P4 pipeline done. Check p4_corpus_v2_100q.json for gate Δ≥0.01 before re-cache + submit."
