#!/usr/bin/env bash
# Tái hiện đầy đủ bài nộp R2AI: R2AIStage1DATA.json → submission.zip + submission_qa.zip (2000 dòng mỗi file).
#
# Yêu cầu: venv, corpus merged, BM25 index, Qdrant collection merged, Ollama qwen3:4b-instruct.
# Xem: docs/04_HUONG_DAN_TAI_HIEN_2000_CAU.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-venv/bin/python}"
LOG="${LOG:-/tmp/r2ai_full_2000_pipeline.log}"

SKIP_CACHE="${SKIP_CACHE:-0}"
SKIP_IR="${SKIP_IR:-0}"
SKIP_QA="${SKIP_QA:-0}"

INPUT="${INPUT:-R2AIStage1DATA.json}"
MAPPING="${MAPPING:-data/law_id_to_title_merged.json}"
CORPUS="${CORPUS:-data/corpus/legal_corpus_merged.json}"
CACHE="${CACHE:-data/augmented/live_retrieval_rrf_wide_merged.json}"

SEED_ZIP="${SEED_ZIP:-submission_variants/submission_cache_seed_top1.zip}"
NO_WL_ZIP="${NO_WL_ZIP:-submission_variants/submission_recall_boost_merged_vn_rerank_no_wl_tight_v1.zip}"
IR_ZIP="${IR_ZIP:-submission_variants/rrf_swap_g008.zip}"
QA_ZIP="${QA_ZIP:-submission_variants/qa_promote_g008_ollama.zip}"

export USE_MERGED_CORPUS=1
export HYBRID_FUSION=rrf
export USE_WIDE_RETRIEVAL_POOL=1
export ENABLE_RERANKING="${ENABLE_RERANKING:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if [[ "${ENABLE_RERANKING}" != "1" ]]; then
  log "WARNING: ENABLE_RERANKING is not 1 — IR will NOT match ~0.631. Export ENABLE_RERANKING=1 before continuing."
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1" >&2
    exit 1
  fi
}

log "=== R2AI full 2000 pipeline ==="
log "ROOT=$ROOT"
log "LOG=$LOG"

require_file "$INPUT"
require_file "$MAPPING"
require_file "$CORPUS"
require_file "$PY"

mkdir -p submission_variants data/augmented

if [[ "$SKIP_CACHE" != "1" ]]; then
  log "Step 1/5: cache retrieval for 2000 questions → $CACHE"
  "$PY" tools/submission/cache_live_retrieval.py \
    --input "$INPUT" \
    --output "$CACHE" \
    --mapping "$MAPPING" \
    --resume \
    2>&1 | tee -a "$LOG"
else
  log "Step 1/5: SKIP cache (SKIP_CACHE=1)"
  require_file "$CACHE"
fi

if [[ "$SKIP_IR" != "1" ]]; then
  log "Step 2/5: seed submission from cache (no prior submission.zip needed)"
  "$PY" tools/submission/create_cache_only_submission.py \
    --input "$INPUT" \
    --cache "$CACHE" \
    --output "$SEED_ZIP" \
    --cap-articles 1 \
    --cap-docs 1 \
    2>&1 | tee -a "$LOG"

  log "Step 3/5: recall boost → $NO_WL_ZIP"
  "$PY" tools/submission/create_recall_boost_submission.py \
    --base "$SEED_ZIP" \
    --output "$NO_WL_ZIP" \
    --debug "${NO_WL_ZIP%.zip}_debug.csv" \
    --cap-articles 2 --cap-docs 1 --article-same-law-only \
    --article-min-score 0.9 --article-min-gap-from-top1 0.03 \
    --live-cache "$CACHE" \
    --mapping "$MAPPING" \
    2>&1 | tee -a "$LOG"

  log "Step 4/5: zone swap → $IR_ZIP"
  NO_WL_JSON="${NO_WL_ZIP%.zip}.json"
  require_file "$NO_WL_JSON"
  "$PY" tools/submission/create_rrf_zone_swap_submission.py \
    --base "$NO_WL_JSON" \
    --output "$IR_ZIP" \
    --debug "${IR_ZIP%.zip}_debug.csv" \
    --zone-cache "$CACHE" \
    --mapping "$MAPPING" \
    --replace-min-gap 0.03 \
    --article-min-score 0.9 \
    2>&1 | tee -a "$LOG"

  cp "$IR_ZIP" submission.zip
  log "Copied $IR_ZIP → submission.zip"
else
  log "Step 2-4/5: SKIP IR build (SKIP_IR=1)"
  require_file submission.zip
fi

if [[ "$SKIP_QA" != "1" ]]; then
  log "Step 5/5: grounded QA answers → $QA_ZIP"
  export OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3:4b-instruct}"
  export OLLAMA_WORKERS="${OLLAMA_WORKERS:-6}"
  "$PY" tools/submission/create_qa_submission.py \
    --base submission.zip \
    --corpus "$CORPUS" \
    --backend ollama \
    --model "$OLLAMA_MODEL" \
    --batch-size 8 \
    --max-articles 3 \
    --max-chars-per-article 1200 \
    --max-new-tokens 1200 \
    --resume \
    --output "$QA_ZIP" \
    --debug "${QA_ZIP%.zip}_debug.csv" \
    2>&1 | tee -a "$LOG"

  cp "$QA_ZIP" submission_qa.zip
  log "Copied $QA_ZIP → submission_qa.zip"
else
  log "Step 5/5: SKIP QA (SKIP_QA=1)"
fi

log "Verify row counts..."
"$PY" <<'PY'
import json, zipfile, sys
for name in ("submission.zip", "submission_qa.zip"):
    try:
        rows = json.loads(zipfile.ZipFile(name).read("results.json"))
    except FileNotFoundError:
        print(f"SKIP verify: {name} not found")
        continue
    n = len(rows)
    arts = sum(1 for r in rows if r.get("relevant_articles"))
    ans = sum(1 for r in rows if r.get("answer"))
    print(f"{name}: total={n}  with_articles={arts}  with_answer={ans}")
    if n != 2000:
        sys.exit(f"{name}: expected 2000 rows, got {n}")
print("OK: 2000 rows verified")
PY

log "=== DONE ==="
log "IR:  submission.zip"
log "QA:  submission_qa.zip"
log "Full log: $LOG"
