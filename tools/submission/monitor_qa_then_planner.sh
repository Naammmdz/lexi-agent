#!/usr/bin/env bash
# Monitor QA batch; on DONE deploy planner benchmark on GPU server.
set -euo pipefail

SSH_HOST="admin@103.9.159.87"
SSH_PORT="31703"
SSH_PASS="${SSHPASS:-IwoMv8x8bbDI}"
REMOTE_ROOT="/data/home/admin/vietnamese-legal-chatbot"
LOG="/data/home/admin/qa_ollama.log"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLANNER_LOG="/data/home/admin/planner_benchmark.log"
PLANNER_MODEL="${OLLAMA_PLANNER_MODEL:-qwen3:14b}"

ssh_cmd() {
  SSHPASS="$SSH_PASS" sshpass -e ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no "$SSH_HOST" "$@"
}

scp_file() {
  local src="$1" dst="$2"
  SSHPASS="$SSH_PASS" sshpass -e scp -P "$SSH_PORT" -o StrictHostKeyChecking=no "$src" "$SSH_HOST:$dst"
}

echo "[monitor] waiting for QA batch to finish..."
while true; do
  line="$(ssh_cmd "grep -E 'generated|DONE' $LOG 2>/dev/null | tail -1" || true)"
  running="$(ssh_cmd "pgrep -f create_qa_submission.py | head -1" || true)"
  echo "[monitor] $(date -Iseconds) $line"
  if echo "$line" | grep -q "DONE ->"; then
    echo "[monitor] QA DONE"
    break
  fi
  if echo "$line" | grep -q "1746/1746"; then
    sleep 30
    if ! ssh_cmd "pgrep -f create_qa_submission.py | grep -q ."; then
      echo "[monitor] QA complete (1746/1746, process ended)"
      break
    fi
  fi
  if [[ -z "$running" ]] && echo "$line" | grep -qE "generated [0-9]+/"; then
    pending="$(echo "$line" | sed -n 's/.*generated \([0-9]*\)\/\([0-9]*\).*/\1 \2/p')"
    done_n="${pending%% *}"
    total_n="${pending##* }"
    if [[ "$done_n" == "$total_n" && "$total_n" != "0" ]]; then
      echo "[monitor] QA complete (all pending done, no process)"
      break
    fi
  fi
  sleep 120
done

echo "[monitor] deploying planner files..."
scp_file "$LOCAL_ROOT/utils/llm_query_planner.py" "$REMOTE_ROOT/utils/"
scp_file "$LOCAL_ROOT/tools/submission/run_planner_retrieval_benchmark.py" "$REMOTE_ROOT/tools/submission/"

echo "[monitor] pulling planner model $PLANNER_MODEL (if missing)..."
ssh_cmd "ollama list | grep -q '${PLANNER_MODEL%%:*}' || get-model ollama $PLANNER_MODEL || true"

# fallback if 14b unavailable
if ! ssh_cmd "ollama list" | grep -q "$PLANNER_MODEL"; then
  PLANNER_MODEL="qwen3-vl:8b"
  echo "[monitor] fallback planner model: $PLANNER_MODEL"
fi

echo "[monitor] running planner benchmark (100q easy)..."
ssh_cmd "bash -s" <<REMOTE
set -e
cd $REMOTE_ROOT
source venv/bin/activate
export OLLAMA_PLANNER_MODEL=$PLANNER_MODEL
export OLLAMA_NUM_PARALLEL=4
nohup python tools/submission/run_planner_retrieval_benchmark.py \
  --max-questions 100 \
  --question-filter easy \
  --planner-model $PLANNER_MODEL \
  > $PLANNER_LOG 2>&1 &
echo planner_pid=\$!
REMOTE

echo "[monitor] planner benchmark started. Log: $PLANNER_LOG"
echo "[monitor] QA output: $REMOTE_ROOT/submission_variants/qa_promote_g008_ollama.zip"
echo "[monitor] DONE — submit qa_promote_g008_ollama.zip for QA promote"
