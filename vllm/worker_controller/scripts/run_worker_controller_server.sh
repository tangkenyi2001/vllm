#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
PYTHON="$REPO_ROOT/.venv/bin/python"

PORT=8000

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/workerlogs.log"

cd "$REPO_ROOT"

# Kill any process already holding the port so the new server can bind
if lsof -ti "tcp:${PORT}" > /dev/null 2>&1; then
    echo "Port ${PORT} is in use. Killing existing process(es)..."
    lsof -ti "tcp:${PORT}" | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Cap KV cache to 16 GiB per GPU (one entry per TP rank) to match std_server
export VLLM_KVC_MEM_GB="16,16"

echo "Logging to: $LOG_FILE"
"$PYTHON" -m vllm.worker_controller.tests.worker_controller_server 2>&1 | tee -a "$LOG_FILE"
