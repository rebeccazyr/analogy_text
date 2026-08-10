#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
RUN_DIR="${REPO_DIR}/runs/shared-recomputed"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Create it with: python3 -m venv .venv" >&2
  exit 1
fi

if [[ -z "${TOGETHER_API_KEY:-}" && ! -f "${REPO_DIR}/.env" ]]; then
  echo "TOGETHER_API_KEY is missing. Copy .env.example to .env and set it." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

EXTRA_ARGS=()
if [[ "${REFRESH_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--refresh-cache)
fi

"${PYTHON_BIN}" run_text_agents.py \
  --mode shared-active \
  --split "${SPLIT:-validation}" \
  --sample-concurrency 12 \
  --max-concurrency 12 \
  --max-retries 8 \
  --reasoning-effort medium \
  --m-reasoning-effort high \
  --shared-max-tokens 5000 \
  --output-dir "${RUN_DIR}" \
  "${EXTRA_ARGS[@]}"
