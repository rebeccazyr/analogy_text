#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
RUN_DIR="${REPO_DIR}/runs/recomputed"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Create it with: python3 -m venv .venv" >&2
  exit 1
fi

if [[ -z "${TOGETHER_API_KEY:-}" && ! -f "${REPO_DIR}/.env" ]]; then
  echo "TOGETHER_API_KEY is missing. Copy .env.example to .env and set it." >&2
  exit 1
fi

mkdir -p \
  "${RUN_DIR}/target_coverage" \
  "${RUN_DIR}/mapping_strength" \
  "${RUN_DIR}/metaphoricity"
cd "${REPO_DIR}"

EXTRA_ARGS=()
if [[ "${REFRESH_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--refresh-cache)
fi

"${PYTHON_BIN}" run_text_agents.py \
  --mode target-coverage \
  --split test \
  --sample-concurrency 12 \
  --max-concurrency 12 \
  --reasoning-effort medium \
  --output-dir "${RUN_DIR}/target_coverage" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" run_text_agents.py \
  --mode mapping-strength-native \
  --split test \
  --sample-concurrency 12 \
  --max-concurrency 12 \
  --reasoning-effort medium \
  --output-dir "${RUN_DIR}/mapping_strength" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" run_text_agents.py \
  --mode m-evidence-reconciled \
  --split test \
  --sample-concurrency 12 \
  --max-concurrency 12 \
  --reasoning-effort high \
  --output-dir "${RUN_DIR}/metaphoricity" \
  "${EXTRA_ARGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/combine_recomputed_metrics.py" \
  --tcc "${RUN_DIR}/target_coverage/test_target_coverage_predictions.csv" \
  --ms "${RUN_DIR}/mapping_strength/test_mapping_strength_predictions.csv" \
  --metaphoricity "${RUN_DIR}/metaphoricity/test_metaphoricity_predictions.csv" \
  --output "${RUN_DIR}/submission.csv"
