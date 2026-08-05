#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

M_FEATURE_SET="${M_FEATURE_SET:-all}"
M_FEATURE_THRESHOLD="${M_FEATURE_THRESHOLD:-}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-20}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-40}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
MAX_TOKENS="${MAX_TOKENS:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/m_feature_ablation_${M_FEATURE_SET}}"

if [[ "${PYTHON_BIN}" == */* ]]; then
  RESOLVED_PYTHON_BIN="${PYTHON_BIN}"
else
  RESOLVED_PYTHON_BIN="$(command -v "${PYTHON_BIN}" || true)"
fi

if [[ -z "${RESOLVED_PYTHON_BIN}" || ! -x "${RESOLVED_PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN or create .venv and install requirements.txt." >&2
  exit 1
fi

if [[ -z "${TOGETHER_API_KEY:-}" && ! -f "${SCRIPT_DIR}/.env" ]]; then
  echo "TOGETHER_API_KEY is missing. Export it or create ${SCRIPT_DIR}/.env." >&2
  exit 1
fi

echo "Running M feature ablation"
echo "  experiments: ${M_FEATURE_SET}"
echo "  threshold override: ${M_FEATURE_THRESHOLD:-pre-registered defaults}"
echo "  embedding: ${EMBEDDING_MODEL} on ${EMBEDDING_DEVICE}"
echo "  concurrency: samples=${SAMPLE_CONCURRENCY}, api=${MAX_CONCURRENCY}"
echo "  LLM: reasoning=${REASONING_EFFORT}, max_tokens=${MAX_TOKENS}"
echo "  output: ${OUTPUT_DIR}"

ARGS=(
  --mode m-features
  --split validation
  --m-feature-set "${M_FEATURE_SET}"
  --model openai/gpt-oss-120b
  --reasoning-effort "${REASONING_EFFORT}"
  --max-tokens "${MAX_TOKENS}"
  --embedding-model "${EMBEDDING_MODEL}"
  --embedding-device "${EMBEDDING_DEVICE}"
  --sample-concurrency "${SAMPLE_CONCURRENCY}"
  --max-concurrency "${MAX_CONCURRENCY}"
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${M_FEATURE_THRESHOLD}" ]]; then
  ARGS+=(--m-feature-threshold "${M_FEATURE_THRESHOLD}")
fi

cd "${SCRIPT_DIR}"
"${RESOLVED_PYTHON_BIN}" run_text_agents.py "${ARGS[@]}"

echo "Scores: ${OUTPUT_DIR}/validation_m_feature_scores.json"
echo "Predictions: ${OUTPUT_DIR}/validation_m_feature_predictions.csv"
echo "Details: ${OUTPUT_DIR}/validation_m_feature_details.jsonl"
