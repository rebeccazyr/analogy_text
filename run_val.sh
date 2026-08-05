#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

M_CONCEPT_WEIGHT="${M_CONCEPT_WEIGHT:-0.5}"
M_COSINE_THRESHOLD="${M_COSINE_THRESHOLD:-0.35}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"
SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
WEIGHT_TAG="${M_CONCEPT_WEIGHT//./p}"
THRESHOLD_TAG="${M_COSINE_THRESHOLD//./p}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/m_cosine_validation_w${WEIGHT_TAG}_t${THRESHOLD_TAG}}"

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

echo "Running M cosine validation"
echo "  concept weight: ${M_CONCEPT_WEIGHT}"
echo "  domain weight: 1 - ${M_CONCEPT_WEIGHT}"
echo "  threshold: ${M_COSINE_THRESHOLD}"
echo "  embedding: ${EMBEDDING_MODEL} on ${EMBEDDING_DEVICE}"
echo "  concurrency: samples=${SAMPLE_CONCURRENCY}, api=${MAX_CONCURRENCY}"
echo "  output: ${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"
"${RESOLVED_PYTHON_BIN}" run_text_agents.py \
  --mode m-cosine \
  --split validation \
  --model openai/gpt-oss-120b \
  --reasoning-effort high \
  --embedding-model "${EMBEDDING_MODEL}" \
  --embedding-device "${EMBEDDING_DEVICE}" \
  --m-concept-weight "${M_CONCEPT_WEIGHT}" \
  --m-cosine-threshold "${M_COSINE_THRESHOLD}" \
  --sample-concurrency "${SAMPLE_CONCURRENCY}" \
  --max-concurrency "${MAX_CONCURRENCY}" \
  --output-dir "${OUTPUT_DIR}"

echo "Scores: ${OUTPUT_DIR}/validation_m_scores.json"
echo "Details: ${OUTPUT_DIR}/validation_metaphoricity_details.jsonl"
