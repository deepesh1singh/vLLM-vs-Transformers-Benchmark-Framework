#!/usr/bin/env bash

set -euo pipefail

if grep -qi microsoft /proc/sys/kernel/osrelease; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
fi

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen25-3b}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
API_KEY="${VLLM_API_KEY:-local-vllm-key}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --dtype float16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --generation-config vllm