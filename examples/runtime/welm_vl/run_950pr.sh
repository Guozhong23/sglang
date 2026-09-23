#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the complete WeLM-VL checkpoint directory}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

# Start with the ordinary TP path; custom text-model performance switches from
# an existing shell must not silently change this first VL validation.
export WELM_NPU_USE_MEGAMOE=0
export WELM_NPU_USE_FLASH_ATTN=0
export SGLANG_NPU_WELMV4_USE_FUSED_TOPK=0
export SGLANG_NPU_WELMV4_FUSED_QKV=0
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0
export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=0
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3000}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-3000}

TP_SIZE=${TP_SIZE:-4}
BASE_DEVICE=${BASE_DEVICE:-0}
preflight=("${SCRIPT_DIR}/check_model.py" "${MODEL_PATH}" --runtime --tp "${TP_SIZE}" --base-device "${BASE_DEVICE}")
extra_args=()
if [[ -n ${CHAT_TEMPLATE:-} ]]; then
    preflight+=(--chat-template "${CHAT_TEMPLATE}")
    extra_args+=(--chat-template "${CHAT_TEMPLATE}")
fi
"${PYTHON_BIN}" "${preflight[@]}"

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --served-model-name "${SERVED_MODEL_NAME:-welmv45-vl}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-6677}" \
    --device npu \
    --dtype bfloat16 \
    --tp-size "${TP_SIZE}" \
    --ep-size 1 \
    --base-gpu-id "${BASE_DEVICE}" \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --moe-a2a-backend none \
    --enable-multimodal \
    --enable-over-encoding \
    --context-length "${CONTEXT_LENGTH:-8192}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-8192}" \
    --max-running-requests 1 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
    --page-size 64 \
    --disable-cuda-graph \
    --disable-prefill-cuda-graph \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --disable-overlap-schedule \
    "${extra_args[@]}" "$@"
