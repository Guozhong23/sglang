#!/usr/bin/env bash
# Run after sourcing the server's CANN, torch_npu and optional DeepEP environment.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../.." && pwd)
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python3}
: "${MODEL_PATH:?Set MODEL_PATH to the complete welm_moonvit_vl checkpoint}"
TP_SIZE=${TP_SIZE:-4}
MOE_A2A_BACKEND=${MOE_A2A_BACKEND:-none}

# First establish an eager correctness baseline; no old WeLM ViT kernels.
export SGLANG_VLM_VISION_USE_ULYSSES=0
export SGLANG_VLM_VISION_USE_DP=0
export SGLANG_VLM_NPU_USE_MC2=0
export SGLANG_VLM_NPU_FUSE_GELU=0
export SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM=0
export SGLANG_VLM_NPU_FUSE_ROPE=0
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0
export WELM_NPU_USE_MEGAMOE=0
export WELM_NPU_MEGAMOE_FUSE_SHARED_EXPERT=0
export WELM_NPU_USE_FLASH_ATTN=0
export SGLANG_NPU_WELMV4_FUSED_QKV=0
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=0
export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=0
export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=0
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=0

case "${MOE_A2A_BACKEND}" in
    none) ep_args=(--ep-size 1 --moe-a2a-backend none) ;;
    deepep)
        export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1
        export SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
        ep_args=(--ep-size "${TP_SIZE}" --moe-a2a-backend deepep --deepep-mode auto)
        ;;
    *) echo 'MOE_A2A_BACKEND must be none or deepep' >&2; exit 2 ;;
esac

"${PYTHON_BIN}" "${script_dir}/check_moonvit.py" "${MODEL_PATH}" --runtime
exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --trust-remote-code \
    --device npu --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --tp-size "${TP_SIZE}" "${ep_args[@]}" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-7788}" \
    --served-model-name "${SERVED_MODEL_NAME:-welmv45-moonvit-vl}" \
    --attention-backend ascend --mm-attention-backend ascend_attn \
    --enable-multimodal --enable-over-encoding --enable-kv-mirror \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --context-length "${CONTEXT_LENGTH:-8192}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-8192}" \
    --max-running-requests 1 --chunked-prefill-size -1 \
    --page-size "${PAGE_SIZE:-64}" \
    --disable-prefill-cuda-graph --disable-cuda-graph \
    --disable-overlap-schedule --disable-radix-cache \
    "$@"
