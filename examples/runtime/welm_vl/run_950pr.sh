#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the complete WeLM-VL checkpoint directory}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

TP_SIZE=${TP_SIZE:-4}
BASE_DEVICE=${BASE_DEVICE:-0}
WELM_VL_PROFILE=${WELM_VL_PROFILE:-optimized}
case "${TP_SIZE}" in
    1|2|4|8) ;;
    *) echo "WeLM-VL requires TP_SIZE=1, 2, 4 or 8." >&2; exit 2 ;;
esac
profile_args=()
case "${WELM_VL_PROFILE}" in
    optimized)
        if [[ ${TP_SIZE} == 1 ]]; then
            echo "The optimized DeepEP profile requires TP_SIZE=2, 4 or 8; use baseline for TP=1." >&2
            exit 2
        fi
        export WELM_NPU_USE_MEGAMOE=${WELM_NPU_USE_MEGAMOE:-1}
        export WELM_NPU_USE_FLASH_ATTN=${WELM_NPU_USE_FLASH_ATTN:-1}
        fused_qkv_default=0
        if [[ ${TP_SIZE} == 4 ]]; then fused_qkv_default=1; fi
        export SGLANG_NPU_WELMV4_FUSED_QKV=${SGLANG_NPU_WELMV4_FUSED_QKV:-${fused_qkv_default}}
        export SGLANG_NPU_USE_MULTI_STREAM=${SGLANG_NPU_USE_MULTI_STREAM:-1}
        export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=${SGLANG_DEEPEP_NORMAL_USE_ALLGATHER:-1}
        export SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=${SGLANG_DEEPEP_NORMAL_USE_ALLTOALL:-0}
        export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=${SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER:-1}
        export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS=${SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS:-1024}
        export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=${SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS:-2}
        export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MIN_CHUNK_TOKENS=${SGLANG_NPU_PREFILL_AG_FUSED_QKV_MIN_CHUNK_TOKENS:-1024}
        export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=${SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS:-4}
        export WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=${WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD:-0}
        # Enable only with a custom TopK build supporting unnormalized sigmoid.
        export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=${SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM:-0}
        CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
        MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-16384}
        MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
        profile_args+=(--ep-size "${TP_SIZE}" --moe-a2a-backend deepep --deepep-mode auto
            --enable-kv-mirror --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}"
            --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-${MAX_RUNNING_REQUESTS}}")
        # The text prefill graph/mixed layout requires native Flash. Explicit
        # Flash=0 keeps the previous eager prefill path unless requested otherwise.
        prefill_default=0
        if [[ ${WELM_NPU_USE_FLASH_ATTN} == 1 ]]; then prefill_default=1; fi
        WELM_VL_PREFILL_GRAPH=${WELM_VL_PREFILL_GRAPH:-${prefill_default}}
        WELM_VL_MIXED_CHUNK=${WELM_VL_MIXED_CHUNK:-${prefill_default}}
        for flag in WELM_VL_PREFILL_GRAPH WELM_VL_MIXED_CHUNK; do
            case "${!flag}" in 0|1) ;; *) echo "${flag} must be 0 or 1." >&2; exit 2 ;; esac
        done
        if [[ ${WELM_NPU_USE_FLASH_ATTN} != 1 && ( ${WELM_VL_PREFILL_GRAPH} == 1 || ${WELM_VL_MIXED_CHUNK} == 1 ) ]]; then
            echo "WeLM-VL prefill graph/mixed chunk requires WELM_NPU_USE_FLASH_ATTN=1." >&2
            exit 2
        fi
        if [[ ${WELM_VL_PREFILL_GRAPH} == 1 ]]; then
            if [[ ${SGLANG_DEEPEP_NORMAL_USE_ALLGATHER} != 1 || ${SGLANG_DEEPEP_NORMAL_USE_ALLTOALL} != 0 ]]; then
                echo "WeLM-VL DeepEP prefill graph requires NORMAL AllGather=1 and AllToAll=0." >&2
                exit 2
            fi
            read -r -a prefill_tokens <<< "${WELM_VL_PREFILL_TOKEN_BUCKETS:-256 512 1024 2048 4096 8192 16384}"
            if [[ ${#prefill_tokens[@]} == 0 ]]; then
                echo "WELM_VL_PREFILL_TOKEN_BUCKETS must contain positive token counts." >&2
                exit 2
            fi
            for size in "${prefill_tokens[@]}"; do
                if [[ ! ${size} =~ ^[1-9][0-9]*$ ]]; then
                    echo "Invalid prefill token bucket: ${size}; use space-separated positive integers." >&2
                    exit 2
                fi
            done
            export SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES=${SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES:-1,2,4,8}
            profile_args+=(--cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill "${prefill_tokens[@]}")
        else
            profile_args+=(--disable-prefill-cuda-graph)
        fi
        if [[ ${WELM_VL_MIXED_CHUNK} == 1 ]]; then profile_args+=(--enable-mixed-chunk); fi
        ;;
    baseline)
        # A reproducible comparison path even when the shell has text tuning set.
        export WELM_NPU_USE_MEGAMOE=0
        export WELM_NPU_USE_FLASH_ATTN=0
        export SGLANG_NPU_WELMV4_FUSED_QKV=0
        export SGLANG_NPU_USE_MULTI_STREAM=0
        export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=0
        export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=0
        export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=0
        export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=0
        CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
        MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
        MAX_RUNNING_REQUESTS=1
        WELM_VL_PREFILL_GRAPH=0
        WELM_VL_MIXED_CHUNK=0
        profile_args+=(--ep-size 1 --moe-a2a-backend none --disable-prefill-cuda-graph
            --disable-cuda-graph --disable-radix-cache --chunked-prefill-size -1
            --disable-overlap-schedule)
        ;;
    *) echo "Unknown WELM_VL_PROFILE=${WELM_VL_PROFILE}; choose optimized or baseline." >&2; exit 2 ;;
esac

# Vision/base/OE embedding stay eager; supported text bodies use prefill/decode graphs.
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3000}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-3000}

preflight=("${SCRIPT_DIR}/check_model.py" "${MODEL_PATH}" --runtime --tp "${TP_SIZE}" --base-device "${BASE_DEVICE}" --profile "${WELM_VL_PROFILE}")
extra_args=()
if [[ -n ${CHAT_TEMPLATE:-} ]]; then
    preflight+=(--chat-template "${CHAT_TEMPLATE}")
    extra_args+=(--chat-template "${CHAT_TEMPLATE}")
fi
echo "WeLM-VL profile=${WELM_VL_PROFILE} TP=${TP_SIZE} MegaMoE=${WELM_NPU_USE_MEGAMOE} FlashAttn=${WELM_NPU_USE_FLASH_ATTN} fusedQKV=${SGLANG_NPU_WELMV4_FUSED_QKV} prefillGraph=${WELM_VL_PREFILL_GRAPH} mixedChunk=${WELM_VL_MIXED_CHUNK}"
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
    --base-gpu-id "${BASE_DEVICE}" \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --enable-multimodal \
    --enable-over-encoding \
    --context-length "${CONTEXT_LENGTH}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" \
    --page-size 64 \
    "${profile_args[@]}" "${extra_args[@]}" "$@"
