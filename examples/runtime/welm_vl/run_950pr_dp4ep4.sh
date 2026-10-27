#!/usr/bin/env bash
# Four attention-DP workers, each with full attention and ViT weights; MoE EP=4.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the complete WeLM-VL checkpoint directory}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

for name in TP_SIZE DP_SIZE EP_SIZE; do
    if [[ ${!name:-4} != 4 ]]; then
        echo "The DP4EP4 launcher requires ${name}=4 (attention TP=1)." >&2
        exit 2
    fi
done
for name in WELM_VL_PREFILL_GRAPH WELM_VL_MIXED_CHUNK; do
    if [[ ${!name:-0} != 0 ]]; then
        echo "DP4EP4 currently requires ${name}=0; prefill runs eagerly." >&2
        exit 2
    fi
done
if [[ -n ${CHUNKED_PREFILL_SIZE:-} ]]; then
    echo 'Use CHUNKED_PREFILL_TOKENS_PER_WORKER (at most 32768), not CHUNKED_PREFILL_SIZE.' >&2
    exit 2
fi
CHUNKED_PREFILL_TOKENS_PER_WORKER=${CHUNKED_PREFILL_TOKENS_PER_WORKER:-32768}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-${CHUNKED_PREFILL_TOKENS_PER_WORKER}}
for name in CHUNKED_PREFILL_TOKENS_PER_WORKER MAX_PREFILL_TOKENS; do
    value=${!name}
    if [[ ! ${value} =~ ^[1-9][0-9]{0,4}$ ]] || (( value > 32768 )); then
        echo "${name} must be a positive per-worker token count no larger than 32768." >&2
        exit 2
    fi
done
# ServerArgs divides --chunked-prefill-size by DP, but does not divide
# --max-prefill-tokens. Keep the CLI and effective worker limits explicit.
global_chunk_tokens=$((CHUNKED_PREFILL_TOKENS_PER_WORKER * 4))

# Prevent extra CLI arguments from silently changing this deployment contract.
for argument in "$@"; do
    case "${argument%%=*}" in
        --tp|--tp-size|--tensor-parallel-size|--dp|--dp-size|--data-parallel-size|\
        --ep|--ep-size|--expert-parallel-size|--moe-a2a-backend|--deepep-mode|\
        --chunked-prefill-size|--max-prefill-tokens|--enable-mixed-chunk|\
        --cuda-graph-config|--cuda-graph-backend-prefill|--cuda-graph-bs-prefill|\
        --cuda-graph-max-bs-prefill)
            echo "${argument} is controlled by the DP4EP4 profile; use its documented environment settings." >&2
            exit 2 ;;
    esac
done
source "${SCRIPT_DIR}/vision_parallel_args.sh" "$@"
if (( ${#vision_dp_args[@]} == 0 )); then
    echo 'DP4EP4 requires an explicit --mm-enable-dp-encoder; each attention-DP worker runs its own ViT.' >&2
    exit 2
fi

export SGLANG_VIT_ENABLE_CUDA_GRAPH=0
export SGLANG_VLM_VISION_METADATA_CACHE=${SGLANG_VLM_VISION_METADATA_CACHE:-1}
export SGLANG_VLM_NPU_USE_MC2=0
export SGLANG_VLM_NPU_FUSE_GELU=${SGLANG_VLM_NPU_FUSE_GELU:-0}
export SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM=${SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM:-0}
export SGLANG_VLM_NPU_FUSE_ROPE=${SGLANG_VLM_NPU_FUSE_ROPE:-0}
export WELM_NPU_USE_MEGAMOE=${WELM_NPU_USE_MEGAMOE:-0}
export WELM_NPU_MEGAMOE_FUSE_SHARED_EXPERT=0
export WELM_NPU_USE_FLASH_ATTN=${WELM_NPU_USE_FLASH_ATTN:-0}
# Attention TP=1 uses ordinary QKV/O projections; TP4/TP2 kernels cannot apply.
export SGLANG_NPU_WELMV4_FUSED_QKV=0
export SGLANG_NPU_WELMV4_DP_FUSED_QKV=0
export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=0
export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=0
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=0
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_SWA_MAX_CHUNKS=0
export SGLANG_NPU_USE_MULTI_STREAM=0
export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=0
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1
export SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3000}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-3000}

WELM_VL_DECODE_GRAPH=${WELM_VL_DECODE_GRAPH:-1}
graph_args=()
case "${WELM_VL_DECODE_GRAPH}" in
    1) graph_args+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-8}") ;;
    0) graph_args+=(--disable-cuda-graph) ;;
    *) echo 'WELM_VL_DECODE_GRAPH must be 0 or 1.' >&2; exit 2 ;;
esac
extra_args=()
if [[ -n ${CHAT_TEMPLATE:-} ]]; then
    extra_args+=(--chat-template "${CHAT_TEMPLATE}")
fi
echo "WeLM-VL DP4EP4: world TP=4, attention TP=1, DP=4, EP=4; prefill eager, decodeGraph=${WELM_VL_DECODE_GRAPH}."
echo "Prefill tokens: CLI chunk=${global_chunk_tokens}, per-worker chunk=${CHUNKED_PREFILL_TOKENS_PER_WORKER}, per-worker max=${MAX_PREFILL_TOKENS}."
"${PYTHON_BIN}" "${SCRIPT_DIR}/check_model.py" "${MODEL_PATH}" \
    --runtime --tp 4 --dp 4 --base-device "${BASE_DEVICE:-0}" \
    --profile dp4ep4 "${vision_dp_args[@]}" "${extra_args[@]}"

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" --trust-remote-code \
    --served-model-name "${SERVED_MODEL_NAME:-welmv45-vl}" \
    --host "${HOST:-0.0.0.0}" --port "${PORT:-6677}" \
    --device npu --dtype bfloat16 --base-gpu-id "${BASE_DEVICE:-0}" \
    --tp-size 4 --dp-size 4 --ep-size 4 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --attention-backend ascend --mm-attention-backend ascend_attn \
    --enable-multimodal --enable-over-encoding \
    --context-length "${CONTEXT_LENGTH:-65536}" \
    --chunked-prefill-size "${global_chunk_tokens}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS:-32}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.80}" --page-size 64 \
    --disable-prefill-cuda-graph \
    "${graph_args[@]}" "${extra_args[@]}" "$@"
