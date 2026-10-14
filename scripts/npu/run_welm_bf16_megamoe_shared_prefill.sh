#!/usr/bin/env bash
# Explicitly invoked prefill worker; no package installation or router startup.
set -euo pipefail

usage() {
    cat <<'USAGE'
Run the BF16 WeLM prefill worker with MegaMoE shared-expert fusion.

Required environment:
  MODEL_PATH             Real local BF16 WeLM model directory.
  MEGAMOE_VENDOR_DIR     Matching shared-version CANN vendor directory.
  ASCEND_MF_STORE_URL    MemFabric store URL for this deployment.
  ASCEND_MF_HCOM_URL     MemFabric HCOM mapping for the selected devices.

Optional environment:
  PYTHON_BIN=python, CANN_ENV_SH=/path/to/cann/set_env.sh
  MEMFABRIC_ENV_SH=/usr/local/memfabric_hybrid/set_env.sh
  WELM_OPS_ROOT=/usr/local/welm-ops (topk/flash/newgmm vendor packages)
  TP_SIZE=4, BASE_GPU_ID=0, HOST=0.0.0.0, PORT=2233, BOOTSTRAP_PORT=8995
  SERVED_MODEL_NAME=welmv45, MEM_FRACTION_STATIC=0.8
  WELM_NPU_USE_MEGAMOE=1 (0 selects the ordinary MoE backend)
  WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=512 (checked only when enabled)
  SGLANG_NPU_USE_MULTI_STREAM=1 (shared overlap on ordinary MoE paths)

Load the matched npu_ops_transformer wheel and CANN dependencies beforehand.
Configure HCCL/GLOO interfaces and MemFabric addresses for the actual machine.
This starts only the prefill worker; decode and router are separate services.
Any additional arguments are forwarded to sglang.launch_server.
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

: "${MODEL_PATH:?Set MODEL_PATH to the real BF16 WeLM model directory}"
: "${MEGAMOE_VENDOR_DIR:?Set MEGAMOE_VENDOR_DIR to the matched shared-version vendor}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

source_env() {
    if [[ ! -f "$1" ]]; then
        echo "Environment script not found: $1" >&2
        exit 1
    fi
    # Vendor setup scripts may read optional variables without nounset guards.
    set +u
    source "$1"
    set -u
}

if [[ -n "${CANN_ENV_SH:-}" ]]; then
    source_env "${CANN_ENV_SH}"
fi
source_env "${MEMFABRIC_ENV_SH:-/usr/local/memfabric_hybrid/set_env.sh}"
for op in topk flash newgmm; do
    source_env "${WELM_OPS_ROOT:-/usr/local/welm-ops}/${op}/vendors/custom_transformer/bin/set_env.bash"
done
source_env "${MEGAMOE_VENDOR_DIR}/bin/set_env.bash"
export ASCEND_CUSTOM_OPP_PATH="${MEGAMOE_VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${MEGAMOE_VENDOR_DIR}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${ASCEND_MF_STORE_URL:?Set the MemFabric store URL for this deployment}"
: "${ASCEND_MF_HCOM_URL:?Set the MemFabric per-device HCOM mapping}"
export ASCEND_MF_STORE_URL ASCEND_MF_HCOM_URL
export ASCEND_MF_TRANSFER_PROTOCOL="${ASCEND_MF_TRANSFER_PROTOCOL:-host_rdma}"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${repo_root}/python${PYTHONPATH:+:${PYTHONPATH}}"
# Every MegaMoE call includes shared experts; fallbacks keep shared overlap.
export WELM_NPU_USE_MEGAMOE="${WELM_NPU_USE_MEGAMOE:-1}"
export WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD="${WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD:-512}"
export SGLANG_NPU_USE_MULTI_STREAM="${SGLANG_NPU_USE_MULTI_STREAM:-1}"
export SGLANG_NPU_WELMV4_USE_FUSED_TOPK=1
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1
export SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
export DEEP_USE_MODE=default
export SGLANG_NPU_WELMV4_FUSED_QKV=1
export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=1
export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS=1024
export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=2
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MIN_CHUNK_TOKENS=256
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=4
export ASCEND_USE_FIA=1
export WELM_NPU_USE_FLASH_ATTN=1
export SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES="1,2,3,4,5,6,7,8,10,12,16"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-CCU_SCHED}"

# DeepEP resolves EP=TP. Keep it as the ordinary backend for ineligible layers.
exec "${PYTHON_BIN:-python}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --tp-size "${TP_SIZE:-4}" \
    --port "${PORT:-2233}" \
    --host "${HOST:-0.0.0.0}" \
    --trust-remote-code \
    --served-model-name "${SERVED_MODEL_NAME:-welmv45}" \
    --nnodes 1 --node-rank 0 --base-gpu-id "${BASE_GPU_ID:-0}" \
    --cuda-graph-config '{"prefill":{"backend":"breakable","max_bs":16512,"bs":[128,256,512,1024,1536,2048,2560,3072,4096,5120,6144,7168,8192,10240,12288,16512]}}' \
    --attention-backend ascend \
    --enable-kv-mirror --chunked-prefill-size 16512 --enable-over-encoding \
    --moe-a2a-backend deepep --deepep-mode auto \
    --max-running-requests 128 --swa-full-tokens-ratio 0.8 \
    --disable-swa-chunk-cap --page-size 128 \
    --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8995}" \
    --disable-overlap-schedule --sampling-backend ascend --enable-cache-report \
    --reasoning-parser qwen3 --tool-call-parser qwen \
    "$@"
