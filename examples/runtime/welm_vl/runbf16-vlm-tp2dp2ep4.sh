#!/usr/bin/env bash
# One four-card worker: attention TP2 x DP2, MoE EP4.
# Run from the server after applying the matching code patch.
set -eo pipefail
SGLANG_CODE_ROOT="${SGLANG_CODE_ROOT:-/data2/hw_sgz/vlm/03_code_vlm/sglang}"
PORT="${PORT:-8899}"
HOST="${HOST:-28.157.240.165}"
PREFILL_GRAPH="${PREFILL_GRAPH:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

export PYTHONPATH="${SGLANG_CODE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
source /usr/local/memfabric_hybrid/set_env.sh

export ASCEND_USE_FIA=1
export WELM_NPU_USE_FLASH_ATTN=1

source /usr/local/welm-ops/topk/vendors/custom_transformer/bin/set_env.bash
source /usr/local/welm-ops/flash/vendors/custom_transformer/bin/set_env.bash
source /usr/local/welm-ops/megamoe1015/vendors/custom_transformer/bin/set_env.bash
source /usr/local/welm-ops/newgmm/vendors/custom_transformer/bin/set_env.bash

export LD_LIBRARY_PATH="/usr/local/welm-ops/hccl/cann/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ACL_DEVICE_SYNC_TIMEOUT=3000
export HCCL_CONNECT_TIMEOUT=3000

export FREQUENCY_PENALTY_EXCLUDE_TOKENS="[409, 520, 590, 767, 818, 864, 890, 1193, 1212, 1309, 1473, 1675, 1996, 2030, 2289, 2298, 2464, 2581, 2636, 2768, 2985, 3021, 3080, 3095, 3099, 3321, 3491, 3579, 3694, 3854, 4051, 4575, 4622, 5017, 5079, 5125, 5242, 5572, 5598, 5776, 5847, 5954, 6287, 6302, 6345, 6588, 6653, 7243, 7319, 7357, 7806, 7860, 8018, 8138, 8163, 8486, 8703, 8939, 9040, 9861, 10132, 10376, 11102, 11320, 11420, 11427, 11623, 11642, 11750, 12053, 12079, 12181, 14144, 14241, 14396, 14840, 14964, 15053, 15797, 16159, 16244, 16411, 16545, 16605, 17915, 19458, 19797, 19814, 20048, 20549, 21582, 22070, 22143, 22229, 23039, 23398, 24992, 25893, 28777, 29455, 29786, 30184, 30403, 31421, 31735, 34154, 34242, 34697, 35621, 35642, 36563, 36792, 38009, 38820, 38990, 39129, 39241, 39321, 41455, 42930, 43530, 43574, 43659, 44280, 45336, 45759, 46908, 47744, 47747, 49010, 49142, 51084, 51936, 53218, 53671, 54260, 56054, 57215, 60468, 64054, 68608, 68952, 69306, 69579, 76012, 76312, 79301, 81746, 82957, 87202, 89304, 98484, 99267, 114495, 123644, 127728, 130644, 134428, 140066, 141719, 146800, 147433, 150361, 154168, 154754, 154755]"

export FREQUENCY_PENALTY_FACTOR_START=0
export FREQUENCY_PENALTY_FACTOR_OFFSET=2048
export SGLANG_FORWARD_UNKNOWN_TOOLS=1

###############################################################################
# Profiling配置
#
# 采集脚本会通过/start_profile中的output_dir为每次采集指定独立目录。
# 这里的目录作为服务端默认/fallback输出目录。
###############################################################################

PROFILE_ROOT="${PROFILE_ROOT:-/data2/hw_sgz/vlm/05_profiling/}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_ROOT="${LOG_ROOT:-${PROFILE_ROOT}}"
LOG_FILE="${LOG_ROOT}/sglang_welmv45_vlm_attntp2_dp2_ep4_port${PORT}_${TIMESTAMP}.log"

mkdir -p "${PROFILE_ROOT}"
mkdir -p "${LOG_ROOT}"

###############################################################################
# PyTorch Profiler配置
###############################################################################

# /start_profile未指定output_dir时使用该目录
export SGLANG_TORCH_PROFILER_DIR="${PROFILE_ROOT}"

# 显式开启Python/C++调用堆栈
export SGLANG_PROFILE_WITH_STACK=False

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_OP_EXPANSION_MODE="CCU_SCHED"
# export HCCL_OP_EXPANSION_MODE="AIV"

export SGLANG_NPU_MIGRATE_PAGES=1
export SGLANG_NPU_AFFINITY_PCORES_PER_PROC=8

export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_NPU_WELMV4_USE_FUSED_TOPK=1
export SGLANG_NPU_USE_SCATTER_PA_KV_CACHE=1

export SGLANG_NPU_MOE_USE_GROUP_LIST_TYPE_2=0

export TRITON_PRINT_AUTOTUNING=0
export TRITON_DEBUG=0

# export SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=1

export DEEP_USE_MODE="default"
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1
export SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER="${SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER:-0}"

export SGLANG_NPU_WELMV4_FUSED_QKV=1

export SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM=1

export WELM_NPU_USE_MEGAMOE=1
export WELM_NPU_MEGAMOE_FUSE_SHARED_EXPERT=1
export WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=512

export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS="${SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS:-1024}"
export SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS="${SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS:-2}"

export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MIN_CHUNK_TOKENS=512
export SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=4

# export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/data2/hw_lly/eplb
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES="${SGLANG_WELMV4_PREFILL_GRAPH_BATCH_SIZES:-1,2,4,8,16}"

# VI配置
export SGLANG_VLM_VISION_USE_ULYSSES=0
export SGLANG_VLM_VISION_METADATA_CACHE=1


MODEL_PATH="${MODEL_PATH:-/data2/hw_zjw/sys_vlm/00_weights/welm_vlm_v1_1015/}"

GLOBAL_CHUNK_TOKENS="${GLOBAL_CHUNK_TOKENS:-16384}"
PREFILL_GRAPH_MAX_TOKENS="${PREFILL_GRAPH_MAX_TOKENS:-8192}"
PREFILL_GRAPH_BUCKETS="${PREFILL_GRAPH_BUCKETS:-64,128,256,512,1024,1536,2048,2560,3072,4096,5120,6144,7168,8192}"
# Each new device path is opt-in until four-rank correctness is verified.
export SGLANG_NPU_WELMV4_DP_FUSED_QKV="${SGLANG_NPU_WELMV4_DP_FUSED_QKV:-0}"
export SGLANG_WELMV4_DP_FUSED_PACK="${SGLANG_WELMV4_DP_FUSED_PACK:-0}"
export SGLANG_WELMV4_DP_REPLICA_ALLGATHER="${SGLANG_WELMV4_DP_REPLICA_ALLGATHER:-0}"
export SGLANG_WELMV4_DP_PREFILL_IDLE_GRAPH="${SGLANG_WELMV4_DP_PREFILL_IDLE_GRAPH:-0}"
export SGLANG_WELMV4_DP_SCRATCH_CACHE="${SGLANG_WELMV4_DP_SCRATCH_CACHE:-0}"
export SGLANG_WELMV4_DP_PROFILE_RANGES="${SGLANG_WELMV4_DP_PROFILE_RANGES:-0}"
# Capture token buckets are per attention-DP group.
# 16384 / DP2 = 8192, divisible by page-size 128.
# Enable SGLANG_WELMV4_DP_PREFILL_IDLE_GRAPH for mask01/10 captures;
# mask11 stays enabled with PREFILL_GRAPH=1.
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0
export SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=0
if [[ "${PREFILL_GRAPH}" == "1" ]]; then
    GRAPH_CONFIG="{\"prefill\":{\"backend\":\"breakable\",\"max_bs\":${PREFILL_GRAPH_MAX_TOKENS},\"bs\":[${PREFILL_GRAPH_BUCKETS}]}}"
elif [[ "${PREFILL_GRAPH}" == "0" ]]; then
    GRAPH_CONFIG='{"prefill":{"backend":"disabled"}}'
else
    printf '%s\n' 'PREFILL_GRAPH must be 0 or 1.' >&2
    exit 2
fi

printf 'WeLM TP2DP2EP4: global_chunk=%s per_dp_chunk=%s graph_max=%s MC2=%s pipeline=%s\n' \
    "${GLOBAL_CHUNK_TOKENS}" "$((GLOBAL_CHUNK_TOKENS / 2))" "${PREFILL_GRAPH_MAX_TOKENS}" \
    "${SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER}" "${SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS}"
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --device npu \
    --dtype bfloat16 \
    --kv-cache-dtype bfloat16 \
    --mem-fraction-static 0.8 \
    --tp-size 4 \
    --dp-size 2 \
    --enable-dp-attention \
    --ep-size 4 \
    --port "${PORT}" \
    --host "${HOST}" \
    --log-level "${LOG_LEVEL}" \
    --trust-remote-code \
    --served-model-name welmv4-vl \
    --nnodes 1 \
    --node-rank 0 \
    --base-gpu-id 0 \
    --cuda-graph-config "${GRAPH_CONFIG}" \
    --cuda-graph-max-bs 72 \
    --attention-backend ascend \
    --mm-attention-backend ascend_attn \
    --enable-multimodal \
    --enable-kv-mirror \
    --chunked-prefill-size "${GLOBAL_CHUNK_TOKENS}" \
    --enable-over-encoding \
    --max-running-requests 128 \
    --swa-full-tokens-ratio 0.8 \
    --disable-swa-chunk-cap \
    --page-size 128 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --enable-cache-report \
    --mm-enable-dp-encoder \
    "$@" 2>&1 | tee "${LOG_FILE}"
