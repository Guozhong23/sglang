# WeLM Ascend MegaMoE validation

This directory contains the EP4 operator probe and the A/B procedure for the
WeLM MXFP8 MegaMoE path.

## Prerequisites

Install the matching custom operator package and Python wheel, then export the
operator API library before every run:

```bash
bash cann-ops-transformer-custom_linux-x86_64.run --quiet
pip install --force-reinstall --no-deps npu_ops_transformer-1.0.0-py3-none-any.whl

source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
```

## 1. Operator validation

Start with small cases that include a semantic BF16 EP reference:

```bash
cd scripts/npu/megamoe
bash run_welm_megamoe_op.sh --padded-rows 1
```

The defaults exercise the real WeLM geometry: EP4, 512 routed experts,
local-E128, H=2048, I=512, TopK=10 and M=1/16/128. It reports latency,
cosine similarity, normalized RMSE and padded-row error to CSV. The M=1 case
cannot contain both a real and padded row, so padding is checked by M=16/128.

Then measure the real prefill shard without the slower reference:

```bash
M_CASES=512,1024,4128 REFERENCE_MAX_M=0 \
  bash run_welm_megamoe_op.sh --warmup 10 --repeat 100
```

Do not proceed to E2E until every rank reports finite output, the small cases
pass the accuracy thresholds, and the zero-weight padded row remains zero.

## 2. Framework A/B

Use the same ModelSlim MXFP8 checkpoint, input corpus, server arguments, NPU
frequency, warmup count and benchmark concurrency for both runs. Change only
the MoE backend and the backend-specific environment variables.

### A: current DeepEP AllGather path

```bash
export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=1
unset SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK
unset SGLANG_NPU_MEGAMOE_VALIDATE_INPUTS

# Keep the existing launch command and use:
#   --quantization modelslim
#   --moe-a2a-backend deepep --deepep-mode auto
```

### B: MegaMoE path

For `--chunked-prefill-size 16512` with EP4, the exact ordinary-prefill shard
is 4128 rows. 4352 leaves padding headroom and registers about 80 MiB per rank.

```bash
export NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE=minimal
export SGLANG_NPU_USE_MULTI_STREAM=1
export SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK=4352
export SGLANG_NPU_MEGAMOE_VALIDATE_INPUTS=1  # first functional run only
unset SGLANG_DEEPEP_NORMAL_USE_ALLGATHER
unset SGLANG_NPU_MXFP8_QUANT_BEFORE_ROUTE
unset USE_MX_FP8_QUANT

# Keep the existing launch command and use:
#   --quantization modelslim
#   --moe-a2a-backend megamoe
```

`SGLANG_NPU_MEGAMOE_VALIDATE_INPUTS=1` intentionally synchronizes routing
tensors to check ID range and uniqueness. After one curl/accuracy smoke run,
restart with it set to `0` before collecting performance.

The B path uses MegaMoE only for ordinary token-sharded prefill layers before
the first target KV-mirror consumer. For the current 48-layer WeLM model this
is layers 0-32. Layers 33-47, decode, verify, KV-mirror full rows, and MTP keep
the existing local-EP plus AllReduce path. The boundary is read from
`kv_mirror_layers` rather than hard-coded. When KV mirror is disabled, the
ordinary prefill layout remains token-sharded and all target layers are
eligible for MegaMoE.

Weight post-processing follows the same split: eligible prefix layers retain
the canonical MegaMoE MXFP8 layout, while the full-row suffix uses the regular
Ascend GMM/FRACTAL_NZ layout. Prefix-layer decode reuses the canonical weights
through the existing zero-copy GMM transpose view; it does not call MegaMoE.
Shared experts remain replicated and execute on SGLang's independent NPU
stream; the main stream waits only at the routed/shared addition.

## 3. Numerical debug against local EP

The debug path is disabled by default. For a short, one-token curl, enable the
following settings on all ranks. `MAX_CALLS=2` normally captures both the
startup warmup and the first user request; set `SKIP_CALLS=1` and
`MAX_CALLS=1` when the server always performs exactly one warmup invocation.

```bash
export SGLANG_NPU_MEGAMOE_VALIDATE_INPUTS=1
export SGLANG_NPU_MEGAMOE_DEBUG=1
export SGLANG_NPU_MEGAMOE_DEBUG_LAYERS=0,1,2,4,8,16,32
export SGLANG_NPU_MEGAMOE_DEBUG_SKIP_CALLS=0
export SGLANG_NPU_MEGAMOE_DEBUG_MAX_CALLS=2
export SGLANG_NPU_MEGAMOE_SHADOW_COMPARE=1
export SGLANG_NPU_MEGAMOE_SHADOW_MAX_GLOBAL_ROWS=256
```

Use only a short request while shadow comparison is enabled. The shadow path
all-gathers the real input rows, runs the existing rank-local MXFP8 experts,
all-reduces their partial outputs, and compares the corresponding local rows
with MegaMoE. Requests above `SHADOW_MAX_GLOBAL_ROWS` are skipped rather than
allocating a large reference workspace.

Search the server log for these markers:

```bash
grep -E 'MegaMoE-(Debug|Shadow)' server.log
```

- `expert-counts mismatches > 0` identifies global expert-id, rank ownership,
  or dispatch-count disagreement before inspecting GEMM numerics.
- `rms_ratio` far from 1, especially by a power of two, points to E8M0 scale
  byte/order or weight-scale block pairing.
- Correct expert counts but a poor layer-0 cosine points to W13 gate/up order,
  scale layout, SwiGLU, or Combine semantics.
- A close layer-0 comparison that progressively worsens in later selected
  layers indicates cumulative numerical error rather than routing ownership.
- The W13 samples include rows 511/512, the WeLM gate/up boundary, and both
  sides of the 32/64-element MX block boundary.

Disable all debug switches and restart before profiling performance.

## 4. Acceptance criteria

1. Functional: deterministic curl output is coherent and no rank exceeds the
   registered local-token capacity.
2. Precision: compare routed output first, then 10/198-case evaluation against
   the current MXFP8 baseline. Do not compare only against BF16 because that
   mixes operator error with checkpoint quantization error.
3. Profiling: the B trace must replace the MoE AllGather dispatch, GMM1,
   SwiGLU-MXQuant, GMM2 and Combine sequence with `aclnnMegaMoe`; the attention
   TP AllGather is independent and remains.
4. Performance: compare the MoE critical path, not the sum of two streams. The
   layer cost is `max(routed_path, shared_expert) + final_sync`.
5. Rollback: switching the backend back to `deepep` restores the unmodified A
   path without changing the checkpoint.
