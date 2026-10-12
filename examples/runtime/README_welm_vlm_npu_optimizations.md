# WeLM-VLM NPU vision optimizations

## Opt-in ViT Ulysses sequence parallelism

```bash
# All TP ranks must inherit the same value. Restart the service after changing it.
export SGLANG_VLM_VISION_USE_ULYSSES=1     # default: 0 (original vision TP)
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0      # vision encoder remains eager
```

The language model's TP/EP and projector TP remain unchanged. The SP size is
the existing TP group size; this first version requires attention-TP and TP
to have the same rank membership/order, BF16 NPU tensors, and the Ascend vision
attention backend. Heads must be divisible by SP size. Encoder DP and vision
graph capture are not supported together with this switch. Text-only decode
graphs and supported language-backbone prefill graphs are not disabled by it.
The bundled `welm_vl/run_950pr.sh` optimized profile preserves an explicit value
and otherwise defaults to 0; its baseline profile forces 0. For an SP-only A/B
comparison, use the SAME profile and change only this variable.

### Layout and numerical contract

Each block loads full QKV/OProj/FFN weights on every rank using TP=1 linear
loaders, while keeping the original checkpoint names. Physical rank 1/2/3 also
use linear `tp_rank=0`: each local output gets exactly one bias. There is no
block OProj/FFN AllReduce, including when the vision MC2 switch is enabled.
Projector MC2 retains its original eligibility checks. GELU and residual-norm
fusions remain independently controlled and operate on the new local shapes.

For N packed patches and P ranks, L=ceil(N/P):

```text
full hidden/position -> consecutive local rows [L, hidden]
  norm -> full QKV -> global-position RoPE
  pack [destination, L, QKV, heads/P, head_dim]
  AllToAll -> [source, L, QKV, heads/P, head_dim]
  flatten source+L -> remove global tail -> [N, heads/P, head_dim] x3
  FIA with original image/frame cumulative lengths
  pad global tail -> AllToAll -> restore [L, heads, head_dim]
  full OProj -> residual/norm -> full FFN -> residual
repeat blocks -> AllGather -> trim to N -> original TP projector
```

Padding is only at the GLOBAL tail and is removed BEFORE softmax; no dummy key
is visible to a real query. At most P-1 rows are added per micro-batch. Empty
local shards still participate using equal-sized buffers. The reverse exchange
restores heads in source-rank order. QKV checkpoint packing is all-Q/all-K/all-V,
not Q/K/V interleaved per head. FP32 rotary coefficients/intermediates are kept.

Different legal image sizes, mixed-size images and existing `grid_thw` temporal
segments preserve their global coordinates and original packed attention
boundaries. This does not add a new video input API. The existing
`SGLANG_VLM_VISION_MAX_PATCHES` micro-batch grouping is retained; each group has
its own SP plan and final gather. A single oversized image still needs memory
for its full-sequence attention and is not split into independent attentions.

Patch embedding and metadata generation are unchanged in this first version.
Gathering before projector avoids splitting a spatial-merge group across ranks
(e.g. 2760 patches / 4 = 690 local rows is not divisible by the merge factor 4).
It also preserves the original image-token output order. No full-vision TP1
fallback or branch on local token count is allowed after starting collectives.

Full block weights increase per-rank HBM consumption; recheck available KV
cache capacity and maximum concurrency. SP reduces communication bytes, not
the number of per-block collectives. Packing/contiguous copies and changed
GEMM shapes can offset gains. Mathematical semantics are unchanged, but BF16
GEMM partition/reduction changes are NOT guaranteed bitwise equivalent to TP.

### Validation

Local CPU tests exercise production layouts, poisoned padding, 1/2/4/8-rank
simulated collectives, real four-process Gloo exchanges, actual encoder/block forwards, 72-wide BF16 RoPE,
micro-batches, mixed images, temporal segments, projector ordering and startup
guards. They cannot certify NPU HCCL/FIA compatibility or task accuracy.

```bash
python -m pytest -q test/registered/unit/models/test_welmv4_vlm_ulysses_layout.py \
  test/registered/unit/models/test_welmv4_vlm_ulysses.py

# From the repository root, on a server with four FREE visible NPUs:
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc-per-node=4 \
  examples/runtime/welm_vl/check_vision_ulysses.py
```

The smoke test needs no checkpoint. It checks real HCCL exchanges, their exact
inverse, segmented FIA, RoPE and BF16 QKV/OProj against full-sequence references.
Default cases include tiny inputs, mixed segments, nondivisible lengths, and
2760 patches. `--cases '4;12,16;3,2,6;2760'` is customizable. Reported timing is
AllToAll+packing+FIA+reverse AllToAll wall time, not pure communication and not
an end-to-end speedup. `--atol/--rtol` are smoke-test tolerances, not model-quality
acceptance criteria; investigate failures rather than loosening them blindly.

Then compare service SP=0/1 using identical weights, images, decoding settings
and all other optimization flags. Check image embeddings/logits and application
accuracy, not only plausible generated text. Profile with image-embedding cache
misses and separate warm metadata from cold startup. For one micro-batch with
27 blocks expect 54 AllToAll calls, one final AllGather and the original
projector reduction (or projector MC2), with no block AllReduce. With multiple
micro-batches these counts scale accordingly. Actual gains require server tests.

Implementation is split between `models/welmv4_vlm.py` and
`layers/welmv4_vision_ulysses.py`. The latter documents destination/source and
sequence/head ordering next to each reshape/permute.

## Position and RoPE metadata cache

The vision encoder caches the interpolated learned position embeddings, host-side
attention cumulative lengths, and FP32 two-dimensional RoPE cos/sin tensors.
The key contains the complete ordered `(t, h, w)` image-grid sequence. Image pixels
and image-dependent activations are never cached here: different images with the
same grid can safely reuse this metadata. This is independent of radix caching
and the multimodal image-embedding cache.

The host grid is decoded once per encoder forward. It is no longer copied to the
NPU just to call `tolist()` / `item()` on it. The RoPE implementation accepts
half-width coefficients, so the previously redundant cos/sin duplication is
removed without changing the rotation or FP32 coefficient precision.

```bash
export SGLANG_VLM_VISION_METADATA_CACHE=1          # default: enabled
export SGLANG_VLM_VISION_METADATA_CACHE_ENTRIES=16
export SGLANG_VLM_VISION_METADATA_CACHE_MB=128    # per TP rank; retained tensors
```

The LRU is bounded by both entry count and tensor bytes. An oversized request
computes its metadata normally but does not retain it. The byte limit is not a
bound on temporary construction memory. Cache hits require eval mode and disabled
autograd. Cache signatures track dtype, device, relevant configuration, position
parameter identity/storage and its version counter when available. Moving the
encoder, `load_state_dict`, and the VLM serving weight loader clear the cache.
Code that modifies weights through `.data` outside these supported loading paths
must explicitly call `model.vision_encoder.clear_metadata_cache()` afterward;
PyTorch inference tensors do not provide version counters.

Validation on a CPU host:

```bash
python -m pytest -q test/registered/unit/models/test_welmv4_vlm_metadata_cache.py \
  test/registered/unit/models/test_welmv4_vlm_vision.py \
  test/registered/unit/models/test_welmv4_vlm_prefill_graph.py
```

Compare cold-cache and hot-cache requests with identical grids but different image
contents, then test changed resolution, multiple images, chunked prefill and
ordinary decode. Warm-cache profiling should lose the repeated position
interpolation/RoPE metadata operations. A first-seen grid still pays construction
cost; neither CPU tests nor cache hits establish an end-to-end speedup on 950PR.

## Opt-in vision fusions

These switches are independent of the language-backbone optimization switches.
All three default to `0`; they are experimental until validated on the target
950PR / CANN / torch_npu / Triton-Ascend combination. Set identical values on
all TP ranks before starting the service.

The bundled `welm_vl/run_950pr.sh` optimized profile enables the metadata cache
and leaves these fusions opt-in; its baseline profile explicitly disables all
four features, including inherited environment values. For per-feature A/B
tests keep the same optimized profile and change only these four switches,
since the baseline profile also changes unrelated text-model/runtime settings.

```bash
export SGLANG_VLM_NPU_USE_MC2=1
export SGLANG_VLM_NPU_MC2_MIN_M=128
export SGLANG_VLM_NPU_MC2_COMM_MODE=ccu
export SGLANG_VLM_NPU_FUSE_GELU=1
export SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM=1
```

| Path | Implementation | Numerical / parallel contract |
| --- | --- | --- |
| ViT attention OProj, MLP fc2, projector mlp.2 | `torch_npu.npu_mm_all_reduce_base`, replacing MatMul + AllReduce | Existing TP group, rank0-only bias, one collective, replicated output |
| ViT MLP fc1 + GELU | Triton BF16 MatMul + bias + tanh GELU | FP32 accumulation, BF16 rounding before activation, no intermediate GEMM tensor |
| Projector mlp.0 + GELU | Same fused kernel, exact GELU | `erf`, not substituted by tanh |
| Attention residual + norm2 in each ViT block | Triton Add + LayerNorm | Residual first rounded to input dtype, then FP32 mean/variance |

The 27-layer configuration has up to 55 MC2 sites (27 OProj + 27 MLP fc2 +
1 projector), 28 FC1/GELU sites, and 27 Add/LayerNorm sites. Norm1 and projector
LayerNorm are unchanged. MLP residual + the next block's Norm1 are not fused.
Checkpoint names, tensor shapes and parameter loaders are unchanged.

### MC2 boundaries

The row-parallel fast path requires contiguous BF16 activations/weights,
unquantized linears, parallel input, TP > 1, and the configured minimum M.
Unsupported paths use the original linear. Explicit skip-AllReduce and existing
forward flags are respected. This is MatMul-AllReduce, **not**
MatMul-ReduceScatter: it introduces no M-divisibility requirement or padding.
In particular, the TP4 FFN local K remains 4304 / 4 = 1076.

The HCCL communicator is taken from the current SGLang TP process group. A
communicator mismatch or missing requested API raises an error. Runtime
collective failures are deliberately not caught and retried through a different
collective, which could leave ranks on inconsistent paths. `ccu` / `ai_cpu`
selection must be supported by the installed runtime and topology; this option
does not change `HCCL_OP_EXPANSION_MODE` automatically.

MC2 may change the floating-point reduction order. Retaining bias placement does
not guarantee bitwise identical results, so both tensor-error and end-to-end
accuracy checks are required. Collective overlap is a performance hypothesis,
not a measured improvement in this patch.

### GELU and LayerNorm boundaries

There is no assumed `torch_npu.npu_bias_gelu` API. The FC1+GELU implementation is
a genuine Triton MatMul epilogue: BF16 inputs, FP32 dot accumulator, bias,
explicit BF16 rounding, FP32 activation, BF16 output. It uses masked N/K tails
without persistent padding or padded weights. QuickGELU retains the existing
implementation. Gather-output, skip-bias, quantized or incompatible linears also
retain their original implementation.

Exact GELU requires Triton-Ascend `tl.erf`; tanh uses
`triton.language.extra.cann.libdevice.tanh`. A missing capability raises an error
instead of changing the activation approximation. The fixed initial tile is
32x64x128; it is not hardware-autotuned. A fused kernel can be slower than CANN
GEMM + GELU if its matrix utilization is worse, so keep the GELU switch off if
the single-operator test shows regression.

The original BF16 `hidden + residual` rounds before `LayerNorm`. The native
`npu_add_layer_norm` may normalize an unrounded FP32 residual sum, so this patch
uses a custom kernel that preserves the BF16 boundary and returns both the
rounded residual and normalized output. FP32 reduction implementations can still
differ slightly; no bitwise guarantee is made for the normalized result.

### Validation before benchmarking the full model

First run the standalone kernels without loading the 80B model:

```bash
python test/manual/test_welmv4_vision_npu_fusions.py --op all
torchrun --standalone --nproc_per_node=4 \
  test/manual/test_welmv4_vision_npu_fusions.py --op mc2 --comm-mode ccu
```

The single-device run covers tanh `[2760,1152] x [1076,1152]`, exact projector
`[690,4608] x [1152,4608]`, small tail cases, and 1152-wide Add/LayerNorm. The
four-rank run checks OProj, FFN K=1076 and projector MC2 against GEMM + AllReduce
with rank0-only bias. Each case reports error, eager-event mean/p50/p90 latency
and speedup; the distributed report also includes the maximum rank mean. Eager
microbenchmark timings are not graph timings or model end-to-end latency.

Then A/B test the service: baseline with all new switches off; metadata cache
only; cache + MC2; cache + GELU; cache + Add/LayerNorm; finally the combination.
Use identical weights, request images/resolutions, TP settings, concurrency and
warmup. Test multi-image order, resolution changes, long input, chunked prefill,
text-only requests and decode. Compare output/logits or your accuracy dataset,
ViT wall time, TTFT and full request time, not only summed operator durations.
Revert a fusion independently by setting its switch to `0` and restarting.

CPU integration checks:

```bash
python -m pytest -q test/registered/unit/models/test_welmv4_vlm_fusions.py \
  test/registered/unit/models/test_welmv4_vlm_metadata_cache.py \
  test/registered/unit/models/test_welmv4_vlm_vision.py \
  test/registered/unit/models/test_welmv4_vlm_prefill_graph.py
```

References:

- [torch_npu MatMul-AllReduce API](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mm_all_reduce_base.md)
- [Ascend AddLayerNorm implementation](https://github.com/Ascend/op-plugin/blob/master/op_plugin/ops/opapi/AddLayerNormKernelNpuOpApi.cpp)
