# WeLM-VLM NPU vision optimizations

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
