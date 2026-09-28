"""CPU numerical and integration contracts without loading the 80B backbone.

The definitions are compiled from the actual model source so this suite can
run on hosts without the serving runtime's accelerator packages. It exercises
real PyTorch arithmetic; the text model is replaced only for wrapper contracts.
"""

import ast
import copy
import dataclasses
import logging
import importlib.util
import os
import runpy
import sys
import types
import weakref
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "python/sglang/srt/models/welmv4_vlm.py"
OPS_PATH = ROOT / "python/sglang/srt/layers/welmv4_vision_op.py"


def _load_ops():
    spec = importlib.util.spec_from_file_location("welmv4_vision_test_ops", OPS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OPS = _load_ops()


def _load_definitions(*names, **extra):
    tree = ast.parse(MODEL_PATH.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    scope = dict(
        torch=torch, nn=nn, F=F, os=os, copy=copy, SimpleNamespace=SimpleNamespace
    )
    scope.update(extra)
    exec(compile(module, str(MODEL_PATH), "exec"), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})


@pytest.mark.parametrize("head_dim", [64, 72])
@pytest.mark.parametrize("full_coefficients", [False, True])
def test_rope_matches_complex_rotation_and_keeps_fp32_coefficients(
    head_dim, full_coefficients
):
    torch.manual_seed(912)
    q, k = [torch.randn(5, 3, head_dim).bfloat16() for _ in range(2)]
    angle = torch.randn(5, head_dim // 2)
    cos, sin = angle.cos(), angle.sin()
    phase = torch.complex(cos, sin).unsqueeze(1)
    expected = []
    for x in (q, k):
        lo, hi = x.float().chunk(2, dim=-1)
        rotated = torch.complex(lo, hi) * phase
        expected.append(torch.cat((rotated.real, rotated.imag), dim=-1).bfloat16())
    if full_coefficients:
        cos, sin = [torch.cat((c, c), dim=-1) for c in (cos, sin)]
    actual = OPS.welmv4_vision_apply_rope(q, k, cos, sin)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
    rounded = OPS.welmv4_vision_apply_rope(q, k, cos.bfloat16(), sin.bfloat16())
    assert not torch.equal(actual[0], rounded[0]), "fixture must expose early rounding"


def test_rope_rejects_wrong_dtype_or_shape():
    x = torch.zeros(2, 3, 72, dtype=torch.bfloat16)
    coeff = torch.ones(2, 36)
    with pytest.raises(TypeError, match="bfloat16"):
        OPS.welmv4_vision_apply_rope(x.float(), x.float(), coeff, coeff)
    with pytest.raises(ValueError, match="equal"):
        OPS.welmv4_vision_apply_rope(x, x[:, :2], coeff, coeff)
    with pytest.raises(ValueError, match="cos and sin"):
        OPS.welmv4_vision_apply_rope(x, x, coeff[:, :32], coeff[:, :32])


def test_quick_gelu_rounds_once():
    x = torch.linspace(-5, 5, 1024, dtype=torch.bfloat16)
    expected = (x.double() / (1 + (-1.702 * x.double()).exp())).bfloat16()
    actual = OPS.welmv4_vision_quick_gelu(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, x * torch.sigmoid(1.702 * x))
    assert OPS.welmv4_vision_quick_gelu(x[:0]).numel() == 0


@pytest.mark.parametrize("chunk_size", [0, 1, 3, 32])
def test_patch_projection_preserves_separate_bf16_bias_add(chunk_size):
    projection = _load_definitions(
        "_patch_embed_forward_matmul"
    )._patch_embed_forward_matmul
    torch.manual_seed(77)
    x = torch.randn(7, 3 * 2 * 2 * 2).bfloat16()
    weight = torch.randn(12, 3, 2, 2, 2).bfloat16()
    bias = torch.randn(12).bfloat16()
    expected = F.linear(x, weight.flatten(1)) + bias
    actual = projection(x, weight, bias, 2, 2, 2, 3, 12, chunk_size=chunk_size)
    torch.testing.assert_close(actual.squeeze(1), expected, atol=0, rtol=0)
    assert not torch.equal(expected, F.linear(x, weight.flatten(1), bias))


def test_position_embedding_merge_order_and_frame_repeat():
    position_type = _load_definitions(
        "WeLMV4VisionPositionEmbedding"
    ).WeLMV4VisionPositionEmbedding
    config = SimpleNamespace(
        hidden_size=1, spatial_merge_size=2, num_position_embeddings=16
    )
    pos = position_type(config)
    pos.weight.data.copy_(torch.arange(16).reshape(16, 1))
    actual = pos(torch.tensor([[2, 4, 4]]))
    # Spatial merge groups the top-left 2x2 block before the top-right block.
    frame = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
    torch.testing.assert_close(actual[:, 0], frame.repeat(2).float())
    # A 2x2 target must sample all four corners of the learned 4x4 grid.
    torch.testing.assert_close(
        pos(torch.tensor([[1, 2, 2]]))[:, 0], torch.tensor([0.0, 3.0, 12.0, 15.0])
    )


def test_encoder_groups_do_not_split_an_image():
    encoder = _load_definitions("WeLMV4VisionEncoder").WeLMV4VisionEncoder
    assert encoder._split_groups_by_image([0, 4, 12, 16, 20], 8) == [
        (0, 1),
        (1, 2),
        (2, 4),
    ]
    assert encoder._split_groups_by_image([0, 12, 16], 8) == [(0, 1), (1, 2)]
    assert encoder._split_groups_by_image([0], 8) == []


class _Backbone(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.text_weight = nn.Parameter(torch.zeros(1))

    def load_weights(self, weights, is_nextn=False):
        self.text_weights = list(weights)

    def forward(self, **kwargs):
        return kwargs

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return config


class _Vision(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))


def _make_wrapper():
    namespace = _load_definitions(
        "_as_config",
        "WeLMV4VLMForConditionalGeneration",
        WeLMV4MoeForCausalLM=_Backbone,
        WeLMV4VisionEncoder=_Vision,
        WeLMV4VisionProjector=_Vision,
        get_mm=lambda: SimpleNamespace(mm_enable_dp_encoder=False),
        add_prefix=lambda suffix, prefix: f"{prefix}.{suffix}" if prefix else suffix,
        default_weight_loader=lambda parameter, value: parameter.data.copy_(value),
    )
    config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=4, vocab_size=100),
        vision_config=SimpleNamespace(out_hidden_size=4),
        image_token_id=77,
    )
    return namespace.WeLMV4VLMForConditionalGeneration(config), config


def test_wrapper_keeps_text_and_multimodal_configs_separate():
    wrapper, original = _make_wrapper()
    assert wrapper.vlm_config is original
    assert wrapper.config is not original.text_config
    assert wrapper.config._welmv4_vlm_target_only
    assert not hasattr(original.text_config, "_welmv4_vlm_target_only")
    assert wrapper._image_token_id() == 77
    assert (
        wrapper.get_model_config_for_expert_location(original) is original.text_config
    )


def test_vision_weights_do_not_enter_text_loader_and_missing_weights_fail():
    wrapper, _ = _make_wrapper()
    wrapper.load_weights(
        iter(
            [
                ("vision_encoder.weight", torch.tensor([1.0, 2.0])),
                ("text_weight", torch.tensor([3.0])),
                ("vision_projector.weight", torch.tensor([4.0, 5.0])),
            ]
        )
    )
    assert [name for name, _ in wrapper.text_weights] == ["text_weight"]
    torch.testing.assert_close(wrapper.vision_encoder.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        wrapper.vision_projector.weight, torch.tensor([4.0, 5.0])
    )
    with pytest.raises(ValueError, match="Missing WeLM vision weights"):
        wrapper.load_weights(iter([("text_weight", torch.tensor([3.0]))]))
    with pytest.raises(ValueError, match="Unexpected WeLM vision weight"):
        wrapper.load_weights(iter([("vision_encoder.typo", torch.tensor([3.0]))]))


def test_fused_vision_qkv_uses_its_own_tp_weight_loader():
    wrapper, _ = _make_wrapper()
    wrapper.vision_encoder.attn = nn.Module()
    wrapper.vision_encoder.attn.qkv_proj = nn.Linear(2, 6, bias=False)
    parameter = wrapper.vision_encoder.attn.qkv_proj.weight
    loader = Mock()
    parameter.weight_loader = loader
    tensor = torch.arange(12).reshape(6, 2).float()
    wrapper.load_weights(
        iter(
            [
                ("vision_encoder.weight", torch.zeros(2)),
                ("vision_projector.weight", torch.zeros(2)),
                ("vision_encoder.attn.qkv.weight", tensor),
            ]
        )
    )
    assert loader.call_count == 1
    assert loader.call_args.args[0] is parameter
    assert loader.call_args.args[1] is tensor
    assert wrapper.text_weights == []


def test_image_rows_override_oe_and_backbone_does_not_fuse_twice():
    wrapper, _ = _make_wrapper()
    embed = nn.Embedding(100, 4)
    embed.weight.data.fill_(2)
    oe = Mock(side_effect=lambda ids, batch, base: base + 10)
    wrapper.model = SimpleNamespace(
        embed_tokens=embed, oe_grams=[2], _compute_oe_embedding=oe
    )
    item = SimpleNamespace(pad_value=100001, is_image=lambda: True)
    batch = SimpleNamespace(
        mm_inputs=[SimpleNamespace(mm_items=[item])],
        input_embeds=None,
        ngram_embedding_info=object(),
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        contains_image_inputs=lambda: True,
    )
    wrapper._get_image_embedding_and_mask = Mock(
        return_value=(
            torch.full((2, 4), 99.0),
            torch.tensor([[False], [True], [True], [False]]),
        )
    )
    result = wrapper.forward(
        torch.tensor([1, 100001, 100001, 2]), torch.arange(4), batch
    )
    assert oe.call_count == 1
    assert result["skip_oe_fusion"] is True
    torch.testing.assert_close(result["input_ids"], torch.tensor([1, 77, 77, 2]))
    torch.testing.assert_close(
        result["input_embeds"][:, 0], torch.tensor([12.0, 99.0, 99.0, 12.0])
    )


def test_decode_and_text_only_use_the_normal_backbone_path():
    wrapper, _ = _make_wrapper()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        contains_image_inputs=lambda: False,
    )
    result = wrapper.forward(torch.tensor([1, 2]), torch.arange(2), batch)
    assert "skip_oe_fusion" not in result
    assert result["input_embeds"] is None


def test_image_embedding_metadata_selects_correct_requests_in_mixed_batch():
    wrapper, _ = _make_wrapper()
    embedding = torch.zeros(3, 4)
    mask = torch.ones(3, 1, dtype=torch.bool)
    get_embedding = Mock(return_value=(embedding, mask, torch.empty(0)))
    method_globals = wrapper._get_image_embedding_and_mask.__func__.__globals__
    method_globals["get_embedding_and_mask"] = get_embedding
    items = [
        SimpleNamespace(is_image=lambda: True, offsets=[(2, 3)], pad_value=100001),
        SimpleNamespace(is_image=lambda: True, offsets=[(5, 5)], pad_value=100002),
    ]
    requests = [SimpleNamespace(mm_items=[item]) for item in items]
    batch = SimpleNamespace(
        extend_prefix_lens_cpu=[1, 7, 4], extend_seq_lens_cpu=[3, 2, 2]
    )
    result = wrapper._get_image_embedding_and_mask(
        torch.tensor([1, 100001, 100001, 8, 9, 2, 100002]),
        batch,
        requests,
        [0, 2],
        items,
    )
    assert result[0] is embedding and result[1] is mask
    kwargs = get_embedding.call_args.kwargs
    assert kwargs["prefix_length"] == [1, 4]
    assert kwargs["extend_length"] == [3, 2]
    assert kwargs["items_size"] == [0, 1, 2]
    assert kwargs["items_offset_list"] == [[(2, 3)], [(5, 5)]]


def test_images_with_external_embeddings_fail_instead_of_skipping_vision():
    wrapper, _ = _make_wrapper()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        contains_image_inputs=lambda: True,
    )
    with pytest.raises(NotImplementedError, match="externally supplied"):
        wrapper.forward(
            torch.tensor([100001]),
            torch.tensor([0]),
            batch,
            input_embeds=torch.zeros(1, 4),
        )


_MM_CHUNK_DEFINITIONS = {
    "get_embedding_chunk", "_get_precomputed_embedding", "_flatten_embedding_result",
    "_can_skip_pre_embed_feature_move", "_move_items_to_device",
    "_acknowledge_deferred_cuda_ipc_cache_hits", "_get_chunked_embedding_full",
    "PerImageRequestInfo", "_batch_encode_per_image_misses",
    "_get_chunked_embedding_by_item", "_assemble_per_image_chunk",
    "_get_chunked_prefill_embedding", "_get_multimodal_mask",
    "_adjust_embedding_length", "get_embedding_and_mask",
}


def _load_real_mm_chunk_cache(cache_bytes):
    """Keep production slicing/LRU code; isolate serving imports and NPU I/O."""
    cache_path = ROOT / "python/sglang/srt/mem_cache/multimodal_cache.py"
    cache_tree = ast.parse(cache_path.read_text())
    cache_tree.body = [
        node for node in cache_tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module.startswith("sglang."))
    ]
    future = ast.parse("from __future__ import annotations").body[0]
    cache_tree.body.insert(0, future)
    cache_module = types.ModuleType("_welm_vlm_test_real_cache")
    sys.modules[cache_module.__name__] = cache_module
    exec(compile(cache_tree, str(cache_path), "exec"), cache_module.__dict__)
    cache = cache_module.MultiModalStaticCache(cache_bytes)

    mm_path = ROOT / "python/sglang/srt/managers/mm_utils.py"
    mm_tree = ast.parse(mm_path.read_text())
    mm_tree.body = [future] + [
        node for node in mm_tree.body
        if getattr(node, "name", None) in _MM_CHUNK_DEFINITIONS
    ]
    mm_module = types.ModuleType("_welm_vlm_test_real_mm_utils")
    sys.modules[mm_module.__name__] = mm_module
    # Select the actual per-image NPU branch; only stream synchronization and
    # pixel transfer are stand-ins, so all tests still use real CPU tensors.
    torch_proxy = SimpleNamespace(
        **{key: value for key, value in vars(torch).items() if key != "npu"},
        npu=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(synchronize=lambda: None)
        ),
    )
    mm_module.__dict__.update(
        torch=torch_proxy,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        embedding_cache=cache,
        EmbeddingResult=cache_module.EmbeddingResult,
        MultiModalStaticCache=cache_module.MultiModalStaticCache,
        EVSEmbeddingResult=type("UnsupportedEVS", (), {}),
        _is_npu=True,
        _is_hip=False,
        logger=logging.getLogger(__name__),
        get_parallel=lambda: SimpleNamespace(attn_tp_rank=0, tp_size=4),
        get_schedule=lambda: SimpleNamespace(chunked_prefill_size=4),
    )
    exec(compile(mm_tree, str(mm_path), "exec"), mm_module.__dict__)
    moved_storage = []

    def move_pixels(items, device):
        # Match _move_items_to_device's mutation and new-storage contract on
        # CPU. Weak refs prove the temporary storage is not retained by Req.
        for item in items:
            item.feature = item.feature.clone()
            moved_storage.append(weakref.ref(item.feature))

    mm_module._move_items_to_device = move_pixels
    return mm_module, cache, moved_storage


class _CacheTestEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail = False

    def forward(self, pixels, grid):
        self.calls.append((pixels.clone(), grid.clone()))
        if self.fail:
            raise RuntimeError("injected vision encoding failure")
        # The real ViT's numerical operations have separate tests above. Here
        # each pixel row represents one output embedding with a unique value.
        return pixels


def _make_chunk_cache_wrapper(cache_bytes=4096):
    wrapper, _ = _make_wrapper()
    encoder = _CacheTestEncoder()
    wrapper.vision_encoder = encoder
    wrapper.vision_projector = nn.Identity()
    embedding = nn.Embedding(100, 4)
    embedding.weight.data.fill_(2)
    oe = Mock(side_effect=lambda ids, batch, base: base + 10)
    wrapper.model = SimpleNamespace(
        embed_tokens=embedding, oe_grams=[2, 2, 3, 3], _compute_oe_embedding=oe
    )
    mm_module, cache, moved = _load_real_mm_chunk_cache(cache_bytes)
    wrapper._get_image_embedding_and_mask.__func__.__globals__[
        "get_embedding_and_mask"
    ] = mm_module.get_embedding_and_mask
    return wrapper, encoder, oe, cache, moved


def _cache_test_image(pad, start, rows, value):
    pixels = (
        torch.arange(value, value + rows).float().unsqueeze(1).expand(-1, 4).clone()
    )
    return SimpleNamespace(
        pad_value=pad,
        hash=pad,
        offsets=[(start, start + rows - 1)],
        feature=pixels,
        original_cpu_feature=pixels,
        image_grid_thw=torch.tensor([[1, 2, rows * 2]]),
        precomputed_embeddings=None,
        is_image=lambda: True,
        acknowledge_deferred_cuda_ipc_feature=lambda count: None,
    )


def _chunk_forward(
    wrapper, tokens, items, prefix, *, decode=False, ngram_table=None
):
    mm = SimpleNamespace(mm_items=items)
    batch = SimpleNamespace(
        mm_inputs=[mm], input_embeds=None,
        ngram_embedding_info=(
            object() if ngram_table is None else SimpleNamespace(
                token_table=ngram_table,
                req_lens=torch.tensor([len(tokens)], dtype=torch.int32),
                column_starts=torch.tensor([prefix], dtype=torch.int32),
            )
        ),
        batch_size=1,
        req_pool_indices=torch.tensor([0]),
        num_token_non_padded_cpu=len(tokens),
        extend_start_loc=torch.tensor([0]),
        extend_prefix_lens_cpu=[prefix], extend_seq_lens_cpu=[len(tokens)],
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_target_verify=lambda: False
        ),
        contains_image_inputs=lambda: bool(items),
    )
    result = wrapper.forward(
        torch.tensor(tokens), torch.arange(prefix, prefix + len(tokens)), batch
    )
    return result, batch


def _assert_retained_cpu_pixels(*items):
    for item in items:
        assert item.feature is item.original_cpu_feature
        assert item.feature.device.type == "cpu"


def test_real_cache_chunks_cross_image_boundaries_and_decode_keeps_cpu_pixels():
    wrapper, encoder, oe, cache, moved = _make_chunk_cache_wrapper()
    first = _cache_test_image(100001, 1, 3, 100)
    second = _cache_test_image(100002, 5, 3, 200)
    prompt = [1, first.pad_value, first.pad_value, first.pad_value, 2,
              second.pad_value, second.pad_value, second.pad_value, 3]
    expected = [12, 100, 101, 102, 12, 200, 201, 202, 12]
    for prefix, length in ((0, 2), (2, 4), (6, 3)):
        result, batch = _chunk_forward(
            wrapper, prompt[prefix:prefix + length], [first, second], prefix
        )
        assert result["skip_oe_fusion"] is True
        torch.testing.assert_close(
            result["input_embeds"][:, 0],
            torch.tensor(expected[prefix:prefix + length]).float(),
        )
        assert batch.mm_inputs is None
        _assert_retained_cpu_pixels(first, second)
        assert all(ref() is None for ref in moved)
    assert oe.call_count == 3
    assert len(encoder.calls) == 2
    assert cache.has(first.hash) and cache.has(second.hash)
    decoded, _ = _chunk_forward(wrapper, [9], [first, second], len(prompt), decode=True)
    assert decoded["input_embeds"] is None
    assert "skip_oe_fusion" not in decoded
    assert oe.call_count == 3 and len(encoder.calls) == 2
    _assert_retained_cpu_pixels(first, second)


@pytest.mark.parametrize("prefix", [2, 4])
@pytest.mark.parametrize("warm_embedding_cache", [False, True])
def test_real_cache_prefix_hit_inside_or_after_image(prefix, warm_embedding_cache):
    wrapper, encoder, oe, _, _ = _make_chunk_cache_wrapper()
    image = _cache_test_image(100001, 1, 3, 100)
    prompt = [1, image.pad_value, image.pad_value, image.pad_value, 2, 3]
    expected = [12, 100, 101, 102, 12, 12]
    if warm_embedding_cache:
        _chunk_forward(wrapper, prompt, [image], 0)
    before = len(encoder.calls)
    result, _ = _chunk_forward(wrapper, prompt[prefix:], [image], prefix)
    assert result["skip_oe_fusion"] is True
    torch.testing.assert_close(
        result["input_embeds"][:, 0], torch.tensor(expected[prefix:]).float()
    )
    expected_extra = int(prefix == 2 and not warm_embedding_cache)
    assert len(encoder.calls) == before + expected_extra
    assert oe.call_count == 1 + int(warm_embedding_cache)
    _assert_retained_cpu_pixels(image)


def test_real_lru_eviction_reencodes_next_chunk_from_original_cpu_pixels():
    # One image embedding fits; the intervening request must evict it.
    wrapper, encoder, oe, cache, moved = _make_chunk_cache_wrapper(cache_bytes=48)
    first = _cache_test_image(100001, 1, 3, 100)
    other = _cache_test_image(100002, 0, 3, 200)
    _chunk_forward(wrapper, [1, first.pad_value], [first], 0)
    assert cache.has(first.hash)
    _chunk_forward(wrapper, [other.pad_value] * 3, [other], 0)
    assert cache.has(other.hash) and not cache.has(first.hash)
    result, _ = _chunk_forward(wrapper, [first.pad_value] * 2, [first], 2)
    torch.testing.assert_close(result["input_embeds"][:, 0], torch.tensor([101., 102.]))
    assert len(encoder.calls) == 3 and oe.call_count == 3
    torch.testing.assert_close(encoder.calls[0][0], encoder.calls[2][0])
    assert cache.has(first.hash) and not cache.has(other.hash)
    _assert_retained_cpu_pixels(first, other)
    assert all(ref() is None for ref in moved)


def test_real_mm_utils_mixed_text_images_and_nonoverlapping_image_request():
    wrapper, encoder, oe, _, _ = _make_chunk_cache_wrapper()
    first = _cache_test_image(100001, 1, 3, 100)
    second = _cache_test_image(100002, 1, 3, 200)
    requests = [None, SimpleNamespace(mm_items=[first]),
                SimpleNamespace(mm_items=[second]), SimpleNamespace(mm_items=[second])]
    batch = SimpleNamespace(
        mm_inputs=requests, input_embeds=None, ngram_embedding_info=object(),
        extend_prefix_lens_cpu=[3, 2, 0, 1], extend_seq_lens_cpu=[2, 2, 1, 2],
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        contains_image_inputs=lambda: True,
    )
    ids = torch.tensor([11, 12, first.pad_value, first.pad_value, 21,
                        second.pad_value, second.pad_value])
    result = wrapper.forward(ids, torch.tensor([3, 4, 2, 3, 0, 1, 2]), batch)
    torch.testing.assert_close(
        result["input_embeds"][:, 0],
        torch.tensor([12.0, 12.0, 101.0, 102.0, 12.0, 200.0, 201.0]),
    )
    assert oe.call_count == 1 and result["skip_oe_fusion"] is True
    assert len(encoder.calls) == 2
    _assert_retained_cpu_pixels(first, second)


def test_vision_failure_restores_pixels_and_next_attempt_can_recompute():
    wrapper, encoder, _, cache, moved = _make_chunk_cache_wrapper()
    image = _cache_test_image(100001, 0, 3, 100)
    encoder.fail = True
    with pytest.raises(RuntimeError, match="injected vision"):
        _chunk_forward(wrapper, [image.pad_value], [image], 0)
    _assert_retained_cpu_pixels(image)
    assert not cache.has(image.hash)
    encoder.fail = False
    result, _ = _chunk_forward(wrapper, [image.pad_value], [image], 0)
    torch.testing.assert_close(result["input_embeds"][:, 0], torch.tensor([100.]))
    assert cache.has(image.hash)
    _assert_retained_cpu_pixels(image)


def test_real_oe_and_visual_embeddings_match_full_prefill_across_chunks():
    backbone_path = ROOT / "python/sglang/srt/models/welmv4.py"
    tree = ast.parse(backbone_path.read_text())
    model_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen2MoeModel"
    )
    oe_method = next(
        node for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_oe_embedding"
    )
    hash_method = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "hash_input_ids_vectorized"
    )
    extracted = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0],
              hash_method, oe_method],
        type_ignores=[],
    )
    namespace = {"torch": torch, "_is_npu": False}
    exec(compile(extracted, str(backbone_path), "exec"), namespace)
    wrapper, _, _, _, _ = _make_chunk_cache_wrapper()
    model = wrapper.model
    model.vocab_size = 100
    model.oe_vocab_sizes = [17, 19, 23, 29]
    # Small deterministic OE tables exercise the production 2/3-gram hash
    # without allocating the checkpoint's multi-million-row tables.
    model.oe_embed = [nn.Embedding(size, 1) for size in model.oe_vocab_sizes]
    for table in model.oe_embed:
        table.weight.data.copy_(torch.arange(table.num_embeddings).reshape(-1, 1))
    model.oe_gate_up_proj = lambda values: (values, None)
    model._compute_oe_embedding = Mock(
        side_effect=MethodType(namespace["_compute_oe_embedding"], model)
    )
    first = _cache_test_image(100001, 1, 3, 100)
    second = _cache_test_image(100002, 5, 3, 200)
    prompt = [1, first.pad_value, first.pad_value, first.pad_value, 2,
              second.pad_value, second.pad_value, second.pad_value, 3]
    table = torch.tensor([[1, 77, 77, 77, 2, 77, 77, 77, 3]], dtype=torch.int32)
    full, _ = _chunk_forward(wrapper, prompt, [first, second], 0, ngram_table=table)
    chunks = []
    for prefix, length in ((0, 2), (2, 4), (6, 3)):
        chunk, _ = _chunk_forward(
            wrapper, prompt[prefix:prefix + length], [first, second], prefix,
            ngram_table=table,
        )
        chunks.append(chunk["input_embeds"])
        assert chunk["skip_oe_fusion"] is True
    torch.testing.assert_close(torch.cat(chunks), full["input_embeds"], atol=0, rtol=0)
    prefix_hit, _ = _chunk_forward(
        wrapper, prompt[3:], [first, second], 3, ngram_table=table
    )
    torch.testing.assert_close(
        prefix_hit["input_embeds"], full["input_embeds"][3:], atol=0, rtol=0
    )
    assert model._compute_oe_embedding.call_count == 5
    _assert_retained_cpu_pixels(first, second)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
