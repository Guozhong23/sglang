"""Metadata cache equivalence and lifetime contracts, runnable on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from test_welmv4_vlm_vision import CACHE, OPS, _load_definitions, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RopeBlock(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

    def forward(self, hidden, cu_seqlens, position_embeddings):
        q = hidden.reshape(-1, 2, 72)
        return OPS.welmv4_vision_apply_rope(q, q, *position_embeddings)[0].flatten(1)


def _encoder(cap=0):
    patch = _load_definitions(
        "_patch_embed_forward_matmul",
        "WeLMV4VisionPatchEmbed",
        _vision_max_patches=lambda: cap,
    )
    pos = _load_definitions("WeLMV4VisionPositionEmbedding")
    cls = _load_definitions(
        "WeLMV4VisionEncoder",
        WeLMV4VisionPatchEmbed=patch.WeLMV4VisionPatchEmbed,
        WeLMV4VisionPositionEmbedding=pos.WeLMV4VisionPositionEmbedding,
        WeLMV4VisionBlock=_RopeBlock,
        _vision_max_patches=lambda: cap,
        add_prefix=lambda name, prefix: name,
        print_warning_once=lambda message: None,
    ).WeLMV4VisionEncoder
    config = SimpleNamespace(
        hidden_size=144,
        num_attention_heads=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        num_hidden_layers=1,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
    )
    encoder = cls(config).bfloat16().eval()
    with torch.no_grad():
        encoder.pos_embed.weight.normal_()
    return encoder


def _reference_rope(encoder, grid):
    head_dim = encoder.config.hidden_size // encoder.config.num_attention_heads
    rope_dim = head_dim // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    max_grid = int(grid[:, 1:].max().item())
    frequencies = torch.outer(torch.arange(max_grid).float(), inv)
    merge = encoder.config.spatial_merge_size
    ids = []
    for t, h, w in grid.tolist():
        axes = [
            torch.arange(h)[:, None].expand(h, w),
            torch.arange(w)[None, :].expand(h, w),
        ]
        axes = [
            axis.reshape(h // merge, merge, w // merge, merge)
            .permute(0, 2, 1, 3)
            .flatten()
            for axis in axes
        ]
        ids.append(torch.stack(axes, -1).repeat(t, 1))
    ids = torch.cat(ids)
    return tuple(
        torch.cat([fn(frequencies)[ids[:, 0]], fn(frequencies)[ids[:, 1]]], -1)
        for fn in (torch.cos, torch.sin)
    )


@pytest.mark.parametrize(
    "grids", [[[1, 4, 6]], [[2, 4, 6], [1, 6, 2]], [[1, 2, 2], [3, 4, 4]]]
)
@torch.inference_mode()
def test_cached_and_uncached_metadata_are_exact(grids):
    encoder = _encoder()
    grid = torch.tensor(grids)
    expected = (
        encoder.pos_embed(grid),
        encoder._cu_seqlens(grid),
        *_reference_rope(encoder, grid),
    )
    first = encoder._metadata(CACHE.vision_grid_key(grid))
    second = encoder._metadata(CACHE.vision_grid_key(grid))
    for reference, cold, hot in zip(expected, first, second):
        torch.testing.assert_close(cold, reference, atol=0, rtol=0)
        assert hot is cold
    assert first[2].dtype == torch.float32
    assert first[2].shape[-1] == 36
    assert first[1].dtype == torch.int32


@pytest.mark.parametrize("cap", [0, 28])
@torch.inference_mode()
def test_forward_cache_and_chunking_preserve_output_and_do_not_cache_pixels(cap):
    torch.manual_seed(733)
    encoder = _encoder(cap)
    grid = torch.tensor([[2, 4, 6], [1, 2, 2]])
    pixels = torch.randn(52, 12).bfloat16()
    encoder._metadata_cache.max_entries = 0
    reference = encoder(pixels, grid)
    encoder._metadata_cache.max_entries = 2
    encoder.pos_embed.forward = Mock(wraps=encoder.pos_embed.forward)
    cold, hot = encoder(pixels, grid), encoder(pixels, grid)
    torch.testing.assert_close(cold, reference, atol=0, rtol=0)
    torch.testing.assert_close(hot, reference, atol=0, rtol=0)
    assert encoder.pos_embed.forward.call_count == 1
    changed = encoder(pixels + 1, grid)
    assert not torch.equal(changed, reference)
    assert encoder.pos_embed.forward.call_count == 1


@torch.no_grad()
def test_cache_invalidates_after_weight_mutation_load_and_dtype_move():
    encoder = _encoder()
    grid = ((1, 4, 4),)
    first = encoder._metadata(grid)
    encoder.pos_embed.weight.add_(1)
    second = encoder._metadata(grid)
    assert second[0] is not first[0]
    assert not torch.equal(first[0], second[0])
    assert len(encoder._metadata_cache.entries) == 1
    encoder.load_state_dict(encoder.state_dict())
    assert not encoder._metadata_cache.entries
    encoder._metadata(grid)
    encoder.float()
    assert not encoder._metadata_cache.entries
    assert encoder._metadata(grid)[0].dtype == torch.float32


@torch.inference_mode()
def test_inference_weights_explicit_clear_and_grid_order():
    encoder = _encoder()
    grid = ((1, 2, 4), (1, 4, 2))
    a = encoder._metadata(grid)
    b = encoder._metadata(grid[::-1])
    assert len(encoder._metadata_cache.entries) == 2
    assert not torch.equal(a[0], b[0])
    encoder.pos_embed.weight.data.add_(1)
    encoder.clear_metadata_cache()
    assert not torch.equal(a[0], encoder._metadata(grid)[0])


def test_grad_and_training_paths_bypass_cache():
    encoder = _encoder()
    grid = ((1, 2, 2),)
    with torch.no_grad():
        encoder._metadata(grid)
    result = encoder._metadata(grid)
    assert result[0].requires_grad
    result[0].float().sum().backward()
    assert encoder.pos_embed.weight.grad is not None
    encoder.train()
    with torch.no_grad():
        assert encoder._metadata(grid)[0] is not encoder._metadata(grid)[0]


def test_lru_entry_limit_byte_limit_and_oversize():
    cache = CACHE.VisionMetadataCache(max_entries=2, max_bytes=24)
    build = lambda: (torch.zeros(2),)
    a = cache.get_or_build("a", 1, build, enabled=True)
    cache.get_or_build("b", 1, build, enabled=True)
    assert cache.get_or_build("a", 1, build, enabled=True) is a
    cache.get_or_build("c", 1, build, enabled=True)
    assert list(cache.entries) == ["a", "c"]
    cache.max_entries = 10
    cache.get_or_build("d", 1, lambda: (torch.zeros(4),), enabled=True)
    assert list(cache.entries) == ["c", "d"]
    assert cache.current_bytes == 24
    cache.get_or_build("huge", 1, lambda: (torch.zeros(100),), enabled=True)
    assert "huge" not in cache.entries
    cache.get_or_build("new_version", 2, build, enabled=True)
    assert list(cache.entries) == ["new_version"]
    cache.clear()
    assert cache.current_bytes == 0


def test_env_disable_and_limits(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_VISION_METADATA_CACHE", "0")
    cache = CACHE.VisionMetadataCache.from_env()
    for _ in range(2):
        cache.get_or_build("a", 1, lambda: (torch.zeros(1),), enabled=True)
    assert not cache.entries
    monkeypatch.setenv("SGLANG_VLM_VISION_METADATA_CACHE", "1")
    monkeypatch.setenv("SGLANG_VLM_VISION_METADATA_CACHE_ENTRIES", "3")
    monkeypatch.setenv("SGLANG_VLM_VISION_METADATA_CACHE_MB", "2")
    cache = CACHE.VisionMetadataCache.from_env()
    assert cache.max_entries == 3 and cache.max_bytes == 2 * 1024**2
    with pytest.raises(ValueError):
        CACHE.VisionMetadataCache(max_bytes=-1)
