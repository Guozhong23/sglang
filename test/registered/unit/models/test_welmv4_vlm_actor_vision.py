"""Actor visual math contracts, with legacy WeLM and image-DP regression checks."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_welmv4_vlm_dp import DP
from test_welmv4_vlm_metadata_cache import _encoder, _reference_rope
from test_welmv4_vlm_ulysses import _Collectives, _types
from test_welmv4_vlm_vision import (
    OPS,
    _load_definitions,
    _make_wrapper,
    register_cpu_ci,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("chunk_size", [0, 1, 3, 32])
def test_actor_patch_projection_uses_bias_epilogue(chunk_size):
    project = _load_definitions(
        "_patch_embed_forward_matmul"
    )._patch_embed_forward_matmul
    torch.manual_seed(77)
    x = torch.randn(7, 24).bfloat16()
    weight = torch.randn(12, 3, 2, 2, 2).bfloat16()
    bias = torch.randn(12).bfloat16()
    expected = F.linear(x, weight.flatten(1), bias)
    actual = project(
        x, weight, bias, 2, 2, 2, 3, 12, chunk_size=chunk_size, actor_match=True
    )
    torch.testing.assert_close(actual.squeeze(1), expected, atol=0, rtol=0)
    legacy = F.linear(x, weight.flatten(1)) + bias
    assert not torch.equal(expected, legacy), "fixture must expose the bias boundary"


@torch.no_grad()
def test_actor_position_embedding_rounds_each_interpolation_axis():
    cls = _load_definitions(
        "WeLMV4VisionPositionEmbedding"
    ).WeLMV4VisionPositionEmbedding
    config = SimpleNamespace(
        hidden_size=5,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        welm_visual_actor_match=True,
    )
    position = cls(config).bfloat16()
    torch.manual_seed(77)
    position.weight.normal_()
    height, width, side = 14, 18, 48
    # Independent per-pixel bilinear reference: both fractional axes cross
    # their BF16 boundary before the four interpolation products are formed.
    rows = []
    for row in range(height):
        y = torch.tensor(row).float() * (side - 1) / (height - 1)
        lo_y = int(y.floor())
        hi_y = min(lo_y + 1, side - 1)
        fy = (y - lo_y).bfloat16()
        for col in range(width):
            x = torch.tensor(col).float() * (side - 1) / (width - 1)
            lo_x = int(x.floor())
            hi_x = min(lo_x + 1, side - 1)
            fx = (x - lo_x).bfloat16()
            corners = position.weight[
                [
                    lo_y * side + lo_x,
                    lo_y * side + hi_x,
                    hi_y * side + lo_x,
                    hi_y * side + hi_x,
                ]
            ]
            weights = torch.stack(
                ((1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx)
            )
            rows.append((corners * weights[:, None]).sum(0))
    expected = (
        torch.stack(rows)
        .view(height // 2, 2, width // 2, 2, 5)
        .permute(0, 2, 1, 3, 4)
        .reshape(-1, 5)
    )
    grid = torch.tensor([[2, height, width]])
    actual = position(grid)
    torch.testing.assert_close(actual, expected.repeat(2, 1), atol=0, rtol=0)
    position._actor_match = False
    assert not torch.equal(actual, position(grid))


@torch.inference_mode()
def test_actor_rope_table_rounding_is_cached_without_disabling_fp32_kernel_contract():
    encoder = _encoder()
    encoder._actor_match = True
    grid = torch.tensor([[1, 4, 6]])
    expected = tuple(x.bfloat16().float() for x in _reference_rope(encoder, grid))
    first = encoder._metadata(grid)
    second = encoder._metadata(grid)
    for actual, reference, cached in zip(first[2:], expected, second[2:]):
        assert actual.dtype == torch.float32
        assert cached is actual
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.manual_seed(331)
    q, k = [torch.randn(24, 2, 72).bfloat16() for _ in range(2)]
    expected_qk = OPS.welmv4_vision_apply_rope(q, k, *(x.bfloat16() for x in expected))
    actual_qk = OPS.welmv4_vision_apply_rope(q, k, *first[2:])
    for actual, reference in zip(actual_qk, expected_qk):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    encoder._actor_match = False
    assert not torch.equal(encoder._metadata(grid)[3], first[3])


def test_wrapper_selects_actor_visual_math_without_mutating_checkpoint():
    wrapper, config = _make_wrapper()
    config.image_token = "<|welm_image_pad|>"
    namespace = type(wrapper).__init__.__globals__
    original = namespace["WeLMV4VisionEncoder"]
    captured = []

    def capture(vision_config, **kwargs):
        captured.append(vision_config)
        return original(vision_config, **kwargs)

    namespace["WeLMV4VisionEncoder"] = capture
    type(wrapper)(config)
    assert captured[0].welm_visual_actor_match is True
    assert captured[0] is not config.vision_config
    assert not hasattr(config.vision_config, "welm_visual_actor_match")


@pytest.mark.parametrize("actor_match", [False, True])
@torch.inference_mode()
def test_projector_keeps_checkpoint_specific_batching(actor_match):
    types = _types(None, 0)
    config = SimpleNamespace(
        hidden_size=8,
        spatial_merge_size=2,
        out_hidden_size=4,
        welm_visual_actor_match=actor_match,
    )
    projector = types.WeLMV4VisionProjector(config).bfloat16().eval()
    hidden = torch.randn(24, 8).bfloat16()
    calls = []
    with projector.register_forward_pre_hook(
        lambda module, args: calls.append(len(args[0]))
    ):
        actual = projector.forward_images(hidden, ((1, 2, 4), (2, 2, 4)))
    assert calls == ([8, 16] if actor_match else [24])
    expected = torch.cat([projector(x) for x in hidden.split([8, 16])])
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.02)


@pytest.mark.parametrize("world", [1, 4])
def test_actor_dp_projects_each_owned_image_then_restores_global_order(world):
    torch.set_num_threads(1)
    types = _types(None, 0)
    config = SimpleNamespace(
        hidden_size=8,
        spatial_merge_size=2,
        out_hidden_size=4,
        welm_visual_actor_match=True,
    )
    torch.manual_seed(128)
    reference = types.WeLMV4VisionProjector(config).bfloat16().eval()
    grids = ((1, 2, 4), (1, 4, 4), (1, 2, 6), (1, 2, 2), (1, 2, 4))
    items = [
        SimpleNamespace(
            feature=torch.randn(t * h * w, 8).bfloat16(),
            image_grid_thw=torch.tensor([[t, h, w]]),
        )
        for t, h, w in grids
    ]

    class IdentityEncoder:
        vision_dp_input_spec = DP.VisionDPInputSpec((2, 2), (8,), 4)
        dtype, device = torch.bfloat16, torch.device("cpu")

        def __call__(self, pixels, grid):
            return pixels

    groups = _Collectives(world)
    calls = [[] for _ in range(world)]

    @torch.inference_mode()
    def run(rank):
        projector = types.WeLMV4VisionProjector(config).bfloat16().eval()
        projector.load_state_dict(reference.state_dict())
        with projector.register_forward_pre_hook(
            lambda module, args: calls[rank].append(len(args[0]))
        ):
            return DP.encode_vision_dp(
                items, IdentityEncoder(), projector, groups.group(rank)
            )

    with torch.inference_mode():
        expected = [reference(item.feature) for item in items]
    with ThreadPoolExecutor(max_workers=world) as pool:
        results = list(pool.map(run, range(world)))
    plan = DP.VisionDPPlan(grids, 2, world, 0)
    for rank, outputs in enumerate(results):
        assert calls[rank] == [plan.patch_counts[i] for i in plan.assignments[rank]]
        for output, target in zip(outputs, expected):
            torch.testing.assert_close(output, target, atol=0, rtol=0)


def test_actor_patch16_temporal2_encoder_preserves_ulysses_math():
    torch.set_num_threads(1)
    torch.manual_seed(415)
    config = SimpleNamespace(
        hidden_size=144,
        num_attention_heads=2,
        intermediate_size=48,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        num_hidden_layers=2,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        hidden_act="gelu_pytorch_tanh",
        out_hidden_size=16,
        welm_visual_actor_match=True,
    )
    baseline_types = _types(None, 0)
    baseline = baseline_types.WeLMV4VisionEncoder(config).bfloat16().eval()
    with torch.no_grad():
        baseline.pos_embed.weight.normal_(std=0.1)
    assert baseline.patch_embed.proj.weight.shape == (144, 3, 2, 16, 16)
    assert baseline.pos_embed.weight.shape == (2304, 144)
    grid = torch.tensor([[1, 2, 2], [2, 2, 4]])
    pixels = torch.randn(20, 1536).bfloat16()
    groups = _Collectives(2)
    encoders = []
    for rank in range(2):
        encoder = (
            _types(groups.group(rank), 0)
            .WeLMV4VisionEncoder(config)
            .bfloat16()
            .eval()
        )
        encoder.load_state_dict(baseline.state_dict())
        encoders.append(encoder)

    @torch.inference_mode()
    def run(rank):
        return encoders[rank](pixels, grid)

    with torch.inference_mode():
        expected = baseline(pixels, grid)
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(run, range(2)))
    for result in actual:
        torch.testing.assert_close(result, expected, atol=0.015625, rtol=0.02)
