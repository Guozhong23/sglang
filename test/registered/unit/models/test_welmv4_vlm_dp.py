"""Whole-image DP math and real Gloo exchanges using production planning.

Tiny BF16 encoders execute real WeLM blocks/projectors on CPU; the NPU attention
backend is replaced by segmented PyTorch attention. Gloo tests use four actual
processes, including idle ranks, and do not require serving packages.
"""

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_welmv4_vlm_ulysses import _Collectives, _runtime, _types
from test_welmv4_vlm_ulysses_layout import ROOT, ULYSSES
from test_welmv4_vlm_vision import _make_wrapper, register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
spec = importlib.util.spec_from_file_location(
    "welm_vision_dp_test", ROOT / "python/sglang/srt/layers/welmv4_vision_dp.py"
)
DP = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = DP
spec.loader.exec_module(DP)


def _items(grids, width=12, dtype=torch.bfloat16):
    return [
        SimpleNamespace(
            hash=100001 + index,
            feature=torch.randn(t * h * w, width).to(dtype),
            image_grid_thw=torch.tensor([[t, h, w]]),
        )
        for index, (t, h, w) in enumerate(grids)
    ]


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("grids", [((1, 2, 2),), ((1, 2, 4), (2, 2, 6), (1, 4, 4))])
def test_plan_assigns_whole_images_exactly_once(world, grids):
    plan = DP.VisionDPPlan(grids, 2, world, 0)
    assert sorted(index for rank in plan.assignments for index in rank) == list(
        range(len(grids))
    )
    assert sum(plan.rows_per_rank) == sum(t * h * w // 4 for t, h, w in grids)
    assert plan.assignments == DP.VisionDPPlan(grids, 2, world, world - 1).assignments
    if len(grids) == 1:
        assert plan.assignments[0] == (0,)
        assert all(not assigned for assigned in plan.assignments[1:])


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("cap", [0, 12])
@pytest.mark.parametrize("grids", [[[1, 2, 2]], [[1, 2, 6], [2, 2, 4], [1, 2, 2]]])
def test_owned_encoder_and_projector_match_full_image_reference(world, cap, grids):
    torch.set_num_threads(1)
    torch.manual_seed(812)
    config = SimpleNamespace(
        hidden_size=144,
        num_attention_heads=2,
        intermediate_size=48,
        spatial_merge_size=2,
        num_position_embeddings=16,
        num_hidden_layers=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        hidden_act="gelu_pytorch_tanh",
        out_hidden_size=16,
    )
    types = _types(None, cap)
    reference = (
        types.WeLMV4VisionEncoder(config, use_data_parallel=True).bfloat16().eval()
    )
    projector = (
        types.WeLMV4VisionProjector(config, use_data_parallel=True).bfloat16().eval()
    )
    with torch.no_grad():
        reference.pos_embed.weight.normal_(std=0.1)
    items = _items(grids)
    features = [item.feature for item in items]
    collectives = _Collectives(world)
    models = []
    calls = [[] for _ in range(world)]
    for rank in range(world):
        encoder = (
            types.WeLMV4VisionEncoder(config, use_data_parallel=True).bfloat16().eval()
        )
        projection = (
            types.WeLMV4VisionProjector(config, use_data_parallel=True)
            .bfloat16()
            .eval()
        )
        encoder.load_state_dict(reference.state_dict(), strict=True)
        projection.load_state_dict(projector.state_dict(), strict=True)
        encoder.register_forward_pre_hook(
            lambda model, args, rank=rank: calls[rank].append(
                (len(args[0]), args[1].clone())
            )
        )
        for linear in [
            *(
                linear
                for block in encoder.blocks
                for linear in (
                    block.attn.qkv_proj,
                    block.attn.proj,
                    block.mlp.fc1,
                    block.mlp.fc2,
                )
            ),
            projection.mlp[0],
            projection.mlp[2],
        ]:
            assert (linear.tp_size, linear.tp_rank) == (1, 0)
        models.append((encoder, projection))

    @torch.inference_mode()
    def run(rank):
        return DP.encode_vision_dp(items, *models[rank], collectives.group(rank))

    with torch.inference_mode():
        expected = projector(reference(torch.cat(features), torch.tensor(grids)))
        with ThreadPoolExecutor(max_workers=world) as pool:
            actual = list(pool.map(run, range(world)))
    plan = DP.VisionDPPlan(tuple(map(tuple, grids)), 2, world, 0)
    for rank, results in enumerate(actual):
        torch.testing.assert_close(
            torch.cat(results), expected, atol=0.015625, rtol=0.02
        )
        assert len(calls[rank]) == bool(plan.assignments[rank])
        if calls[rank]:
            count, owned_grid = calls[rank][0]
            assert count == sum(plan.patch_counts[i] for i in plan.assignments[rank])
            assert owned_grid.tolist() == [grids[i] for i in plan.assignments[rank]]
        # Each image cache entry owns storage; none pins the full gathered tensor.
        assert len({value.untyped_storage().data_ptr() for value in results}) == len(
            items
        )
    assert collectives.calls == (["ag"] if world > 1 else [])
    assert all(item.feature is feature for item, feature in zip(items, features))


@pytest.mark.parametrize("mode", ["0", "false", "1", "true", "bad"])
def test_startup_dp_switch_and_full_heads(monkeypatch, mode):
    group = _runtime(monkeypatch)
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_ULYSSES", "0")
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_DP", mode)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.welmv4_vision_ulysses", ULYSSES)
    config = SimpleNamespace(
        num_attention_heads=2, hidden_size=144, num_hidden_layers=27
    )
    if mode == "bad":
        with pytest.raises(ValueError, match="must be"):
            DP.create_vision_dp_group(config)
    else:
        assert DP.create_vision_dp_group(config) is (
            group if mode in {"1", "true"} else None
        )
        assert (
            ULYSSES.create_vision_ulysses_group(config, use_data_parallel=True) is None
        )


@pytest.mark.parametrize(
    "options,sp,message",
    [
        ({}, "1", "mutually exclusive"),
        ({"npu": False}, "0", "NPU"),
        ({"graph": True}, "0", "graphs"),
        ({"dp": True}, "0", "encoder DP"),
        ({"mismatched": True}, "0", "differ"),
        ({"attn_ranks": [0, 1]}, "0", "identical"),
    ],
)
def test_startup_rejects_unsafe_dp_modes(monkeypatch, options, sp, message):
    _runtime(monkeypatch, **options)
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_ULYSSES", sp)
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_DP", "1")
    config = SimpleNamespace(
        num_attention_heads=16, hidden_size=1152, num_hidden_layers=27
    )
    with pytest.raises(ValueError, match=message):
        ULYSSES.resolve_vision_parallel_group(config)


def test_cache_coordination_uses_union_of_misses_and_validates_order(monkeypatch):
    items = _items([[1, 2, 2], [1, 2, 4]])
    group = SimpleNamespace(world_size=2, cpu_group=object())

    def gather(peers, state, **kwargs):
        peers[:] = [state, (state[0], (False, True))]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert DP.synchronize_vision_dp_cache(items, [True, False], group=group) == [
        True,
        True,
    ]

    def different(peers, state, **kwargs):
        peers[:] = [state, (tuple(reversed(state[0])), state[1])]

    monkeypatch.setattr(dist, "all_gather_object", different)
    with pytest.raises(ValueError, match="order/grids"):
        DP.synchronize_vision_dp_cache(items, [True, True], group=group)


@pytest.mark.parametrize("enable_dp", [False, True])
def test_wrapper_defers_pixels_and_enables_cross_request_batching_only_for_dp(
    enable_dp,
):
    wrapper, _ = _make_wrapper()
    wrapper._vision_dp_group = object() if enable_dp else None
    get_embedding = Mock(return_value=(None, None, torch.tensor([1])))
    namespace = wrapper._get_image_embedding_and_mask.__func__.__globals__
    namespace["get_embedding_and_mask"] = get_embedding
    item = _items([[1, 2, 2]])[0]
    item.offsets, item.pad_value, item.is_image = [(0, 0)], 100001, lambda: True
    batch = SimpleNamespace(extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[1])
    wrapper._get_image_embedding_and_mask(
        torch.tensor([100001]), batch, [SimpleNamespace(mm_items=[item])], [0], [item]
    )
    kwargs = get_embedding.call_args.kwargs
    assert kwargs["defer_feature_move"] is True
    if enable_dp:
        assert kwargs["batch_encode_per_image"] is True
        assert callable(kwargs["synchronize_cache_misses"])
    else:
        assert "batch_encode_per_image" not in kwargs
        assert "synchronize_cache_misses" not in kwargs


def test_wrapper_constructs_both_encoder_and_projector_as_full_dp_replicas(monkeypatch):
    wrapper, config = _make_wrapper()
    namespace = type(wrapper).__init__.__globals__
    group = object()
    calls = []

    def vision(config, prefix, use_data_parallel):
        calls.append((prefix, use_data_parallel))
        return torch.nn.Identity()

    monkeypatch.setitem(namespace, "create_vision_dp_group", lambda *a, **kw: group)
    monkeypatch.setitem(namespace, "WeLMV4VisionEncoder", vision)
    monkeypatch.setitem(namespace, "WeLMV4VisionProjector", vision)
    created = type(wrapper)(config)
    assert created._vision_dp_group is group
    assert calls == [("vision_encoder", True), ("vision_projector", True)]


class _EchoEncoder:
    def __init__(self):
        self.config = SimpleNamespace(
            spatial_merge_size=2,
            in_channels=1,
            temporal_patch_size=1,
            patch_size=1,
            out_hidden_size=2,
        )
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self.calls = 0

    def __call__(self, pixels, grids):
        self.calls += 1
        return pixels.expand(-1, 2).contiguous()


def _gloo_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )
    try:
        group = SimpleNamespace(
            world_size=world,
            rank_in_group=rank,
            cpu_group=dist.group.WORLD,
            all_gather_into_tensor=lambda output, value: dist.all_gather_into_tensor(
                output, value
            ),
        )
        for grids in (
            [[1, 2, 2]],
            [[1, 2, 2], [2, 2, 6]],
            [[1, 2, 4], [1, 4, 4], [2, 2, 2], [1, 2, 6], [1, 2, 2]],
        ):
            items = _items(grids, width=1, dtype=torch.float32)
            for index, item in enumerate(items):
                item.feature.copy_(
                    torch.arange(item.feature.numel()).reshape(-1, 1) + index * 100
                )
            # Every rank must enter coordination, including a fully warm cache.
            cached = [not (rank == 2 and index == 0) for index in range(len(items))]
            assert DP.synchronize_vision_dp_cache(items, cached, group=group) == [
                True
            ] + [False] * (len(items) - 1)
            encoder = _EchoEncoder()
            projector = lambda hidden: hidden.reshape(-1, 4, 2).sum(1)
            results = DP.encode_vision_dp(items, encoder, projector, group)
            plan = DP.VisionDPPlan(tuple(map(tuple, grids)), 2, world, rank)
            assert encoder.calls == bool(plan.assignments[rank])
            for item, output in zip(items, results):
                expected = projector(item.feature.expand(-1, 2))
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert DP.synchronize_vision_dp_cache([], [], group=group) == []
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_real_four_process_dp_handles_idle_ranks_padding_and_order(tmp_path):
    mp.spawn(
        _gloo_worker, args=(4, (tmp_path / "dp-store").as_uri()), nprocs=4, join=True
    )
