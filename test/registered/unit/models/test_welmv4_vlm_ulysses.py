"""Run actual WeLM encoder/block forwards with CPU collectives and attention.

Only accelerator/runtime dependencies are substituted. Production Ulysses
packing, RoPE, residuals, micro-batching and gather-before-projector all run.
"""

import ast
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from test_welmv4_vlm_fusions import HELPERS
from test_welmv4_vlm_ulysses_layout import ROOT, ULYSSES, Plan
from test_welmv4_vlm_vision import OPS, _load_definitions, register_cpu_ci
from torch import nn

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _ascend_backend_name():
    # Read the real registry: do not let a made-up backend name in a mock
    # silently approve a startup guard which would reject the real runtime.
    tree = ast.parse(
        (ROOT / "python/sglang/srt/layers/attention/vision.py").read_text(
            encoding="utf-8"
        )
    )
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "QKV_BACKEND_IMPL" for t in node.targets
        )
    )
    return next(
        key.value
        for key, value in zip(registry.keys, registry.values)
        if isinstance(value, ast.Name) and value.id == "VisionAscendAttention"
    )


class _Collectives:
    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=20)
        self.inputs, self.outputs, self.kinds = {}, {}, {}
        self.calls = []

    def call(self, rank, kind, output, input):
        self.inputs[rank] = input.clone()
        self.outputs[rank] = output
        self.kinds[rank] = kind
        self.barrier.wait()
        if rank == 0:
            assert len(set(self.kinds.values())) == 1
            self.calls.append(kind)
            for dst in range(self.size):
                if kind == "a2a":
                    received = torch.stack(
                        [self.inputs[src][dst] for src in range(self.size)]
                    )
                else:
                    received = torch.cat([self.inputs[src] for src in range(self.size)])
                self.outputs[dst].copy_(received)
        self.barrier.wait()
        # Prevent next call from replacing slots before all ranks finish.
        self.barrier.wait()

    def group(self, rank):
        return SimpleNamespace(
            world_size=self.size,
            rank_in_group=rank,
            all_to_all_single=lambda out, inp: self.call(rank, "a2a", out, inp),
            all_gather_into_tensor=lambda out, inp: self.call(rank, "ag", out, inp),
        )


class _Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, **kwargs):
        super().__init__()
        self.tp_size = kwargs.get("tp_size", 1)
        self.tp_rank = kwargs.get("tp_rank", 0)
        self.constructor_kwargs = kwargs
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.03)
        self.bias = nn.Parameter(torch.randn(output_size) * 0.02) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


class _AttentionBackend:
    def forward(self, q, k, v, cu_seqlens, **kwargs):
        assert q.shape[0] == int(cu_seqlens[-1])
        result = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            qs, ks, vs = [
                x[int(start) : int(end)].transpose(0, 1).float() for x in (q, k, v)
            ]
            # Explicit segmented FP32 reference, not an emulation of packing.
            scores = qs @ ks.transpose(-1, -2) * qs.shape[-1] ** -0.5
            result.append((scores.softmax(-1) @ vs).transpose(0, 1).to(q.dtype))
        return torch.cat(result)


class _AttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, use_data_parallel, **kwargs):
        super().__init__()
        self.head_size = embed_dim // num_heads
        self.q_size = self.kv_size = embed_dim
        self.softmax_scale = None
        self.qkv_backend_name = _ascend_backend_name()
        self.qkv_backend = _AttentionBackend()
        self.num_heads = num_heads
        self.replicated = use_data_parallel
        self.qkv_proj = _Linear(embed_dim, embed_dim * 3, tp_size=1, tp_rank=0)
        self.proj = _Linear(embed_dim, embed_dim, tp_size=1, tp_rank=0)

    def forward(self, x, cu_seqlens, position_embeddings):
        qkv = self.qkv_proj(x.squeeze(0))[0]
        q, k, v = [
            x.reshape(-1, self.num_heads, self.head_size) for x in qkv.chunk(3, -1)
        ]
        q, k = OPS.welmv4_vision_apply_rope(q, k, *position_embeddings)
        y = self.qkv_backend.forward(q, k, v, cu_seqlens)
        return self.proj(y.flatten(1))[0].unsqueeze(0)


def _types(group, cap):
    return _load_definitions(
        "_vision_dp_tp_kwargs",
        "_patch_embed_forward_matmul",
        "WeLMV4VisionPatchEmbed",
        "WeLMV4VisionPositionEmbedding",
        "WeLMV4VisionAttention",
        "WeLMV4VisionMLP",
        "WeLMV4VisionBlock",
        "WeLMV4VisionEncoder",
        "WeLMV4VisionProjector",
        VisionAttention=_AttentionBase,
        ColumnParallelLinear=_Linear,
        RowParallelLinear=_Linear,
        VisionNPUFusionConfig=HELPERS.VisionNPUFusionConfig,
        vision_column_parallel_linear_gelu=HELPERS.vision_column_parallel_linear_gelu,
        vision_add_layer_norm=HELPERS.vision_add_layer_norm,
        welmv4_vision_quick_gelu=OPS.welmv4_vision_quick_gelu,
        welmv4_vision_apply_rope=OPS.welmv4_vision_apply_rope,
        _apply_welmv4_vision_rope=lambda *args: None,
        create_vision_ulysses_group=lambda *args, **kwargs: group,
        validate_vision_ulysses_input=lambda x: None,
        VisionUlyssesPlan=Plan,
        _vision_max_patches=lambda: cap,
        add_prefix=lambda name, prefix: name,
        print_warning_once=lambda message: None,
    )


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("cap", [0, 12])
@pytest.mark.parametrize("grids", [[[1, 2, 2]], [[1, 2, 6], [2, 2, 4]]])
def test_full_encoder_and_projector_match_reference(world, cap, grids):
    torch.set_num_threads(1)
    torch.manual_seed(610)
    # Real BF16 RoPE, 72-wide heads; small model but two real residual blocks.
    config = SimpleNamespace(
        hidden_size=8 * 72,
        num_attention_heads=8,
        intermediate_size=96,
        spatial_merge_size=2,
        num_position_embeddings=16,
        num_hidden_layers=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=1,
        hidden_act="gelu_pytorch_tanh",
        out_hidden_size=32,
    )
    reference_types = _types(None, cap)
    reference = reference_types.WeLMV4VisionEncoder(config).bfloat16().eval()
    projector = reference_types.WeLMV4VisionProjector(config).bfloat16().eval()
    with torch.no_grad():
        reference.pos_embed.weight.normal_(std=0.1)
    n = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(n, 12).bfloat16()
    grid = torch.tensor(grids)
    collectives = _Collectives(world)
    models = []
    for rank in range(world):
        types = _types(collectives.group(rank), cap)
        encoder = types.WeLMV4VisionEncoder(config).bfloat16().eval()
        encoder.load_state_dict(reference.state_dict(), strict=True)
        for block in encoder.blocks:
            assert block.attn.replicated
            for linear in (
                block.attn.qkv_proj,
                block.attn.proj,
                block.mlp.fc1,
                block.mlp.fc2,
            ):
                assert linear.tp_size == 1 and linear.tp_rank == 0
        models.append(encoder)

    @torch.inference_mode()
    def run(rank):
        return models[rank](pixels, grid)

    with torch.inference_mode():
        expected = reference(pixels, grid)
        expected_image = projector(expected)
        with ThreadPoolExecutor(max_workers=world) as pool:
            actual = list(pool.map(run, range(world)))
        for hidden in actual:
            # BF16 matmul may pick a different kernel for shorter M.
            torch.testing.assert_close(hidden, expected, atol=0.03125, rtol=0.02)
            torch.testing.assert_close(
                projector(hidden), expected_image, atol=0.015625, rtol=0.02
            )
        assert len(actual[0]) == n
        if world == 1:
            assert not collectives.calls
        else:
            groups = 1 if cap == 0 else 3
            if n == 4:
                groups = 1
            assert collectives.calls == (["a2a"] * 4 + ["ag"]) * groups


def _runtime(
    monkeypatch,
    *,
    npu=True,
    world=4,
    graph=False,
    dp=False,
    mismatched=False,
    attn_ranks=None,
):
    group = SimpleNamespace(
        world_size=world, rank_in_group=0, ranks=list(range(world)), cpu_group=object()
    )
    attention_group = (
        group
        if attn_ranks is None
        else SimpleNamespace(
            world_size=len(attn_ranks),
            rank_in_group=0,
            ranks=attn_ranks,
            cpu_group=object(),
        )
    )
    import runpy

    policy = ModuleType("sglang.srt.layers.welmv4_vision_parallel")
    policy.__dict__.update(
        runpy.run_path(str(ROOT / "python/sglang/srt/layers/welmv4_vision_parallel.py"))
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.welmv4_vision_parallel", policy)
    monkeypatch.delenv("SGLANG_VLM_VISION_USE_DP", raising=False)
    modules = {
        "sglang.srt.utils": {"is_npu": lambda: npu},
        "sglang.srt.distributed": {
            "get_tp_group": lambda: group,
            "get_attn_tp_group": lambda: attention_group,
        },
        "sglang.srt.environ": {
            "envs": SimpleNamespace(
                SGLANG_VIT_ENABLE_CUDA_GRAPH=SimpleNamespace(get=lambda: graph)
            )
        },
        "sglang.srt.runtime_context": {
            "get_mm": lambda: SimpleNamespace(mm_enable_dp_encoder=dp)
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def gather(output, value, **kwargs):
        output[:] = [value] * len(output)
        if mismatched:
            output[-1] = ("different",)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    return group


@pytest.mark.parametrize("value", ["0", "false", "1", "true", "bad"])
def test_startup_switch_and_group(monkeypatch, value):
    group = _runtime(monkeypatch)
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_ULYSSES", value)
    config = SimpleNamespace(
        num_attention_heads=16, hidden_size=1152, num_hidden_layers=27
    )
    if value == "bad":
        with pytest.raises(ValueError, match="must be"):
            ULYSSES.create_vision_ulysses_group(config)
    else:
        actual = ULYSSES.create_vision_ulysses_group(config)
        assert actual is (group if value in {"1", "true"} else None)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"npu": False}, "NPU"),
        ({"graph": True}, "graphs"),
        ({"dp": True}, "mutually exclusive"),
        ({"mismatched": True}, "differ"),
        ({"attn_ranks": [0, 1]}, "requires --mm-enable-dp-encoder"),
        ({"world": 3}, "divisible"),
    ],
)
def test_startup_rejects_unsupported_or_inconsistent_modes(
    monkeypatch, options, message
):
    _runtime(monkeypatch, **options)
    monkeypatch.setenv("SGLANG_VLM_VISION_USE_ULYSSES", "1")
    config = SimpleNamespace(
        num_attention_heads=16, hidden_size=1152, num_hidden_layers=27
    )
    with pytest.raises(ValueError, match=message):
        ULYSSES.create_vision_ulysses_group(config)


def test_runtime_rejects_non_npu_activations():
    with pytest.raises(ValueError, match="BF16 NPU"):
        ULYSSES.validate_vision_ulysses_input(torch.ones(3, 4).bfloat16())


def test_real_attention_constructor_replicates_qkv_and_projection_on_nonzero_rank():
    tree = ast.parse(
        (ROOT / "python/sglang/srt/layers/attention/vision.py").read_text(
            encoding="utf-8"
        )
    )
    base = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionAttention"
    )
    # Run the production constructor and backend selection. Substitute only
    # allocating linears and the accelerator backend, not parallel decisions.
    base.body = [
        node
        for node in base.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"__init__", "_determine_attention_backend"}
    ]

    class QKV(_Linear):
        def __init__(
            self, hidden_size, head_size, total_num_heads, total_num_kv_heads, **kwargs
        ):
            width = (total_num_heads + 2 * total_num_kv_heads) * head_size
            super().__init__(hidden_size, width // kwargs["tp_size"], **kwargs)

    scope = dict(
        torch=torch,
        nn=nn,
        dist_utils=SimpleNamespace(divide=lambda a, b: a // b),
        get_parallel=lambda: SimpleNamespace(attn_tp_size=4, attn_tp_rank=3),
        get_mm=lambda: SimpleNamespace(mm_attention_backend=_ascend_backend_name()),
        print_info_once=lambda text: None,
        QKV_BACKEND_IMPL={_ascend_backend_name(): lambda **kwargs: _AttentionBackend()},
        QKVParallelLinear=QKV,
        RowParallelLinear=_Linear,
        add_prefix=lambda name, prefix: name,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    compiled = ast.fix_missing_locations(
        ast.Module(body=[future, base], type_ignores=[])
    )
    exec(compile(compiled, "vision.py", "exec"), scope)
    types = _load_definitions(
        "WeLMV4VisionAttention",
        VisionAttention=scope["VisionAttention"],
        RowParallelLinear=_Linear,
        _apply_welmv4_vision_rope=lambda *args: None,
    )
    config = SimpleNamespace(hidden_size=1152, num_attention_heads=16)
    sp = types.WeLMV4VisionAttention(
        config, ulysses_group=SimpleNamespace(world_size=4, rank_in_group=3)
    )
    tp = types.WeLMV4VisionAttention(config)
    assert (sp.q_size, sp.kv_size) == (1152, 1152)
    assert (tp.q_size, tp.kv_size) == (288, 288)
    assert sp.qkv_proj.weight.shape == (3456, 1152)
    assert tp.qkv_proj.weight.shape == (864, 1152)
    for linear in (sp.qkv_proj, sp.proj):
        assert (linear.tp_size, linear.tp_rank) == (1, 0)
    assert (tp.qkv_proj.tp_size, tp.qkv_proj.tp_rank) == (4, 3)


def _gloo_worker(rank, world, rendezvous):
    torch.set_num_threads(1)
    from datetime import timedelta

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
            all_to_all_single=lambda out, inp: dist.all_to_all_single(out, inp),
            all_gather_into_tensor=lambda out, inp: dist.all_gather_into_tensor(
                out, inp
            ),
        )
        for n in (1, 11, 28):
            plan = Plan(n, 8, world, rank)
            full = torch.arange(n * 8 * 6).reshape(n, 8, 6).bfloat16()
            local = plan.local_slice(full)
            heads = plan.sequence_to_heads(local, local + 2, local + 4, group)
            for i, tensor in enumerate(heads):
                expected = (full + i * 2)[:, rank * 2 : (rank + 1) * 2]
                torch.testing.assert_close(tensor, expected, atol=0, rtol=0)
            restored = plan.heads_to_sequence(heads[0], group)
            torch.testing.assert_close(restored, local, atol=0, rtol=0)
            torch.testing.assert_close(
                plan.gather_sequence(restored, group), full, atol=0, rtol=0
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable"
)
def test_real_four_process_collective_contract(tmp_path):
    # FileStore also works on Windows without a libuv TCPStore dependency.
    mp.spawn(
        _gloo_worker,
        args=(4, (tmp_path / "ulysses-store").as_uri()),
        nprocs=4,
        join=True,
    )
