"""Independent ACLNN precision checks for the BF16 and MXFP8 fused QKV kernels.

Run on a supported Ascend device with cannbot-dsl 0.8.0:
    python -m pytest test/manual/ascend/test_welmv4_fused_qkv_precision.py -q -s

Includes ordinary dynamic-M AOT calls and the network's rank-chunk ABI.
Rank chunks are emulated on one NPU: this validates input/output row mapping,
not distributed collective scheduling or full-model graph capture.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend NPU is required"
)
ROOT = Path(__file__).resolve().parents[3]
LAYERS = ROOT / "python/sglang/srt/layers"
SLOTS = 24576
MAX_POS = 8192 + 16500
HEAD = 256
Q_WIDTH = 1536
SENTINEL = -7.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LAYERS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["bf16", "quant"])
def operator(request):
    quant = request.param == "quant"
    module = _load(
        "fused_quant_qkv_proj_norm_rope_cache"
        if quant
        else "fused_qkv_proj_norm_rope_cache"
    )
    return SimpleNamespace(
        quant=quant,
        compile=getattr(module, "compile_aot_quant" if quant else "compile_aot"),
        compile_rank=getattr(
            module,
            "compile_aot_quant_rank_chunk" if quant else "compile_aot_rank_chunk",
        ),
    )


def _resident(operator, width):
    gen = torch.Generator().manual_seed(41)
    weight = (torch.randn(width, 2048, generator=gen) * 0.02).bfloat16().npu()
    gamma = torch.randn(1, HEAD, generator=gen).bfloat16().npu()
    angle = torch.arange(MAX_POS).float()[:, None] / (
        10000.0 ** (torch.arange(32).float()[None, :] / 32)
    )
    cs = torch.cat([angle.cos(), angle.sin()], dim=1).npu()
    if operator.quant:
        weight, scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight, dst_type=torch.float8_e4m3fn
        )
        # Recreate the network's stored transpose views, then undo them
        # exactly as WelmV4Attention does before invoking the fused ABI.
        stored_w = weight.transpose(0, 1)
        stored_s = scale.view(width, 32, 2).transpose(0, 1)
        weights = (
            stored_w.transpose(0, 1).view(torch.int8),
            stored_s.transpose(0, 1).view(torch.int8),
        )
    else:
        weights = (weight,)
    return weights, gamma, cs


def _batch(operator, weights, m, contiguous, seed):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(m, 2048, generator=gen).bfloat16().npu()
    if contiguous:
        positions = torch.arange(7, m + 7, dtype=torch.int64)
    else:
        positions = torch.randint(0, MAX_POS, (m,), generator=gen, dtype=torch.int64)
    slots = torch.randperm(SLOTS, generator=gen)[:m].to(torch.int64)
    # Padding must compute dense outputs but must never scatter to the cache.
    slots[-1] = -1
    if operator.quant:
        x, scale = torch.ops.npu.npu_dynamic_mx_quant(
            hidden, dst_type=torch.float8_e4m3fn
        )
        w = weights[0].view(torch.float8_e4m3fn).transpose(0, 1)
        sw = weights[1].view(torch.float8_e8m0fnu).transpose(0, 1)
        projection = torch.ops.npu.npu_quant_matmul(
            x,
            w,
            sw,
            scale_dtype=torch.float8_e8m0fnu,
            pertoken_scale=scale,
            pertoken_scale_dtype=torch.float8_e8m0fnu,
            output_dtype=torch.bfloat16,
            group_sizes=[1, 1, 32],
        )
        inputs = (x.view(torch.int8), scale.view(torch.int8))
    else:
        projection = hidden @ weights[0].transpose(0, 1)
        inputs = (hidden,)
    torch.npu.synchronize()
    return inputs, positions.npu(), slots.npu(), projection.cpu()


def _golden(projection, gamma, positions, cs, width):
    q = projection[:, :Q_WIDTH].clone()
    k = projection[:, Q_WIDTH : Q_WIDTH + HEAD].float()
    k = (
        k
        * torch.rsqrt((k * k).sum(-1, keepdim=True) / HEAD + 1e-6)
        * gamma.cpu().float()
    ).bfloat16()
    v = projection[:, Q_WIDTH + HEAD : Q_WIDTH + 2 * HEAD].clone()
    table = cs.cpu()[positions.cpu()]
    for value, heads in [(q, 6), (k, 1)]:
        shaped = value.view(-1, heads, HEAD)
        first = shaped[..., 192:224].float().clone()
        second = shaped[..., 224:256].float().clone()
        cos = table[:, None, :32]
        sin = table[:, None, 32:]
        shaped[..., 192:224] = (first * cos - second * sin).bfloat16()
        shaped[..., 224:256] = (first * sin + second * cos).bfloat16()
    outputs = [q, k, v]
    if width == 2560:
        outputs += [projection[:, 2048:2304].clone(), projection[:, 2304:2560].clone()]
    return outputs


def _assert_close(name, output, golden):
    actual = output.detach().float().cpu().numpy()
    expected = golden.float().cpu().numpy()
    if actual.size == 0:
        return 0.0
    mismatch = float((~np.isclose(actual, expected, rtol=1e-3, atol=1e-3)).mean())
    assert mismatch <= 1e-3, (
        f"{name}: mismatch={mismatch:.6%}, max_error={np.abs(actual - expected).max()}"
    )
    return mismatch


def _outputs(m, width):
    shapes = [(m, Q_WIDTH), (m, HEAD), (m, HEAD)]
    shapes += [(m if width == 2560 else 1, HEAD)] * 2
    return [
        torch.full(shape, SENTINEL, dtype=torch.bfloat16, device="npu")
        for shape in shapes
    ]


def _caches():
    # The serving pool exposes 4-D pages, flattened only at the ABI boundary.
    return [
        torch.full(
            (SLOTS // 64, 64, 1, HEAD), SENTINEL, dtype=torch.bfloat16, device="npu"
        ).view(-1, HEAD)
        for _ in range(2)
    ]


def _check_cache(caches, outputs, golden, slots, rows, expected_caches, before):
    slot_cpu = slots.cpu()
    valid_rows = rows[slot_cpu[rows] != -1]
    valid_slots = slot_cpu[valid_rows]
    for name, cache, out, ref, expected, previous in zip(
        ["k", "v"], caches, outputs[1:3], golden[1:3], expected_caches, before
    ):
        expected[valid_slots] = ref[valid_rows]
        _assert_close(
            f"{name}_touched_cache", cache[valid_slots.npu()], ref[valid_rows]
        )
        assert torch.equal(cache[valid_slots.npu()].cpu(), out[valid_rows.npu()].cpu())
        # Check untouched and previously written rows across successive calls.
        _assert_close(f"{name}_whole_cache", cache, expected)
        untouched = torch.ones(SLOTS, dtype=torch.bool)
        untouched[valid_slots] = False
        assert torch.equal(cache.cpu()[untouched], previous[untouched])


@pytest.mark.parametrize("width", [2048, 2560], ids=["plain", "mirror"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["prefill", "decode"])
def test_dynamic_precision(operator, width, contiguous):
    weights, gamma, cs = _resident(operator, width)
    fn = operator.compile(
        width, SLOTS, MAX_POS, return_v=True, positions_contiguous=contiguous
    )
    caches = _caches()
    expected_caches = [cache.cpu() for cache in caches]
    batches = [33, 257, 16500] if contiguous else [1, 4, 56, 200]
    for m in batches:
        inputs, positions, slots, projection = _batch(
            operator, weights, m, contiguous, m + width
        )
        golden = _golden(projection, gamma, positions, cs, width)
        outputs = _outputs(m, width)
        before = [cache.cpu() for cache in caches]
        fn(*inputs, *weights, gamma, positions, cs, slots, *outputs, *caches, 1e-6)
        torch.npu.synchronize()
        stats = [
            _assert_close(name, out, ref)
            for name, out, ref in zip(
                ["q", "k", "v", "mirror_k", "mirror_v"], outputs, golden
            )
        ]
        _check_cache(
            caches, outputs, golden, slots, torch.arange(m), expected_caches, before
        )
        if width == 2048:
            assert all(bool((out == SENTINEL).all()) for out in outputs[3:])
        print(f"dynamic quant={operator.quant} width={width} M={m}: mismatch={stats}")


@pytest.mark.parametrize("width", [2048, 2560], ids=["plain", "mirror"])
@pytest.mark.parametrize("contiguous", [True, False], ids=["prefill", "decode"])
def test_rank_chunk_precision(operator, width, contiguous):
    weights, gamma, cs = _resident(operator, width)
    fn = operator.compile_rank(width, SLOTS, MAX_POS, positions_contiguous=contiguous)
    for local_rows, chunks in [
        (289, [(0, 1), (1, 256), (256, 289)]),
        (513, [(0, 256), (256, 513)]),
    ]:
        ranks = 4
        m = ranks * local_rows
        inputs, positions, slots, projection = _batch(
            operator, weights, m, contiguous, local_rows
        )
        golden = _golden(projection, gamma, positions, cs, width)
        outputs = _outputs(m, width)
        caches = _caches()
        expected_caches = [cache.cpu() for cache in caches]
        written = torch.zeros(m, dtype=torch.bool)
        for start, end in chunks:
            # One packed chunk contains this interval from every rank, as
            # produced by all_gather_into_tensor in the network.
            rows = torch.cat(
                [
                    torch.arange(r * local_rows + start, r * local_rows + end)
                    for r in range(ranks)
                ]
            )
            packed = [value[rows.npu()].contiguous() for value in inputs]
            before = [cache.cpu() for cache in caches]
            fn(
                *packed,
                *weights,
                gamma,
                positions,
                cs,
                slots,
                *outputs,
                *caches,
                1e-6,
                local_rows,
                end - start,
                start,
            )
            torch.npu.synchronize()
            written[rows] = True
            for name, out, ref in zip(
                ["q", "k", "v", "mirror_k", "mirror_v"], outputs, golden
            ):
                _assert_close(name, out[written.npu()], ref[written])
                assert bool((out[(~written).npu()] == SENTINEL).all()), name
            _check_cache(caches, outputs, golden, slots, rows, expected_caches, before)
            if width == 2048:
                assert all(bool((out == SENTINEL).all()) for out in outputs[3:])
        assert bool(written.all())
        print(
            f"rank-chunk quant={operator.quant} width={width} local_M={local_rows}: all clean"
        )


def test_network_quant_factory():
    """Run the actual factory with the renamed module, without server imports."""
    import sys

    module = _load("fused_quant_qkv_proj_norm_rope_cache")
    name = "sglang.srt.layers.fused_quant_qkv_proj_norm_rope_cache"
    tree = ast.parse((ROOT / "python/sglang/srt/models/welmv4.py").read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_welm_fused_quant_qkv_program"
    )
    for arg in fn.args.args + fn.args.kwonlyargs:
        arg.annotation = None
    fn.returns = None
    calls = []
    for attr in ["compile_aot_quant", "compile_aot_quant_rank_chunk"]:
        setattr(
            module,
            attr,
            lambda *args, _entry=attr, **kwargs: (
                calls.append((_entry, args, kwargs)) or object()
            ),
        )
    namespace = {"_WELMV4_FUSED_QUANT_QKV_PROGRAMS": {}}
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, name, module)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                "<network-quant-factory>",
                "exec",
            ),
            namespace,
        )
        factory = namespace[fn.name]
        for rank in [False, True]:
            first = factory(
                "npu:0", 2560, SLOTS, MAX_POS, False, rank_chunk_layout=rank
            )
            assert first is factory(
                "npu:0", 2560, SLOTS, MAX_POS, False, rank_chunk_layout=rank
            )
    assert [entry for entry, _, _ in calls] == [
        "compile_aot_quant",
        "compile_aot_quant_rank_chunk",
    ]
    assert calls[0][2]["return_v"] is True


def test_repeated_prefill_slot_reuse(operator):
    """Catch stale cos/sin or UB rows that element-rate tolerances can hide."""
    width, m = 2560, 2048  # 80 items > 32 cores: slots are reused per launch.
    weights, gamma, cs = _resident(operator, width)
    inputs, positions, slots, projection = _batch(operator, weights, m, True, 91)
    golden = _golden(projection, gamma, positions, cs, width)
    fn = operator.compile(
        width, SLOTS, MAX_POS, return_v=True, positions_contiguous=True
    )
    caches = _caches()
    valid = slots != -1
    expected_rope = [
        golden[0].view(m, 6, HEAD)[..., 192:].float(),
        golden[1][:, 192:].float(),
    ]
    for launch in range(20):
        outputs = _outputs(m, width)
        fn(*inputs, *weights, gamma, positions, cs, slots, *outputs, *caches, 1e-6)
        torch.npu.synchronize()
        actual_rope = [
            outputs[0].cpu().view(m, 6, HEAD)[..., 192:].float(),
            outputs[1].cpu()[:, 192:].float(),
        ]
        for name, actual, expected in zip(["q", "k"], actual_rope, expected_rope):
            bad_rows = ((actual - expected).abs() > 0.05).any(-1).sum().item()
            assert bad_rows == 0, (
                f"{name}: launch {launch} has {bad_rows} stale RoPE rows"
            )
        for cache, output in zip(caches, outputs[1:3]):
            assert torch.equal(cache[slots[valid]].cpu(), output[valid].cpu())
    print(f"slot reuse quant={operator.quant}: 20 launches clean")
