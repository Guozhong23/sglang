"""CPU contracts for the merged WeLM MegaMoE sidecar and MXFP8 layout.

Execute the real implementation definitions with vendor imports substituted;
these tests verify admission, buffer arguments and tensor layout without an NPU.
"""

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Parameter

ROOT = Path(__file__).resolve().parents[4]
SIDECAR = ROOT / "python/sglang/srt/hardware_backend/npu/moe/welmv4_megamoe.py"
METHODS = ROOT / "python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py"


def _env(**overrides):
    values = dict(
        SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK=2048,
        WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=1024,
        SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS="all",
        SGLANG_NPU_MEGAMOE_SYNC_AFTER_OP=False,
    )
    values.update(overrides)
    return SimpleNamespace(**{
        name: SimpleNamespace(get=lambda value=value: value)
        for name, value in values.items()
    })


def _sidecar(env=None):
    tree = ast.parse(SIDECAR.read_text())
    # This module's only serving dependency is its environment descriptors.
    tree.body = [
        node for node in tree.body
        if not isinstance(node, ast.ImportFrom) or node.module != "sglang.srt.environ"
    ]
    scope = {"envs": env or _env()}
    exec(compile(tree, str(SIDECAR), "exec"), scope)
    return SimpleNamespace(**scope)


def _runtime(monkeypatch, mode="bf16", env=None):
    vendor = SimpleNamespace(
        get_symm_buffer_for_mega_moe=Mock(return_value=Mock()),
        mega_moe=Mock(),
    )
    monkeypatch.setitem(sys.modules, "npu_ops_transformer.ops.mega_moe", vendor)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    group = SimpleNamespace(_get_backend=lambda device: Mock())
    config = SimpleNamespace(
        num_experts=512, top_k=10, hidden_size=2048,
        intermediate_size_per_partition=512,
    )
    module = _sidecar(env)
    runtime = module.WelmPrefillMegaMoE(
        group, config=config, device=torch.device("cpu"), weight_mode=mode,
    )
    return module, runtime, vendor


@pytest.mark.parametrize("mode", ["bf16", "mxfp8"])
def test_buffer_combines_custom_capacity_and_upstream_quant_contract(monkeypatch, mode):
    _, runtime, vendor = _runtime(monkeypatch, mode)
    args = vendor.get_symm_buffer_for_mega_moe.call_args.kwargs
    assert args["num_max_tokens_per_rank"] == 2048
    assert args["dispatch_quant_mode"] == (4 if mode == "mxfp8" else 0)
    assert args["intermediate_hidden"] == (0 if mode == "mxfp8" else 512)
    assert args.get("dispatch_quant_out_dtype") == (24 if mode == "mxfp8" else None)
    assert runtime.max_local_rows == 2048


def test_threshold_uses_strict_global_row_boundary_and_keeps_layer_admission(monkeypatch):
    _, runtime, _ = _runtime(
        monkeypatch, env=_env(SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS="0-3,7")
    )
    assert not runtime.meets_prefill_threshold(1024)
    assert runtime.meets_prefill_threshold(1025)
    # Threshold is a global OProj-row decision; local shard capacity is separate.
    assert runtime.meets_prefill_threshold(512 * 4)
    assert runtime.can_run(512, 3)
    assert not runtime.can_run(512, 4)
    assert runtime.can_run(512, 7)
    assert not runtime.can_run(2049, 3)
    assert not runtime.can_run(0, 3)


def test_default_capacity_and_disabled_threshold(monkeypatch):
    _, runtime, _ = _runtime(monkeypatch, env=_env(
        SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK=0,
        WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=0,
    ))
    assert runtime.max_local_rows == 16384
    assert runtime.meets_prefill_threshold(1)


@pytest.mark.parametrize("selection", ["2-1", "-1", "1--2", "x"])
def test_invalid_layer_selection_fails_explicitly(monkeypatch, selection):
    _, runtime, _ = _runtime(
        monkeypatch, env=_env(SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS=selection)
    )
    with pytest.raises(RuntimeError, match="Invalid SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS"):
        runtime.can_run(1, 0)


def _experts(dtype=torch.bfloat16, canonical=True, processed=True):
    return SimpleNamespace(
        w13_weight=torch.zeros(2, 128, 64, dtype=dtype),
        w2_weight=torch.zeros(2, 64, 64, dtype=dtype),
        w13_weight_scale=torch.full((2, 128, 2), 127, dtype=torch.uint8),
        w2_weight_scale=torch.full((2, 64, 2), 127, dtype=torch.uint8),
        w13_kernel=SimpleNamespace(use_megamoe_canonical_layout=canonical),
        w2_kernel=SimpleNamespace(use_megamoe_canonical_layout=canonical),
        _npu_megamoe_weights_processed=processed,
    )


def test_mxfp8_admission_requires_paired_processing_and_both_canonical_markers():
    module = _sidecar()
    assert module._weight_mode(_experts()) == "bf16"
    experts = _experts(torch.float8_e4m3fn)
    assert module._weight_mode(experts) == "mxfp8"
    experts.w2_kernel.use_megamoe_canonical_layout = False
    assert module._weight_mode(experts) is None
    experts.w2_kernel.use_megamoe_canonical_layout = True
    experts._npu_megamoe_weights_processed = False
    assert module._weight_mode(experts) is None
    experts._npu_megamoe_weights_processed = True
    experts.w13_weight_scale = None
    assert module._weight_mode(experts) is None


def test_physical_nd_check_includes_mxfp8_scales(monkeypatch):
    module = _sidecar()
    experts = _experts(torch.float8_e4m3fn)
    get_format = Mock(return_value=2)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(get_npu_format=get_format))
    module._validate_nd_tensors(experts, "mxfp8")
    assert get_format.call_count == 4
    get_format.side_effect = lambda tensor: 29 if tensor is experts.w2_weight_scale else 2
    with pytest.raises(RuntimeError, match="w2_weight_scale.*29"):
        module._validate_nd_tensors(experts, "mxfp8")


@pytest.mark.parametrize("zero_padding", [False, True])
def test_forward_zeros_padding_without_modifying_fp32_router_weights(monkeypatch, zero_padding):
    _, runtime, vendor = _runtime(monkeypatch)
    hidden = torch.zeros(3, 4, dtype=torch.bfloat16)
    routing = SimpleNamespace(
        topk_ids=torch.tensor([[0, 1], [1, 0], [0, 1]]),
        topk_weights=torch.full((3, 2), 0.5),
    )
    vendor.mega_moe.return_value = (torch.ones_like(hidden), None)
    output = runtime.forward_layer(
        _experts(), hidden, routing, 2, zero_output_padding=zero_padding
    )
    passed_weights = vendor.mega_moe.call_args.args[2]
    assert passed_weights.dtype == torch.bfloat16
    assert not passed_weights[-1].any()
    assert routing.topk_weights[-1].tolist() == [0.5, 0.5]
    assert output[-1].tolist() == ([0.0] * 4 if zero_padding else [1.0] * 4)


def _mxfp8_type(monkeypatch):
    class Base:
        @staticmethod
        def _set_dispatcher_output_dtype(layer, dtype):
            layer.dispatcher_dtype = dtype

    backend = SimpleNamespace(is_megamoe=lambda: False, is_deepep=lambda: True)
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.moe", SimpleNamespace(
        get_moe_a2a_backend=lambda: backend
    ))
    tree = ast.parse(METHODS.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "NPUMXFP8MoEMethod")
    methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
    assert len(methods) == len(set(methods)), "merge must not leave duplicate methods"
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        cls,
    ], type_ignores=[])
    scope = dict(
        torch=torch, Parameter=Parameter, _NPUMoEMethodBase=Base,
        _get_float8_e8m0fnu_dtype=lambda: torch.float8_e8m0fnu,
        _require_e8m0_dtype=lambda: torch.float8_e8m0fnu,
        logger=logging.getLogger(__name__),
    )
    exec(compile(ast.fix_missing_locations(module), str(METHODS), "exec"), scope)
    return scope["NPUMXFP8MoEMethod"]


def _mxfp8_layer(monkeypatch):
    cls = _mxfp8_type(monkeypatch)
    layer = _experts(torch.float8_e4m3fn, canonical=False, processed=False)
    layer.num_local_experts = 2
    layer.intermediate_size_per_partition = 64
    layer.hidden_size = 64
    layer.welm_megamoe_keep_nd = True
    for prefix in ("w13", "w2"):
        kernel = cls.__new__(cls)
        kernel.use_megamoe_canonical_layout = False
        setattr(layer, f"{prefix}_kernel", kernel)
    return cls, layer


def test_paired_mxfp8_processing_preserves_e8m0_bits_and_is_idempotent(monkeypatch):
    cls, layer = _mxfp8_layer(monkeypatch)
    assert cls.maybe_process_megamoe_weights(layer)
    assert layer.dispatcher_dtype == "bf16"
    assert layer._npu_megamoe_weights_processed
    assert _sidecar()._weight_mode(layer) == "mxfp8"
    expected_shapes = {"w13": (2, 128, 1, 2), "w2": (2, 64, 1, 2)}
    for prefix in ("w13", "w2"):
        scale = getattr(layer, f"{prefix}_weight_scale")
        assert scale.shape == expected_shapes[prefix]
        assert scale.dtype == torch.float8_e8m0fnu
        assert torch.all(scale.view(torch.uint8) == 127)
    original = layer.w13_weight
    layer.w13_kernel.process_weights_after_loading(layer, "w13")
    layer.w2_kernel.process_weights_after_loading(layer, "w2")
    assert layer.w13_weight is original


def test_scale_view_matches_fallback_without_copying_and_rejects_missing_scale(monkeypatch):
    cls, layer = _mxfp8_layer(monkeypatch)
    cls.maybe_process_megamoe_weights(layer)
    scale = layer.w13_weight_scale
    view = layer.w13_kernel._weight_scale_for_gmm(scale)
    assert view.shape == (2, 1, 128, 2)
    assert view.data_ptr() == scale.data_ptr()
    with pytest.raises(RuntimeError, match="requires a weight scale"):
        layer.w13_kernel._weight_scale_for_gmm(None)
    layer.w13_kernel.use_megamoe_canonical_layout = False
    assert layer.w13_kernel._weight_scale_for_gmm(scale) is scale


@pytest.mark.parametrize("malformation", ["shape", "odd_scale_axis", "scale_dtype"])
def test_mxfp8_postprocess_rejects_unsafe_layouts(monkeypatch, malformation):
    cls, layer = _mxfp8_layer(monkeypatch)
    if malformation == "shape":
        layer.w13_weight = layer.w13_weight.transpose(1, 2)
    elif malformation == "odd_scale_axis":
        layer.w13_weight_scale = torch.zeros(2, 128, 3, dtype=torch.uint8)
    else:
        layer.w13_weight_scale = layer.w13_weight_scale.float()
    with pytest.raises(RuntimeError):
        cls.maybe_process_megamoe_weights(layer)
