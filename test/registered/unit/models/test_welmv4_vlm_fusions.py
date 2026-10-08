import importlib.util
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from test_welmv4_vlm_vision import OPS, _load_definitions

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
SPEC = importlib.util.spec_from_file_location(
    "welmv4_vision_npu_test_helpers",
    ROOT / "python/sglang/srt/layers/welmv4_vision_npu.py",
)
HELPERS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HELPERS
SPEC.loader.exec_module(HELPERS)


@pytest.mark.parametrize("flag", ["USE_MC2", "FUSE_GELU", "FUSE_ADD_LAYERNORM"])
def test_flags_are_strict_and_independent(monkeypatch, flag):
    for suffix in ["USE_MC2", "FUSE_GELU", "FUSE_ADD_LAYERNORM"]:
        monkeypatch.delenv(f"SGLANG_VLM_NPU_{suffix}", raising=False)
    monkeypatch.setenv(f"SGLANG_VLM_NPU_{flag}", "true")
    config = HELPERS.VisionNPUFusionConfig.from_env()
    assert sum((config.mc2, config.gelu, config.add_layer_norm)) == 1
    monkeypatch.setenv(f"SGLANG_VLM_NPU_{flag}", "typo")
    with pytest.raises(ValueError):
        HELPERS.VisionNPUFusionConfig.from_env()


@pytest.mark.parametrize("options", [{"mc2_min_m": 0}, {"mc2_comm_mode": "auto"}])
def test_invalid_mc2_configuration_fails_early(options):
    with pytest.raises(ValueError):
        HELPERS.VisionNPUFusionConfig(**options)


def test_cpu_dispatch_retains_existing_linear_paths():
    hidden = torch.randn(3, 8).bfloat16()
    config = HELPERS.VisionNPUFusionConfig(mc2=True, gelu=True)
    assert HELPERS.vision_row_parallel_linear(None, hidden, config) is None
    assert (
        HELPERS.vision_column_parallel_linear_gelu(None, hidden, "none", config) is None
    )
    assert (
        HELPERS.vision_column_parallel_linear_gelu(None, hidden, "quick", config)
        is None
    )


def test_add_norm_reference_preserves_bf16_residual_rounding():
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(11, 72, generator=generator).bfloat16()
    residual = torch.randn(11, 72, generator=generator).bfloat16()
    norm = torch.nn.LayerNorm(72, eps=1e-6).bfloat16()
    summed, result = HELPERS.vision_add_layer_norm(
        hidden, residual, norm, HELPERS.VisionNPUFusionConfig(add_layer_norm=True)
    )
    expected = hidden + residual
    assert torch.equal(summed, expected)
    assert torch.equal(result, norm(expected))
    unrounded = F.layer_norm(
        hidden.float() + residual.float(),
        (72,),
        norm.weight.float(),
        norm.bias.float(),
        norm.eps,
    ).bfloat16()
    assert not torch.equal(result, unrounded)


def _install_mc2_mocks(monkeypatch, rank, operation):
    backend = SimpleNamespace(get_hccl_comm_name=Mock(return_value="vision_tp"))
    process_group = SimpleNamespace(
        _get_backend=Mock(return_value=backend), rank=lambda: rank
    )
    group = SimpleNamespace(
        world_size=4, rank_in_group=rank, device_group=process_group
    )
    distributed = types.ModuleType("sglang.srt.distributed")
    distributed.get_tp_group = lambda: group
    npu = types.ModuleType("torch_npu")
    npu.npu_mm_all_reduce_base = operation
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    monkeypatch.setattr(HELPERS, "_mc2_eligible", lambda *args: True)
    monkeypatch.setattr(
        HELPERS.VisionNPUFusionConfig, "validate_runtime", lambda *args: None
    )
    return group, backend


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("skip_bias", [False, True])
def test_mc2_preserves_bias_and_uses_one_collective(monkeypatch, rank, skip_bias):
    hidden = torch.randn(1, 3, 1076).bfloat16()
    layer = SimpleNamespace(
        tp_size=4,
        tp_rank=rank,
        skip_bias_add=skip_bias,
        weight=torch.randn(8, 1076).bfloat16(),
        bias=torch.randn(8).bfloat16(),
    )
    operation = Mock(return_value=torch.ones(3, 8).bfloat16())
    _, backend = _install_mc2_mocks(monkeypatch, rank, operation)
    config = HELPERS.VisionNPUFusionConfig(mc2=True)
    output, output_bias = HELPERS.vision_row_parallel_linear(layer, hidden, config)
    assert output.shape == (1, 3, 8)
    assert output_bias is (layer.bias if skip_bias else None)
    assert operation.call_count == 1
    positional, keyword = operation.call_args
    assert positional[0].shape == (3, 1076)
    assert positional[1].shape == (1076, 8)
    assert positional[2] == "vision_tp"
    assert keyword["bias"] is (layer.bias if rank == 0 and not skip_bias else None)
    assert keyword["reduce_op"] == "sum"
    assert keyword["comm_mode"] == "ccu"
    HELPERS.vision_row_parallel_linear(layer, hidden, config)
    assert backend.get_hccl_comm_name.call_count == 1


def test_mc2_collective_failure_is_not_swallowed(monkeypatch):
    operation = Mock(side_effect=RuntimeError("HCCL collective failed"))
    _install_mc2_mocks(monkeypatch, 0, operation)
    layer = SimpleNamespace(
        tp_size=4, tp_rank=0, skip_bias_add=False, weight=torch.ones(4, 8), bias=None
    )
    with pytest.raises(RuntimeError, match="HCCL collective failed"):
        HELPERS.vision_row_parallel_linear(
            layer, torch.ones(3, 8), HELPERS.VisionNPUFusionConfig(mc2=True)
        )


def test_mc2_wrong_tp_group_fails_before_collective(monkeypatch):
    operation = Mock()
    _install_mc2_mocks(monkeypatch, 0, operation)
    layer = SimpleNamespace(tp_size=2, tp_rank=0)
    with pytest.raises(RuntimeError, match="TP group"):
        HELPERS.vision_row_parallel_linear(
            layer, torch.ones(3, 8), HELPERS.VisionNPUFusionConfig(mc2=True)
        )
    operation.assert_not_called()


class _NPUTensorProxy:
    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_fused_gelu_dispatch_keeps_mode_and_bias(monkeypatch, approximate):
    hidden = _NPUTensorProxy(torch.randn(3, 72).bfloat16())
    layer = SimpleNamespace(
        weight=_NPUTensorProxy(torch.randn(1076, 72).bfloat16()),
        bias=_NPUTensorProxy(torch.randn(1076).bfloat16()),
        gather_output=False,
        skip_bias_add=False,
        quant_method=type("UnquantizedLinearMethod", (), {})(),
    )
    kernel = Mock(return_value=torch.zeros(3, 1076).bfloat16())
    module = types.ModuleType("sglang.srt.layers.welmv4_vision_npu_kernels")
    module.linear_gelu = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = HELPERS.vision_column_parallel_linear_gelu(
        layer, hidden, approximate, HELPERS.VisionNPUFusionConfig(gelu=True)
    )
    assert result.shape == (3, 1076)
    kernel.assert_called_once_with(hidden, layer.weight, layer.bias, approximate)


@pytest.mark.parametrize(
    "attribute,value", [("gather_output", True), ("skip_bias_add", True)]
)
def test_fused_gelu_falls_back_for_unsupported_column_contract(attribute, value):
    hidden = _NPUTensorProxy(torch.randn(3, 72).bfloat16())
    layer = SimpleNamespace(gather_output=False, skip_bias_add=False)
    setattr(layer, attribute, value)
    result = HELPERS.vision_column_parallel_linear_gelu(
        layer, hidden, "tanh", HELPERS.VisionNPUFusionConfig(gelu=True)
    )
    assert result is None


class _Linear(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.bias = torch.nn.Parameter(torch.randn(output_size) * 0.1) if bias else None

    def forward(self, hidden, skip_all_reduce=False, forward_batch=None):
        return F.linear(hidden, self.weight, self.bias), None


def _model_definitions(*names, **extra):
    return _load_definitions(
        *names,
        VisionNPUFusionConfig=HELPERS.VisionNPUFusionConfig,
        vision_column_parallel_linear_gelu=HELPERS.vision_column_parallel_linear_gelu,
        vision_row_parallel_linear=HELPERS.vision_row_parallel_linear,
        vision_add_layer_norm=HELPERS.vision_add_layer_norm,
        RowParallelLinear=_Linear,
        ColumnParallelLinear=_Linear,
        WeLMV4VisionRowParallelLinear=_Linear,
        _vision_dp_tp_kwargs=lambda use_dp: {},
        welmv4_vision_quick_gelu=OPS.welmv4_vision_quick_gelu,
        should_skip_mlp_all_reduce=lambda: False,
        add_prefix=lambda name, prefix: name,
        **extra,
    )


@pytest.mark.parametrize("activation", ["gelu_pytorch_tanh", "gelu", "quick_gelu"])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_model_mlp_fallback_preserves_all_activation_modes(
    monkeypatch, activation, enabled
):
    monkeypatch.setenv("SGLANG_VLM_NPU_FUSE_GELU", enabled)
    cls = _model_definitions("WeLMV4VisionMLP").WeLMV4VisionMLP
    model = cls(
        SimpleNamespace(hidden_size=8, intermediate_size=12, hidden_act=activation)
    ).bfloat16()
    hidden = torch.randn(3, 8).bfloat16()
    fc1 = model.fc1(hidden)[0]
    activated = (
        OPS.welmv4_vision_quick_gelu(fc1)
        if activation == "quick_gelu"
        else model.act(fc1)
    )
    reference = model.fc2(activated)[0]
    torch.testing.assert_close(model(hidden), reference, rtol=0, atol=0)
    assert set(model.state_dict()) == {
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
    }


def test_model_projector_preserves_exact_gelu_and_parameter_names():
    cls = _model_definitions("WeLMV4VisionProjector").WeLMV4VisionProjector
    model = cls(
        SimpleNamespace(hidden_size=8, spatial_merge_size=2, out_hidden_size=6)
    ).bfloat16()
    hidden = torch.randn(12, 8).bfloat16()
    x = model.ln_q(hidden).view(-1, 32)
    reference = model.mlp[2](F.gelu(model.mlp[0](x)[0], approximate="none"))[0]
    torch.testing.assert_close(model(hidden), reference, rtol=0, atol=0)
    assert model.mlp[1].approximate == "none"
    assert set(model.state_dict()) == {
        "ln_q.weight",
        "ln_q.bias",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
    }


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_model_block_retains_rounded_residual_norm(monkeypatch, enabled):
    monkeypatch.setenv("SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM", enabled)

    class _Attention(torch.nn.Module):
        def __init__(self, config, **kwargs):
            super().__init__()

        def forward(self, x, **kwargs):
            return x * 0.31

    mlp = _model_definitions("WeLMV4VisionMLP").WeLMV4VisionMLP
    cls = _model_definitions(
        "WeLMV4VisionBlock", WeLMV4VisionAttention=_Attention, WeLMV4VisionMLP=mlp
    ).WeLMV4VisionBlock
    model = cls(
        SimpleNamespace(
            hidden_size=8, intermediate_size=12, hidden_act="gelu_pytorch_tanh"
        )
    ).bfloat16()
    hidden = torch.randn(3, 8).bfloat16()
    residual = hidden + model.attn(model.norm1(hidden))
    expected = residual + model.mlp(model.norm2(residual))
    torch.testing.assert_close(model(hidden, None, None), expected, rtol=0, atol=0)


@pytest.mark.parametrize("skip", [False, True])
def test_row_subclass_fused_result_is_not_reduced_twice(monkeypatch, skip):
    cls = _model_definitions(
        "WeLMV4VisionRowParallelLinear"
    ).WeLMV4VisionRowParallelLinear
    model = cls(8, 4).bfloat16()
    hidden = torch.randn(3, 8).bfloat16()
    globals_ = cls.forward.__globals__
    fused_output = (torch.ones(3, 4).bfloat16(), None)
    dispatch = Mock(return_value=fused_output)
    monkeypatch.setitem(globals_, "vision_row_parallel_linear", dispatch)
    result = model(hidden, skip_all_reduce=skip)
    if skip:
        dispatch.assert_not_called()
        torch.testing.assert_close(
            result[0], F.linear(hidden, model.weight, model.bias), rtol=0, atol=0
        )
    else:
        dispatch.assert_called_once()
        assert result is fused_output


def test_attention_passes_local_projection_factory():
    class _AttentionBase(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    cls = _model_definitions(
        "WeLMV4VisionAttention",
        VisionAttention=_AttentionBase,
        _apply_welmv4_vision_rope=lambda *args: None,
    ).WeLMV4VisionAttention
    model = cls(SimpleNamespace(hidden_size=1152, num_attention_heads=16))
    assert model.kwargs["proj_linear_cls"] is _Linear
