import ast
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


@pytest.mark.parametrize("flag", ["FUSE_ROPE", "FUSE_GELU", "FUSE_ADD_LAYERNORM"])
def test_flags_are_strict_and_independent(monkeypatch, flag):
    for suffix in ["FUSE_ROPE", "FUSE_GELU", "FUSE_ADD_LAYERNORM"]:
        monkeypatch.delenv(f"SGLANG_VLM_NPU_{suffix}", raising=False)
    monkeypatch.setenv(f"SGLANG_VLM_NPU_{flag}", "true")
    config = HELPERS.VisionNPUFusionConfig.from_env()
    assert sum((config.rope, config.gelu, config.add_layer_norm)) == 1
    monkeypatch.setenv(f"SGLANG_VLM_NPU_{flag}", "typo")
    with pytest.raises(ValueError):
        HELPERS.VisionNPUFusionConfig.from_env()


@pytest.mark.parametrize("value", ["1", "true"])
def test_removed_mc2_flag_fails_early(monkeypatch, value):
    monkeypatch.setenv("SGLANG_VLM_NPU_USE_MC2", value)
    with pytest.raises(ValueError, match="MC2 has been removed"):
        HELPERS.VisionNPUFusionConfig.from_env()


def test_cpu_dispatch_retains_existing_linear_paths():
    hidden = torch.randn(3, 8).bfloat16()
    config = HELPERS.VisionNPUFusionConfig(gelu=True)
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
        vision_add_layer_norm=HELPERS.vision_add_layer_norm,
        RowParallelLinear=_Linear,
        ColumnParallelLinear=_Linear,
        _vision_dp_tp_kwargs=lambda use_dp: {},
        welmv4_vision_quick_gelu=OPS.welmv4_vision_quick_gelu,
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
    assert model.kwargs.get("proj_linear_cls", _Linear) is _Linear


@pytest.mark.parametrize("bias_shape", [(1,), (2, 1076), (1077,)])
def test_fused_gelu_rejects_incompatible_bias_shape(bias_shape):
    layer = SimpleNamespace(
        weight=_NPUTensorProxy(torch.randn(1076, 72).bfloat16()),
        bias=_NPUTensorProxy(torch.randn(*bias_shape).bfloat16()),
        gather_output=False,
        skip_bias_add=False,
        quant_method=type("UnquantizedLinearMethod", (), {})(),
    )
    result = HELPERS.vision_column_parallel_linear_gelu(
        layer,
        _NPUTensorProxy(torch.randn(3, 72).bfloat16()),
        "tanh",
        HELPERS.VisionNPUFusionConfig(gelu=True),
    )
    assert result is None


def _kernel_definition(name, **globals_):
    path = ROOT / "python/sglang/srt/layers/welmv4_vision_npu_kernels.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, **globals_}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_gelu_runtime_checks_only_the_requested_activation():
    validate = _kernel_definition(
        "validate_gelu_runtime", tl=SimpleNamespace(erf=Mock()), libdevice=None
    )
    validate("none")
    with pytest.raises(RuntimeError, match="libdevice.tanh"):
        validate("tanh")
    validate.__globals__["tl"] = SimpleNamespace()
    validate.__globals__["libdevice"] = SimpleNamespace(tanh=Mock())
    validate("tanh")
    with pytest.raises(RuntimeError, match="tl.erf"):
        validate("none")
    with pytest.raises(ValueError, match="Unsupported"):
        validate("quick")


class _TensorPointer:
    """CPU storage pointer for evaluating the actual elementwise Triton body."""

    def __init__(self, tensor, offsets=None):
        self.storage = tensor.as_strided(
            (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0
        )
        self.offsets = tensor.storage_offset() if offsets is None else offsets

    def __add__(self, offset):
        return _TensorPointer(self.storage, self.offsets + offset)


class _ElementwiseTL:
    constexpr = None
    float32 = torch.float32
    program = 0

    def program_id(self, _):
        return self.program

    def arange(self, start, end):
        return torch.arange(start, end)

    def load(self, pointer, mask, other):
        output = torch.full(mask.shape, other, dtype=pointer.storage.dtype)
        output[mask] = pointer.storage[pointer.offsets[mask]]
        return output

    def store(self, pointer, value, mask):
        pointer.storage[pointer.offsets[mask]] = value[mask].to(pointer.storage.dtype)


@pytest.mark.parametrize("heads", [4, 16])
@pytest.mark.parametrize("full_width", [False, True])
def test_fused_rope_actual_kernel_handles_head72_strides_and_tail(heads, full_width):
    generator = torch.Generator().manual_seed(178)
    tokens, dim = 5, 72
    # Split QKV views are strided, and K need not share Q's strides.
    q = torch.randn(tokens * 2, heads, 3, dim * 2, generator=generator).bfloat16()[
        ::2, :, 0, ::2
    ]
    k = (
        torch.randn(heads, tokens, dim * 2, generator=generator)
        .bfloat16()
        .permute(1, 0, 2)[..., ::2]
    )
    width = dim if full_width else dim // 2
    # The second half is intentionally different: only the first half is read.
    cos = torch.randn(tokens, width * 2, generator=generator)[:, ::2]
    sin = torch.randn(tokens * 2, width, generator=generator)[::2]
    expected = OPS.welmv4_vision_apply_rope(q, k, cos, sin)
    actual = (
        torch.full(q.shape, float("nan")).bfloat16(),
        torch.full(k.shape, float("nan")).bfloat16(),
    )
    tl = _ElementwiseTL()
    kernel = _kernel_definition("_apply_rope_kernel", tl=tl)
    pairs = tokens * heads * (dim // 2)
    for program in range((pairs + 255) // 256):
        tl.program = program
        kernel(
            *map(_TensorPointer, (q, k, cos, sin, *actual)),
            pairs,
            heads,
            dim // 2,
            *q.stride(),
            *k.stride(),
            *cos.stride(),
            *sin.stride(),
            256,
        )
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    rounded_coefficients = OPS.welmv4_vision_apply_rope(
        q, k, cos.bfloat16(), sin.bfloat16()
    )
    assert not torch.equal(expected[0], rounded_coefficients[0])


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_rope_cpu_reference_remains_available_when_fusion_enabled(monkeypatch, enabled):
    monkeypatch.setenv("SGLANG_VLM_NPU_FUSE_ROPE", enabled)
    q = torch.randn(3, 4, 72).bfloat16()
    k = torch.randn_like(q)
    cos, sin = torch.randn(3, 36), torch.randn(3, 36)
    output = OPS.welmv4_vision_apply_rope(q, k, cos, sin)
    assert output[0].dtype == torch.bfloat16
    assert output[0].device.type == "cpu"


def test_npu_rope_dispatch_passes_strided_qk_and_fp32_coefficients(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_NPU_FUSE_ROPE", "1")
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.welmv4_vision_npu", HELPERS)
    qkv = torch.randn(3, 4, 3, 72).bfloat16()
    query, key = _NPUTensorProxy(qkv[:, :, 0, :]), _NPUTensorProxy(qkv[:, :, 1, :])
    cos, sin = _NPUTensorProxy(torch.randn(3, 36)), _NPUTensorProxy(torch.randn(3, 36))
    assert not query.is_contiguous()
    expected = (torch.zeros(3, 4, 72).bfloat16(), torch.zeros(3, 4, 72).bfloat16())
    kernel = Mock(return_value=expected)
    module = types.ModuleType("sglang.srt.layers.welmv4_vision_npu_kernels")
    module.apply_rope = kernel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert OPS.welmv4_vision_apply_rope(query, key, cos, sin) is expected
    kernel.assert_called_once_with(query, key, cos, sin)
