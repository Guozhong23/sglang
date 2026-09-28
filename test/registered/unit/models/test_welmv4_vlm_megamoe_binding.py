"""Exercise the real runner admission method without starting NPU workers."""

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VL_ARCH = "WeLMV4VLMForConditionalGeneration"
TEXT_ARCH = "WeLMV4MoeForCausalLM"
SIDECAR_MODULE = "sglang.srt.hardware_backend.npu.moe.welmv4_megamoe"


@pytest.fixture
def binding(monkeypatch):
    """Load production logic; replace only configuration and device construction."""
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    tree = ast.parse(path.read_text())
    runner_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "maybe_init_welm_prefill_megamoe"
    )
    enabled = SimpleNamespace(value=True)
    parallel = SimpleNamespace(moe_tp_size=1)
    namespace = {
        "torch": torch,
        "envs": SimpleNamespace(
            WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: enabled.value)
        ),
        "get_parallel": lambda: parallel,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    sidecar = object()
    construct = Mock(return_value=sidecar)
    module = ModuleType(SIDECAR_MODULE)
    module.init_welm_prefill_megamoe = construct
    monkeypatch.setitem(sys.modules, SIDECAR_MODULE, module)
    runner = SimpleNamespace(
        device="npu",
        is_draft_worker=False,
        dtype=torch.bfloat16,
        ps=SimpleNamespace(moe_ep_size=4),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[VL_ARCH])
        ),
        welm_prefill_megamoe=object(),
    )
    return SimpleNamespace(
        initialize=namespace[method.name],
        runner=runner,
        enabled=enabled,
        parallel=parallel,
        construct=construct,
        sidecar=sidecar,
    )


@pytest.mark.parametrize("arch", [VL_ARCH, TEXT_ARCH])
@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_bf16_npu_target_binds_real_runner_to_sidecar(binding, arch, ep_size):
    binding.runner.model_config.hf_config.architectures = [arch]
    binding.runner.ps.moe_ep_size = ep_size
    binding.initialize(binding.runner)
    binding.construct.assert_called_once_with(binding.runner)
    assert binding.runner.welm_prefill_megamoe is binding.sidecar


@pytest.mark.parametrize("architectures", [["Other", VL_ARCH], ["Other", TEXT_ARCH]])
def test_supported_architecture_can_follow_another_architecture(binding, architectures):
    binding.runner.model_config.hf_config.architectures = architectures
    binding.initialize(binding.runner)
    binding.construct.assert_called_once_with(binding.runner)
    assert binding.runner.welm_prefill_megamoe is binding.sidecar


@pytest.mark.parametrize(
    "change",
    [
        "disabled",
        "cpu",
        "cuda",
        "draft",
        "fp16",
        "fp32",
        "ep_zero",
        "ep_one",
        "moe_tp_two",
        "wrong_arch",
        "nextn_arch",
        "empty_arch",
        "none_arch",
    ],
)
def test_ineligible_runner_clears_binding_without_constructing_sidecar(binding, change):
    runner = binding.runner
    if change == "disabled":
        binding.enabled.value = False
    elif change in ("cpu", "cuda"):
        runner.device = change
    elif change == "draft":
        runner.is_draft_worker = True
    elif change in ("fp16", "fp32"):
        runner.dtype = torch.float16 if change == "fp16" else torch.float32
    elif change in ("ep_zero", "ep_one"):
        runner.ps.moe_ep_size = 0 if change == "ep_zero" else 1
    elif change == "moe_tp_two":
        binding.parallel.moe_tp_size = 2
    else:
        runner.model_config.hf_config.architectures = {
            "wrong_arch": ["Qwen2MoeForCausalLM"],
            "nextn_arch": ["WeLMV4MoeForCausalLMNextN"],
            "empty_arch": [],
            "none_arch": None,
        }[change]
    binding.initialize(runner)
    binding.construct.assert_not_called()
    assert runner.welm_prefill_megamoe is None


def test_optional_sidecar_may_decline_binding(binding):
    binding.construct.return_value = None
    binding.initialize(binding.runner)
    binding.construct.assert_called_once_with(binding.runner)
    assert binding.runner.welm_prefill_megamoe is None


def test_sidecar_initialization_error_is_not_silently_ignored(binding):
    binding.construct.side_effect = RuntimeError("invalid expert weight layout")
    with pytest.raises(RuntimeError, match="invalid expert weight layout"):
        binding.initialize(binding.runner)
    binding.construct.assert_called_once_with(binding.runner)
    assert binding.runner.welm_prefill_megamoe is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
