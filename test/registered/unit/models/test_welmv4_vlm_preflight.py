"""CPU coverage for profile-aware VL extension checks and artifact-only mode."""

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MEGAMOE = "npu_ops_transformer.ops.mega_moe"
FLASH = "cann_ops_transformer"
FUSED = "sglang.srt.layers.fused_qkv_proj_norm_rope_cache"
FLAGS = (
    "WELM_NPU_USE_MEGAMOE",
    "WELM_NPU_USE_FLASH_ATTN",
    "SGLANG_NPU_WELMV4_FUSED_QKV",
)
APIS = {
    "deep_ep": ("Buffer", "Config"),
    MEGAMOE: ("mega_moe", "get_symm_buffer_for_mega_moe"),
    FLASH: ("flash_attn", "flash_attn_metadata"),
    FUSED: ("compile_aot", "compile_aot_rank_chunk"),
}


@pytest.fixture
def preflight(monkeypatch):
    path = ROOT / "examples/runtime/welm_vl/check_model.py"
    spec = importlib.util.spec_from_file_location("welm_vl_preflight_deps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    extensions = {
        name: SimpleNamespace(**{symbol: Mock() for symbol in symbols})
        for name, symbols in APIS.items()
    }
    importer = Mock(side_effect=extensions.__getitem__)
    monkeypatch.setattr(module.importlib, "import_module", importer)
    return SimpleNamespace(module=module, extensions=extensions, importer=importer)


def _enable_all(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, "1")


def test_default_baseline_needs_no_optional_extensions(preflight):
    preflight.module.check_optimization_dependencies(tp=4)
    preflight.importer.assert_not_called()


def test_optimized_always_checks_deepep_even_with_optional_features_disabled(preflight):
    preflight.module.check_optimization_dependencies(tp=4, profile="optimized")
    preflight.importer.assert_called_once_with("deep_ep")
    for extension in preflight.extensions.values():
        for api in vars(extension).values():
            api.assert_not_called()


def test_enabled_features_import_actual_runtime_modules_without_executing_apis(
    preflight, monkeypatch
):
    _enable_all(monkeypatch)
    preflight.module.check_optimization_dependencies(tp=4, profile="optimized")
    assert [call.args[0] for call in preflight.importer.call_args_list] == list(APIS)
    for extension in preflight.extensions.values():
        for api in vars(extension).values():
            api.assert_not_called()


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_fused_qkv_check_respects_actual_tp4_specialization(preflight, monkeypatch, tp):
    monkeypatch.setenv("SGLANG_NPU_WELMV4_FUSED_QKV", "1")
    preflight.module.check_optimization_dependencies(tp=tp)
    preflight.importer.assert_not_called()


@pytest.mark.parametrize(
    "flag,value,module_name",
    [
        (FLAGS[0], "1", MEGAMOE),
        (FLAGS[0], "TRUE", MEGAMOE),
        (FLAGS[0], "yes", MEGAMOE),
        (FLAGS[1], "1", FLASH),
        (FLAGS[1], "True", FLASH),
        (FLAGS[2], "1", FUSED),
    ],
)
def test_standalone_runtime_checks_explicitly_enabled_features(
    preflight, monkeypatch, flag, value, module_name
):
    monkeypatch.setenv(flag, value)
    preflight.module.check_optimization_dependencies(tp=4)
    preflight.importer.assert_called_once_with(module_name)


@pytest.mark.parametrize("value", ["0", "false"])
def test_disabled_flags_do_not_import_extensions(preflight, monkeypatch, value):
    for flag in FLAGS:
        monkeypatch.setenv(flag, value)
    preflight.module.check_optimization_dependencies(tp=4)
    preflight.importer.assert_not_called()


def test_fused_qkv_true_string_is_disabled_like_production(preflight, monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_WELMV4_FUSED_QKV", "true")
    preflight.module.check_optimization_dependencies(tp=4)
    preflight.importer.assert_not_called()


@pytest.mark.parametrize(
    "module_name,symbol",
    [(module, symbol) for module, symbols in APIS.items() for symbol in symbols],
)
def test_missing_or_noncallable_api_reports_extension_and_symbol(
    preflight, monkeypatch, module_name, symbol
):
    _enable_all(monkeypatch)
    setattr(preflight.extensions[module_name], symbol, None)
    with pytest.raises(ValueError) as error:
        preflight.module.check_optimization_dependencies(tp=4, profile="optimized")
    assert module_name in str(error.value)
    assert symbol in str(error.value)
    assert "missing callable API" in str(error.value)
    if module_name == FUSED:
        assert "cannbotdsl" in str(error.value)


@pytest.mark.parametrize(
    "module_name,error",
    [
        ("deep_ep", ModuleNotFoundError("No module named deep_ep")),
        (MEGAMOE, OSError("libcustom_transformer.so: cannot open shared object")),
        (FLASH, RuntimeError("undefined symbol in libflash_attn.so")),
        (FUSED, ImportError("cannot import name ChannelKind from cannbotdsl")),
    ],
)
def test_import_failures_report_extension_and_underlying_failure(
    preflight, monkeypatch, module_name, error
):
    _enable_all(monkeypatch)

    def import_extension(name):
        if name == module_name:
            raise error
        return preflight.extensions[name]

    preflight.importer.side_effect = import_extension
    with pytest.raises(ValueError) as caught:
        preflight.module.check_optimization_dependencies(tp=4, profile="optimized")
    assert module_name in str(caught.value)
    assert str(error) in str(caught.value)
    assert caught.value.__cause__ is error


def _configure_cli(preflight, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["check_model.py", "/checkpoint", *args])
    monkeypatch.setattr(
        preflight.module,
        "check_artifacts",
        lambda _: {
            "text_config": {"num_attention_heads": 24},
            "vision_config": {"num_attention_heads": 16, "intermediate_size": 4096},
        },
    )


def test_artifact_only_mode_never_imports_enabled_extensions(preflight, monkeypatch):
    _enable_all(monkeypatch)
    _configure_cli(preflight, monkeypatch, "--profile", "optimized")
    runtime = Mock()
    monkeypatch.setattr(preflight.module, "check_runtime", runtime)
    preflight.module.main()
    preflight.importer.assert_not_called()
    runtime.assert_not_called()


@pytest.mark.parametrize("profile", [None, "baseline", "optimized"])
def test_cli_forwards_profile_to_runtime_with_standalone_baseline_default(
    preflight, monkeypatch, profile
):
    args = ["--runtime"]
    if profile is not None:
        args.extend(["--profile", profile])
    _configure_cli(preflight, monkeypatch, *args)
    runtime = Mock()
    monkeypatch.setattr(preflight.module, "check_runtime", runtime)
    preflight.module.main()
    runtime.assert_called_once_with(
        Path("/checkpoint"), 4, 0, None, profile or "baseline"
    )


def test_unknown_profile_is_rejected_before_imports(preflight):
    with pytest.raises(ValueError, match="Unknown WeLM-VL profile"):
        preflight.module.check_optimization_dependencies(tp=4, profile="unknown")
    preflight.importer.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
