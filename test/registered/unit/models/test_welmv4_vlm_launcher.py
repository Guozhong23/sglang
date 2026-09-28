"""CPU contract tests of the real WeLM VL launcher using a Python recorder.

No checkpoint code, SGLang runtime, or accelerator is loaded. Each invocation
records the exact preflight/launch argv and environment before returning.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
LAUNCHER = ROOT / "examples/runtime/welm_vl/run_950pr.sh"
PERFORMANCE_FLAGS = (
    "WELM_NPU_USE_MEGAMOE",
    "WELM_NPU_USE_FLASH_ATTN",
    "SGLANG_NPU_WELMV4_FUSED_QKV",
    "SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER",
    "SGLANG_NPU_MOE_GATING_TOPK_SIGMOID_NO_RENORM",
    "SGLANG_NPU_USE_MULTI_STREAM",
)
RECORDED_ENV = PERFORMANCE_FLAGS + (
    "SGLANG_VIT_ENABLE_CUDA_GRAPH",
    "PYTHONPATH",
)


@pytest.fixture
def invoke_launcher(tmp_path):
    recorder = tmp_path / "python recorder"
    recordings = tmp_path / "calls.jsonl"
    checkpoint = tmp_path / "VL checkpoint"
    checkpoint.mkdir()
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"keys = {RECORDED_ENV!r}\n"
        "record = {'argv': sys.argv[1:], "
        "'env': {key: os.environ.get(key) for key in keys}}\n"
        "with open(os.environ['WELM_TEST_RECORDER_OUT'], 'a') as output:\n"
        "    output.write(json.dumps(record) + '\\n')\n"
        "if sys.argv[1:3] != ['-m', 'sglang.launch_server']:\n"
        "    sys.exit(int(os.environ.get('WELM_TEST_PREFLIGHT_EXIT', '0')))\n"
    )
    recorder.chmod(0o755)

    def invoke(*args, **environment):
        recordings.unlink(missing_ok=True)
        # Deliberately exclude inherited WeLM/device flags to make defaults
        # reproducible; tests inject inherited overrides explicitly.
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "MODEL_PATH": str(checkpoint),
            "PYTHON_BIN": str(recorder),
            "WELM_TEST_RECORDER_OUT": str(recordings),
            **environment,
        }
        result = subprocess.run(
            ["bash", str(LAUNCHER), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        calls = (
            [json.loads(line) for line in recordings.read_text().splitlines()]
            if recordings.exists()
            else []
        )
        return result, calls

    return invoke


def _option(argv, name):
    assert name in argv, f"missing {name}: {argv}"
    index = argv.index(name)
    assert index + 1 < len(argv), f"missing value for {name}"
    return argv[index + 1]


def _successful_calls(result, calls):
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2, calls
    preflight, launch = calls
    assert Path(preflight["argv"][0]).name == "check_model.py"
    assert "--runtime" in preflight["argv"]
    assert launch["argv"][:2] == ["-m", "sglang.launch_server"]
    assert _option(launch["argv"], "--model-path") == preflight["argv"][1]
    assert _option(launch["argv"], "--dtype") == "bfloat16"
    assert "--enable-multimodal" in launch["argv"]
    assert "--trust-remote-code" in launch["argv"]
    assert "--disable-prefill-cuda-graph" in launch["argv"]
    expected_profile = (
        "optimized"
        if _option(launch["argv"], "--moe-a2a-backend") == "deepep"
        else "baseline"
    )
    assert _option(preflight["argv"], "--profile") == expected_profile
    for call in calls:
        assert call["env"]["SGLANG_VIT_ENABLE_CUDA_GRAPH"] == "0"
        assert call["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "python")
    return preflight, launch


def _assert_optimized(launch, tp):
    argv = launch["argv"]
    assert _option(argv, "--tp-size") == str(tp)
    assert _option(argv, "--ep-size") == str(tp)
    assert _option(argv, "--moe-a2a-backend") == "deepep"
    assert _option(argv, "--deepep-mode") == "auto"
    assert _option(argv, "--chunked-prefill-size") == "16384"
    assert "--disable-cuda-graph" not in argv
    assert "--disable-radix-cache" not in argv


def test_default_profile_uses_optimized_tp4(invoke_launcher):
    result, calls = invoke_launcher()
    preflight, launch = _successful_calls(result, calls)
    _assert_optimized(launch, 4)
    assert _option(preflight["argv"], "--tp") == "4"
    for call in calls:
        for flag in PERFORMANCE_FLAGS[:3]:
            assert call["env"][flag] == "1"


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_optimized_supports_requested_tp_and_limits_default_fused_qkv(
    invoke_launcher, tp
):
    result, calls = invoke_launcher(WELM_VL_PROFILE="optimized", TP_SIZE=str(tp))
    preflight, launch = _successful_calls(result, calls)
    _assert_optimized(launch, tp)
    assert _option(preflight["argv"], "--tp") == str(tp)
    for call in calls:
        assert call["env"]["SGLANG_NPU_WELMV4_FUSED_QKV"] == ("1" if tp == 4 else "0")
        assert call["env"]["WELM_NPU_USE_MEGAMOE"] == "1"
        assert call["env"]["WELM_NPU_USE_FLASH_ATTN"] == "1"


def test_optimized_respects_explicit_zero_performance_flags(invoke_launcher):
    result, calls = invoke_launcher(**dict.fromkeys(PERFORMANCE_FLAGS, "0"))
    _, launch = _successful_calls(result, calls)
    _assert_optimized(launch, 4)
    for call in calls:
        assert all(call["env"][flag] == "0" for flag in PERFORMANCE_FLAGS)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_baseline_disables_optimizations_and_inherited_flags(invoke_launcher, tp):
    result, calls = invoke_launcher(
        WELM_VL_PROFILE="baseline",
        TP_SIZE=str(tp),
        SGLANG_VIT_ENABLE_CUDA_GRAPH="1",
        MAX_RUNNING_REQUESTS="99",
        CHUNKED_PREFILL_SIZE="2048",
        CUDA_GRAPH_MAX_BS="64",
        **dict.fromkeys(PERFORMANCE_FLAGS, "1"),
    )
    preflight, launch = _successful_calls(result, calls)
    argv = launch["argv"]
    assert _option(preflight["argv"], "--tp") == str(tp)
    assert _option(argv, "--tp-size") == str(tp)
    assert _option(argv, "--ep-size") == "1"
    assert _option(argv, "--moe-a2a-backend") == "none"
    assert _option(argv, "--chunked-prefill-size") == "-1"
    assert _option(argv, "--max-running-requests") == "1"
    assert "--disable-cuda-graph" in argv
    assert "--disable-radix-cache" in argv
    for call in calls:
        assert all(call["env"][flag] == "0" for flag in PERFORMANCE_FLAGS)


@pytest.mark.parametrize("profile", ["optimized", "baseline"])
def test_profiles_preserve_template_base_device_and_trailing_args(
    invoke_launcher, profile, tmp_path
):
    template = str(tmp_path / "original VL template.jinja")
    trailing = ["--log-level", "debug", "--served-model-name", "model with spaces"]
    result, calls = invoke_launcher(
        *trailing,
        WELM_VL_PROFILE=profile,
        CHAT_TEMPLATE=template,
        BASE_DEVICE="8",
        PYTHONPATH="/existing python path",
    )
    preflight, launch = _successful_calls(result, calls)
    assert _option(preflight["argv"], "--chat-template") == template
    assert _option(launch["argv"], "--chat-template") == template
    assert _option(preflight["argv"], "--base-device") == "8"
    assert _option(launch["argv"], "--base-gpu-id") == "8"
    assert launch["argv"][-len(trailing) :] == trailing
    assert launch["env"]["PYTHONPATH"].split(os.pathsep)[1:] == [
        "/existing python path"
    ]


@pytest.mark.parametrize("profile", ["optimized", "baseline"])
def test_failed_preflight_never_launches_server(invoke_launcher, profile):
    result, calls = invoke_launcher(
        WELM_VL_PROFILE=profile, WELM_TEST_PREFLIGHT_EXIT="23"
    )
    assert result.returncode == 23
    assert len(calls) == 1
    assert Path(calls[0]["argv"][0]).name == "check_model.py"


@pytest.mark.parametrize(
    "profile,tp",
    [
        ("optimized", "1"),
        ("optimized", "3"),
        ("optimized", "16"),
        ("baseline", "3"),
        ("baseline", "16"),
        ("optimized", "four"),
    ],
)
def test_invalid_tp_fails_with_diagnostic_before_preflight(
    invoke_launcher, profile, tp
):
    result, calls = invoke_launcher(WELM_VL_PROFILE=profile, TP_SIZE=tp)
    assert result.returncode != 0
    assert calls == []
    diagnostic = result.stdout + result.stderr
    assert "TP" in diagnostic or "tp" in diagnostic
    assert all(allowed in diagnostic for allowed in ("2", "4", "8"))


def test_unknown_profile_fails_with_diagnostic_before_preflight(invoke_launcher):
    result, calls = invoke_launcher(WELM_VL_PROFILE="unknown-profile")
    assert result.returncode != 0
    assert calls == []
    assert "unknown-profile" in result.stdout + result.stderr
