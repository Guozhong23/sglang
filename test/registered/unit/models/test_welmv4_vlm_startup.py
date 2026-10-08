"""Exercise startup policy and checkpoint checks without importing NPU workers."""

import ast
import importlib.util
import json
import logging
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
FIXTURE = Path(__file__).parent / "fixtures/welmv4_vlm_config.json"
ARCH = "WeLMV4VLMForConditionalGeneration"


def _definitions(path, names, namespace):
    """Execute the actual CPU-only definitions, omitting runtime-only imports."""
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef):
            nodes.extend(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name in names
            )
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _model_helpers():
    return _definitions(
        SRT / "configs/model_config.py",
        {
            "WELMV4_MODEL_ARCHS",
            "MIMO_V2_MODEL_ARCHS",
            "MIMO_V2_MULTIMODAL_ARCHS",
            "multimodal_model_archs",
            "multimodal_breakable_cuda_graph_supported_model_archs",
            "is_multimodal_breakable_cuda_graph_supported",
            "is_multimodal_model",
            "is_hybrid_swa_model",
            "_detect_attention_sinks",
            "_get_welmv4_full_window_limit",
            "get_welmv4_layerwise_sliding_windows",
            "get_hybrid_layer_ids",
        },
        {"List": List, "Optional": Optional, "PretrainedConfig": object},
    )


def test_real_checkpoint_multimodal_sinks_and_48_layer_windows():
    raw = json.loads(FIXTURE.read_text())
    text = SimpleNamespace(**raw["text_config"])
    helpers = _model_helpers()
    assert helpers["is_multimodal_model"]([ARCH])
    assert helpers["is_multimodal_breakable_cuda_graph_supported"]([ARCH])
    assert helpers["is_hybrid_swa_model"]([ARCH], text)
    model = SimpleNamespace(hf_config=SimpleNamespace(**raw), hf_text_config=text)
    assert helpers["_detect_attention_sinks"](model)
    swa, full = helpers["get_hybrid_layer_ids"]([ARCH], text, context_len=8192)
    assert set(swa) | set(full) == set(range(48))
    assert not set(swa) & set(full)
    assert swa == [
        i
        for i, window in enumerate(text.sliding_window_size_layerwise[:48])
        if window == 512
    ]
    # MTP layer IDs must not allocate a target KV-cache layer.
    assert all(layer < 48 for layer in swa + full)


def test_nested_text_config_enables_oe_history_table():
    raw = json.loads(FIXTURE.read_text())
    model = SimpleNamespace(
        hf_config=SimpleNamespace(**raw),
        hf_text_config=SimpleNamespace(**raw["text_config"]),
    )
    tree = ast.parse((SRT / "configs/model_config.py").read_text())
    fields = {"use_ngram_embedding", "ngram_embedding_n", "ngram_embedding_k"}
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            (isinstance(target, ast.Name) and target.id == "oe_grams")
            or (isinstance(target, ast.Attribute) and target.attr in fields)
            for target in node.targets
        ):
            nodes.append(node)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<OE initialization>", "exec"),
        {"self": model},
    )
    assert model.use_ngram_embedding
    assert model.ngram_embedding_n == 3
    assert model.ngram_embedding_k == 4


def _server_args(**changes):
    values = dict(
        speculative_algorithm=None,
        enable_dp_attention=False,
        mm_enable_dp_encoder=False,
        encoder_only=False,
        language_only=False,
        disaggregation_mode="null",
        enable_pdmux=False,
        pp_size=1,
        ep_size=1,
        tp_size=4,
        moe_dp_size=1,
        moe_a2a_backend="none",
        quantization=None,
        dtype="bfloat16",
        enable_multimodal=None,
        enable_mixed_chunk=True,
        dcp_size=1,
        attn_cp_size=1,
        kv_cache_dtype="auto",
        enable_lora=False,
        chunked_prefill_size=4096,
        disable_radix_cache=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="breakable"),
        ),
    )
    values.update(changes)
    args = SimpleNamespace(**values)
    args._resolved = lambda: args
    args.get_model_config = lambda: SimpleNamespace(dtype=torch.bfloat16)
    return args


def _startup_policy(megamoe=False, vit_graph=False):
    return _definitions(
        SRT / "server_args.py",
        {"_handle_welm_vlm_adjustments"},
        {
            "Backend": SimpleNamespace(DISABLED="disabled"),
            "envs": SimpleNamespace(
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe),
                SGLANG_VIT_ENABLE_CUDA_GRAPH=SimpleNamespace(get=lambda: vit_graph),
            ),
            "logger": logging.getLogger(__name__),
        },
    )["_handle_welm_vlm_adjustments"]


def test_startup_keeps_requested_decode_graph_chunk_and_cache_settings():
    args = _server_args()
    _startup_policy()(args)
    assert args.enable_multimodal
    assert not args.disable_radix_cache and args.chunked_prefill_size == 4096
    assert args.cuda_graph_config.decode.backend == "full"
    assert args.cuda_graph_config.prefill.backend == "breakable"


@pytest.mark.parametrize(
    "changes",
    [
        {"speculative_algorithm": "NEXTN"},
        {"enable_dp_attention": True},
        {"mm_enable_dp_encoder": True},
        {"encoder_only": True},
        {"language_only": True},
        {"disaggregation_mode": "prefill"},
        {"enable_pdmux": True},
        {"pp_size": 2},
        {"moe_a2a_backend": "megamoe"},
        {"quantization": "modelslim"},
        {"dtype": "float16"},
        {"enable_multimodal": False},
    ],
)
def test_basic_startup_rejects_unsupported_combinations(changes):
    with pytest.raises(ValueError, match="WeLM-VL"):
        _startup_policy()(_server_args(**changes))


def test_startup_defers_ep_validation_until_deepep_resolution():
    # DeepEP changes EP=1 to EP=TP later in ServerArgs initialization.
    args = _server_args(moe_a2a_backend="deepep")
    _startup_policy(megamoe=True)(args)
    assert args.moe_a2a_backend == "deepep"


def _parallel_policy(megamoe=False):
    return _definitions(
        SRT / "server_args.py",
        {"_validate_welm_vlm_parallel_config"},
        {
            "resolved_view": lambda args: args,
            "envs": SimpleNamespace(
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe)
            ),
        },
    )["_validate_welm_vlm_parallel_config"]


@pytest.mark.parametrize("megamoe", [False, True])
def test_resolved_deepep_ep_tp_is_accepted(megamoe):
    _parallel_policy(megamoe)(_server_args(moe_a2a_backend="deepep", ep_size=4))


@pytest.mark.parametrize(
    "changes,megamoe",
    [
        ({"ep_size": 4}, False),
        # An environment override can change the backend after the early policy.
        ({"moe_a2a_backend": "megamoe"}, False),
        ({"moe_a2a_backend": "deepep", "ep_size": 2}, False),
        ({"moe_a2a_backend": "deepep", "ep_size": 4, "moe_dp_size": 2}, False),
        ({}, True),
        ({"moe_a2a_backend": "deepep", "ep_size": 1, "tp_size": 1}, True),
    ],
)
def test_resolved_parallel_config_rejects_incompatible_layouts(changes, megamoe):
    with pytest.raises(ValueError, match="WeLM-VL"):
        _parallel_policy(megamoe)(_server_args(**changes))


def test_explicit_baseline_options_remain_disabled():
    args = _server_args(chunked_prefill_size=-1, disable_radix_cache=True)
    args.cuda_graph_config.decode.backend = "disabled"
    args.cuda_graph_config.prefill.backend = "disabled"
    _startup_policy()(args)
    _parallel_policy()(args)
    assert args.chunked_prefill_size == -1 and args.disable_radix_cache
    assert args.cuda_graph_config.decode.backend == "disabled"


@pytest.mark.parametrize("native_flash", [False, True])
def test_decode_graph_uses_nested_vl_text_metadata(native_flash):
    raw = json.loads(FIXTURE.read_text())
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(**raw),
        hf_text_config=SimpleNamespace(**raw["text_config"]),
    )
    helper = _definitions(
        SRT / "hardware_backend/npu/graph_runner/npu_graph_runner.py",
        {"welmv4_graph_uses_device_attention_metadata"},
        {"ModelRunner": object, "get_bool_env_var": lambda *_: native_flash},
    )["welmv4_graph_uses_device_attention_metadata"]
    runner = SimpleNamespace(model_config=model_config)
    assert helper(runner)
    model_config.hf_text_config.enable_attn_sink_layerwise[0] = False
    assert helper(runner) == native_flash


def test_basic_startup_rejects_inherited_vision_graphs():
    with pytest.raises(ValueError, match="SGLANG_VIT_ENABLE_CUDA_GRAPH"):
        _startup_policy(vit_graph=True)(_server_args())


def _artifact_checker():
    path = ROOT / "examples/runtime/welm_vl/check_model.py"
    spec = importlib.util.spec_from_file_location("welm_vl_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_artifacts


def test_config_only_directory_reports_all_missing_artifacts(tmp_path):
    (tmp_path / "config.json").write_text(FIXTURE.read_text())
    with pytest.raises(ValueError) as error:
        _artifact_checker()(tmp_path)
    for missing in ("configuration_welmv4_vlm.py", "tokenizer", "processor", "weights"):
        assert missing in str(error.value)


def test_artifact_checker_catches_missing_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text(FIXTURE.read_text())
    for name in (
        "configuration_welmv4_vlm.py",
        "configuration_welmv4_moe.py",
        "tokenizer_config.json",
        "processor_config.json",
        "model-1.safetensors",
    ):
        (tmp_path / name).touch()
    index = {"weight_map": {"vision_encoder.weight": "model-2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="model-2.safetensors"):
        _artifact_checker()(tmp_path)
    (tmp_path / "model-2.safetensors").touch()
    assert _artifact_checker()(tmp_path)["architectures"] == [ARCH]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def _apply_shared_welm_prefill_policy(args, monkeypatch, native_flash=True, npu=True):
    """Execute the production policy nodes shared by text and VL architectures."""
    monkeypatch.setenv("WELM_NPU_USE_FLASH_ATTN", "1" if native_flash else "0")
    tree = ast.parse((SRT / "server_args.py").read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_handle_model_specific_adjustments"
    )
    nodes = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "welm_bf16_breakable"
            for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if (
                test.startswith("self.enable_mixed_chunk and")
                or "not welm_bf16_breakable" in test
            ):
                nodes.append(node)
    assert len(nodes) == 3
    nodes.sort(key=lambda node: node.lineno)
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), "<WeLM prefill policy>", "exec"
        ),
        {
            "self": args,
            "os": os,
            "torch": torch,
            "is_npu": lambda: npu,
            "raw_spec_algorithm": args.speculative_algorithm or "",
            "Backend": SimpleNamespace(BREAKABLE="breakable", DISABLED="disabled"),
            "logger": logging.getLogger(__name__),
        },
    )


def test_vl_preserves_supported_prefill_graph_and_mixed_chunk(monkeypatch):
    args = _server_args()
    _startup_policy()(args)
    _apply_shared_welm_prefill_policy(args, monkeypatch)
    assert args.cuda_graph_config.prefill.backend == "breakable"
    assert args.enable_mixed_chunk


@pytest.mark.parametrize(
    "changes",
    [
        {"enable_dp_attention": True},
        {"attn_cp_size": 2},
        {"dcp_size": 2},
        {"quantization": "modelslim"},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"enable_lora": True},
    ],
)
def test_unsupported_shared_prefill_layout_falls_back_to_eager(monkeypatch, changes):
    args = _server_args(**changes)
    _apply_shared_welm_prefill_policy(args, monkeypatch)
    assert args.cuda_graph_config.prefill.backend == "disabled"
    assert not args.enable_mixed_chunk


@pytest.mark.parametrize("native_flash,npu", [(False, True), (True, False)])
def test_graph_and_mixed_require_native_npu_flash(monkeypatch, native_flash, npu):
    args = _server_args()
    _apply_shared_welm_prefill_policy(args, monkeypatch, native_flash, npu)
    assert args.cuda_graph_config.prefill.backend == "disabled"
    assert not args.enable_mixed_chunk
