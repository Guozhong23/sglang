"""Exercise startup policy and checkpoint checks without importing NPU workers."""

import ast
import importlib.util
import json
import logging
import os
import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
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


@pytest.mark.parametrize(
    "architecture,fixture",
    [
        (ARCH, FIXTURE),
        (
            "WeLMMoonViTVLForConditionalGeneration",
            FIXTURE.with_name("welm_moonvit_vl_config.json"),
        ),
    ],
)
def test_real_checkpoint_multimodal_sinks_and_48_layer_windows(architecture, fixture):
    raw = json.loads(fixture.read_text())
    text = SimpleNamespace(**raw["text_config"])
    helpers = _model_helpers()
    assert helpers["is_multimodal_model"]([architecture])
    assert helpers["is_multimodal_breakable_cuda_graph_supported"]([architecture])
    assert helpers["is_hybrid_swa_model"]([architecture], text)
    model = SimpleNamespace(hf_config=SimpleNamespace(**raw), hf_text_config=text)
    assert helpers["_detect_attention_sinks"](model)
    swa, full = helpers["get_hybrid_layer_ids"]([architecture], text, context_len=8192)
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
        disaggregation_transfer_backend="ascend",
        disaggregation_decode_enable_radix_cache=False,
        enable_hierarchical_cache=False,
        nnodes=1,
        dp_size=1,
        enable_pdmux=False,
        enable_two_batch_overlap=False,
        elastic_ep_backend=None,
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
    args.get_model_config = lambda: SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(scale_seq_times=0)
    )
    return args


def _startup_policy(
    megamoe=False, vit_graph=False, npu=True, random_hash=False, skip_all_gather=False
):
    methods = _definitions(
        SRT / "server_args.py",
        {"_handle_welm_vlm_adjustments", "_validate_welm_vlm_attention_dp_config"},
        {
            "Backend": SimpleNamespace(DISABLED="disabled"),
            "is_npu": lambda: npu,
            "envs": SimpleNamespace(
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe),
                SGLANG_VIT_ENABLE_CUDA_GRAPH=SimpleNamespace(get=lambda: vit_graph),
                SGLANG_MM_SKIP_COMPUTE_HASH=SimpleNamespace(get=lambda: random_hash),
                SGLANG_SCHEDULER_SKIP_ALL_GATHER=SimpleNamespace(
                    get=lambda: skip_all_gather
                ),
            ),
            "logger": logging.getLogger(__name__),
        },
    )

    def apply(args):
        args._validate_welm_vlm_attention_dp_config = MethodType(
            methods["_validate_welm_vlm_attention_dp_config"], args
        )
        return methods["_handle_welm_vlm_adjustments"](args)

    return apply


@pytest.mark.parametrize("dp", [False, True])
def test_startup_keeps_requested_decode_graph_chunk_and_cache_settings(dp):
    args = _server_args(mm_enable_dp_encoder=dp)
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
        {"encoder_only": True},
        {"language_only": True},
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


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("vision_dp", [False, True])
def test_ascend_pd_is_accepted_without_changing_multimodal_or_vision_mode(
    mode, vision_dp
):
    args = _server_args(disaggregation_mode=mode, mm_enable_dp_encoder=vision_dp)
    _startup_policy()(args)
    assert args.enable_multimodal
    assert args.mm_enable_dp_encoder == vision_dp
    assert args.disaggregation_mode == mode


@pytest.mark.parametrize(
    "changes",
    [
        {"disaggregation_transfer_backend": "mooncake"},
        {"disaggregation_transfer_backend": "fake"},
        {"nnodes": 2},
        {"dp_size": 2},
        {"attn_cp_size": 2},
        {"dcp_size": 2},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"enable_lora": True},
        {"enable_hierarchical_cache": True},
        {"disaggregation_decode_enable_radix_cache": True},
    ],
)
def test_pd_rejects_unimplemented_layouts(changes):
    with pytest.raises(ValueError, match="WeLM-VL PD"):
        _startup_policy()(_server_args(disaggregation_mode="prefill", **changes))


@pytest.mark.parametrize("options", [{"npu": False}, {"random_hash": True}])
def test_pd_requires_npu_and_deterministic_image_hashes(options):
    with pytest.raises(ValueError, match="WeLM-VL PD"):
        _startup_policy(**options)(_server_args(disaggregation_mode="decode"))


def test_startup_defers_ep_validation_until_deepep_resolution():
    # DeepEP changes EP=1 to EP=TP later in ServerArgs initialization.
    args = _server_args(moe_a2a_backend="deepep")
    _startup_policy(megamoe=True)(args)
    assert args.moe_a2a_backend == "deepep"


def _parallel_policy(megamoe=False):
    methods = _definitions(
        SRT / "server_args.py",
        {
            "_validate_welm_vlm_parallel_config",
            "_validate_welm_vlm_attention_dp_config",
        },
        {
            "resolved_view": lambda args: args,
            "Backend": SimpleNamespace(DISABLED="disabled"),
            "is_npu": lambda: True,
            "envs": SimpleNamespace(
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe),
                SGLANG_SCHEDULER_SKIP_ALL_GATHER=SimpleNamespace(get=lambda: False),
            ),
        },
    )

    def apply(args):
        args._validate_welm_vlm_attention_dp_config = MethodType(
            methods["_validate_welm_vlm_attention_dp_config"], args
        )
        return methods["_validate_welm_vlm_parallel_config"](args)

    return apply


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


def _apply_shared_welm_prefill_policy(
    args, monkeypatch, native_flash=True, npu=True, model_arch=ARCH
):
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
            "model_arch": model_arch,
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


def test_text_modelslim_prefill_graph_remains_supported(monkeypatch):
    args = _server_args(quantization="modelslim")
    _apply_shared_welm_prefill_policy(
        args, monkeypatch, model_arch="WeLMV4MoeForCausalLM"
    )
    assert args.cuda_graph_config.prefill.backend == "breakable"


@pytest.mark.parametrize("native_flash,npu", [(False, True), (True, False)])
def test_graph_and_mixed_require_native_npu_flash(monkeypatch, native_flash, npu):
    args = _server_args()
    _apply_shared_welm_prefill_policy(args, monkeypatch, native_flash, npu)
    assert args.cuda_graph_config.prefill.backend == "disabled"
    assert not args.enable_mixed_chunk


def _attention_dp_args(**changes):
    values = dict(
        enable_dp_attention=True,
        tp_size=4,
        dp_size=2,
        ep_size=4,
        mm_enable_dp_encoder=True,
        moe_a2a_backend="deepep",
    )
    values.update(changes)
    return _server_args(**values)


@pytest.mark.parametrize("dtype", ["auto", "bfloat16"])
@pytest.mark.parametrize("kv_dtype", ["auto", "bf16", "bfloat16"])
def test_attention_tp2_dp2_ep4_accepts_bf16_and_keeps_eager_optimizations(
    dtype, kv_dtype
):
    args = _attention_dp_args(dtype=dtype, kv_cache_dtype=kv_dtype)
    _startup_policy(megamoe=True)(args)
    _parallel_policy(megamoe=True)(args)
    assert args.enable_dp_attention and args.mm_enable_dp_encoder
    assert args.moe_a2a_backend == "deepep" and args.ep_size == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_size": 8},
        {"dp_size": 4},
        {"nnodes": 2},
        {"attn_cp_size": 2},
        {"dcp_size": 2},
        {"moe_a2a_backend": "none"},
        {"moe_dp_size": 2},
        {"mm_enable_dp_encoder": False},
        {"kv_cache_dtype": "fp8_e4m3"},
        {"disaggregation_mode": "prefill"},
        {"disaggregation_mode": "decode"},
        {"enable_lora": True},
        {"enable_two_batch_overlap": True},
        {"elastic_ep_backend": "mooncake"},
    ],
)
def test_attention_dp_precise_profile_rejects_unimplemented_combinations(changes):
    with pytest.raises(ValueError, match="WeLM-VL attention DP"):
        _startup_policy()(_attention_dp_args(**changes))


def test_attention_dp_requires_npu_and_rejects_scale_seq():
    with pytest.raises(ValueError, match="non-NPU"):
        _startup_policy(npu=False)(_attention_dp_args())
    args = _attention_dp_args()
    args.get_model_config = lambda: SimpleNamespace(
        hf_text_config=SimpleNamespace(scale_seq_times=1)
    )
    with pytest.raises(ValueError, match="scale-seq"):
        _startup_policy()(args)


def test_attention_dp_validates_ep_after_deepep_resolution():
    args = _attention_dp_args(ep_size=1)
    _startup_policy()(args)
    with pytest.raises(ValueError, match="resolved EP=4"):
        _parallel_policy()(args)
    args.ep_size = 4
    _parallel_policy()(args)


def test_attention_dp_keeps_breakable_prefill_but_not_mixed_chunk(monkeypatch):
    args = _attention_dp_args()
    _startup_policy()(args)
    _apply_shared_welm_prefill_policy(args, monkeypatch)
    assert args.cuda_graph_config.prefill.backend == "breakable"
    assert not args.enable_mixed_chunk


def test_attention_dp_requires_scheduler_counts_but_ordinary_tp_is_unchanged():
    with pytest.raises(ValueError, match="SGLANG_SCHEDULER_SKIP_ALL_GATHER"):
        _startup_policy(skip_all_gather=True)(_attention_dp_args())
    _startup_policy(skip_all_gather=True)(_server_args())


@pytest.mark.parametrize("field", ["max_bs", "bs"])
def test_attention_dp_graph_capacity_uses_resolved_per_group_chunk(field):
    args = _attention_dp_args(chunked_prefill_size=8192)
    setattr(
        args.cuda_graph_config.prefill,
        field,
        16512 if field == "max_bs" else [512, 16512],
    )
    # Early validation precedes DP division and must defer this check.
    _startup_policy()(args)
    with pytest.raises(ValueError, match="8192 after DP division"):
        _parallel_policy()(args)
    setattr(
        args.cuda_graph_config.prefill,
        field,
        8192 if field == "max_bs" else [512, 8192],
    )
    _parallel_policy()(args)
    args.cuda_graph_config.prefill.backend = "disabled"
    setattr(
        args.cuda_graph_config.prefill, field, 16512 if field == "max_bs" else [16512]
    )
    _parallel_policy()(args)
