"""Exercise startup policy and checkpoint checks without importing NPU workers."""

import ast
import importlib.util
import json
import logging
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

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
        moe_a2a_backend="none",
        quantization=None,
        dtype="bfloat16",
        enable_multimodal=None,
        chunked_prefill_size=4096,
        disable_radix_cache=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="full"),
            prefill=SimpleNamespace(backend="full"),
        ),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _startup_policy(megamoe=False, vit_graph=False):
    return _definitions(
        SRT / "server_args.py",
        {"_handle_welm_vlm_basic_mode"},
        {
            "Backend": SimpleNamespace(DISABLED="disabled"),
            "envs": SimpleNamespace(
                WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe),
                SGLANG_VIT_ENABLE_CUDA_GRAPH=SimpleNamespace(get=lambda: vit_graph),
            ),
            "logger": logging.getLogger(__name__),
        },
    )["_handle_welm_vlm_basic_mode"]


def test_basic_startup_keeps_multimodal_and_disables_unvalidated_paths():
    args = _server_args()
    _startup_policy()(args)
    assert args.enable_multimodal
    assert args.disable_radix_cache and args.chunked_prefill_size == -1
    assert args.cuda_graph_config.decode.backend == "disabled"
    assert args.cuda_graph_config.prefill.backend == "disabled"


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
        {"ep_size": 4},
        {"moe_a2a_backend": "deepep"},
        {"quantization": "modelslim"},
        {"dtype": "float16"},
        {"enable_multimodal": False},
    ],
)
def test_basic_startup_rejects_unsupported_combinations(changes):
    with pytest.raises(ValueError, match="WeLM-VL"):
        _startup_policy()(_server_args(**changes))


def test_basic_startup_rejects_inherited_megamoe():
    with pytest.raises(ValueError, match="WELM_NPU_USE_MEGAMOE"):
        _startup_policy(megamoe=True)(_server_args())


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
