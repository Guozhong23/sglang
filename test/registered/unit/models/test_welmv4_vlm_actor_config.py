"""CPU regressions for the Actor checkpoint's visual and language contracts.

The fixture is the supplied Actor config.json, not a serving-ready checkpoint.
Hardware kernels, checkpoint weights and MemFabric transport are not exercised.
"""

import ast
import copy
import importlib.util
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import PretrainedConfig

from test_welmv4_vlm_startup import _model_helpers

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
FIXTURES = Path(__file__).with_name("fixtures")
runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
](est_time=5, suite="base-a-test-cpu")


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _module(SRT / "models/welmv4_vlm_utils.py", "actor_vlm_helpers")
PD = _module(SRT / "disaggregation/welm_vlm.py", "actor_pd_helpers")


def _fixture(name="welmv4_vlm_actor_config.json"):
    return json.loads((FIXTURES / name).read_text())


def _hf_config(raw):
    # Base HF config does not implement the checkpoint's remote WeLM subclass.
    # Set model-specific attributes after generic HF initialization, as in the
    # existing WeLM PD tests, to avoid running an unrelated RoPE validator.
    config = PretrainedConfig()
    config.update(copy.deepcopy(raw))
    for field in ("text_config", "vision_config"):
        value = getattr(config, field, None)
        if isinstance(value, dict):
            setattr(config, field, _hf_config(value))
    return config


def _snapshot(config):
    if hasattr(config, "to_dict"):
        return copy.deepcopy(config.to_dict())
    if hasattr(config, "__dict__"):
        return copy.deepcopy(vars(config))
    return copy.deepcopy(config)


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    tree.body = ast.parse("from __future__ import annotations").body + selected
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "factory",
    [
        copy.deepcopy,
        lambda raw: SimpleNamespace(**copy.deepcopy(raw)),
        lambda raw: SimpleNamespace(
            **{
                **copy.deepcopy(raw),
                "vision_config": SimpleNamespace(**raw["vision_config"]),
            }
        ),
        _hf_config,
    ],
)
@pytest.mark.parametrize(
    "fixture,expected",
    [
        ("welmv4_vlm_actor_config.json", True),
        ("welmv4_vlm_config.json", False),
        ("welm_moonvit_vl_config.json", False),
    ],
)
def test_visual_contract_uses_checkpoint_token_across_config_types(
    factory, fixture, expected
):
    raw = _fixture(fixture)
    config = factory(raw)
    before = _snapshot(config)
    assert HELPERS.welmv4_vlm_visual_actor_match(config) is expected
    assert _snapshot(config) == before


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize(
    "fixture", ["welmv4_vlm_actor_config.json", "welmv4_vlm_config.json"]
)
def test_explicit_visual_math_override_wins_over_checkpoint_token(override, fixture):
    config = _fixture(fixture)
    config["vision_config"]["welm_visual_actor_match"] = override
    assert HELPERS.welmv4_vlm_visual_actor_match(config) is override
    assert HELPERS.welmv4_vlm_visual_actor_match(_hf_config(config)) is override


@pytest.mark.parametrize("invalid", ["false", "true", "0", 0, 1, None, [], {}])
def test_visual_math_override_rejects_non_boolean_values(invalid):
    config = _fixture()
    config["vision_config"]["welm_visual_actor_match"] = invalid
    with pytest.raises(ValueError, match="must be a boolean"):
        HELPERS.welmv4_vlm_visual_actor_match(config)


def test_visual_contract_does_not_guess_from_version_stage_or_geometry():
    config = _fixture()
    del config["image_token"]
    assert not HELPERS.welmv4_vlm_visual_actor_match(config)
    config["image_token"] = "<|image_pad|>"
    assert not HELPERS.welmv4_vlm_visual_actor_match(config)


def test_actor_target_mirror_keeps_active_pairs_without_mutating_checkpoint():
    raw = _fixture()
    text = SimpleNamespace(**raw["text_config"], _welmv4_vlm_target_only=True)
    before = copy.deepcopy(vars(text))
    runtime = HELPERS.welmv4_vlm_target_config(text)
    assert text.kv_mirror_layers[:3] == [50, 49, 48]
    assert runtime.kv_mirror_layers == list(range(47, 32, -1))
    assert runtime.kv_mirror_imitated_layers == list(range(1, 16))
    assert runtime.kv_mirror_repeat == [1] * 15
    assert runtime.num_nextn_predict_layers == 3
    assert vars(text) == before
    with pytest.raises(ValueError, match="speculative"):
        HELPERS.welmv4_vlm_target_config(text, "EAGLE")


def test_actor_nested_text_dtype_tie_and_rope_survive_hf_resolution():
    config = _hf_config(_fixture())
    namespace = _definitions(
        SRT / "utils/hf_transformers_patches.py",
        {"normalize_rope_scaling_compat"},
        {},
    )
    namespace.update(PretrainedConfig=PretrainedConfig)
    _definitions(
        SRT / "utils/hf_transformers/common.py",
        {"_patch_text_config", "get_hf_text_config"},
        namespace,
    )
    text = namespace["get_hf_text_config"](config)
    assert text is config.text_config
    assert text.dtype == config.dtype
    assert str(text.dtype) in ("bfloat16", "torch.bfloat16")
    assert config.tie_word_embeddings is True
    assert text.tie_word_embeddings is False
    assert text.rope_parameters["rope_theta"] == 500000.0
    assert text.rope_parameters["factor"] == 8.0
    assert text.rope_parameters["apply_softmax_scale"] is True


@pytest.mark.parametrize("remove_legacy_type", [False, True])
def test_actor_v5_yarn_parameters_select_existing_backbone_rope(remove_legacy_type):
    text = _fixture()["text_config"]
    params = copy.deepcopy(text["rope_parameters"])
    if remove_legacy_type:
        params.pop("type")
    yarn = Mock(return_value=object())
    namespace = _definitions(
        SRT / "models/welmv4.py",
        {"get_rope"},
        {
            "torch": torch,
            "_ROPE_DICT": {},
            "Qwen2MoeYarnScalingRotaryEmbedding": yarn,
        },
    )
    result = namespace["get_rope"](
        text["head_dim"],
        text["qk_rope_head_dim"],
        text["max_position_embeddings"],
        params["rope_theta"],
        rope_scaling=params,
        dtype=torch.bfloat16,
    )
    assert result is yarn.return_value
    assert yarn.call_args.args == (
        256, 64, 32768, 500000.0, True, 8.0, torch.bfloat16
    )
    assert yarn.call_args.kwargs == dict(
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        compress=1.0,
        max_position=262144,
    )


def test_actor_uses_registered_multimodal_graph_sinks_and_target_windows():
    raw = _fixture()
    text = SimpleNamespace(**raw["text_config"])
    model = SimpleNamespace(hf_config=SimpleNamespace(**raw), hf_text_config=text)
    helpers = _model_helpers()
    assert helpers["is_multimodal_model"](raw["architectures"])
    assert helpers["is_multimodal_breakable_cuda_graph_supported"](
        raw["architectures"]
    )
    assert helpers["_detect_attention_sinks"](model)
    windows = helpers["get_welmv4_layerwise_sliding_windows"](
        text, context_len=32768
    )
    assert len(windows) == 48
    assert set(windows) == {-1, 511}
    swa, full = helpers["get_hybrid_layer_ids"](
        raw["architectures"], text, context_len=32768
    )
    assert len(swa) == 23 and len(full) == 25
    assert sorted(swa + full) == list(range(48))
    assert PD.is_welm_vlm(model.hf_config)


@pytest.mark.parametrize("change", ["visual_math", "rope", "image_token"])
def test_pd_contract_rejects_actor_semantic_mismatch(change):
    raw = _fixture()
    other = copy.deepcopy(raw)
    other["_name_or_path"] = "/another/model/mount"

    def digest(value):
        return PD._digest(PD._semantic_config(value))

    assert digest(raw) == digest(other)
    if change == "visual_math":
        other["vision_config"]["welm_visual_actor_match"] = False
    elif change == "rope":
        other["text_config"]["rope_parameters"]["factor"] = 4.0
    else:
        other["image_token"] = "<|image_pad|>"
    with pytest.raises(ValueError, match="config_digest"):
        PD.validate_welm_vlm_pd_contract(
            {"config_digest": digest(raw)}, {"config_digest": digest(other)}
        )
