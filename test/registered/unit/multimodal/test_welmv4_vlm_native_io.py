"""Exercise Actor's API and nested processor configuration on CPU."""

import ast
import copy
import logging
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Optional, Union
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _definitions(path, names, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _video_part(data):
    names = (
        "PreSampledVideoFrame",
        "ChatCompletionMessageContentVideoURL",
        "ChatCompletionMessageContentVideoPart",
    )
    namespace = _definitions(
        SRT / "entrypoints/openai/protocol.py",
        names,
        dict(
            BaseModel=BaseModel,
            ConfigDict=ConfigDict,
            Optional=Optional,
            Union=Union,
            List=List,
            Literal=Literal,
        ),
    )
    return namespace[names[-1]].model_validate(data).model_dump()


def _render_video(part):
    namespace = _definitions(
        SRT / "parser/jinja_template_utils.py",
        {"process_content_for_template_format"},
        {"ImageData": SimpleNamespace, "logger": logging.getLogger(__name__)},
    )
    videos = []
    message = namespace["process_content_for_template_format"](
        {"role": "user", "content": [part]}, "openai", [], videos, [], []
    )
    return message, videos


def test_video_api_retains_frame_order_timestamps_and_resize_controls():
    frames = [
        {
            "image": "data:image/png;base64,first",
            "timestamp": "0.000s",
            "resized_height": 32,
            "resized_width": 64,
        },
        {"image": "data:image/png;base64,second", "timestamp": "1.500s"},
    ]
    part = _video_part(
        {
            "type": "video_url",
            "video_url": {
                "url": frames,
                "fps": 2,
                "resized_height": 64,
                "resized_width": 96,
                "max_pixels": 6144,
            },
            "fps": 1,
            "num_frames": 2,
        }
    )
    message, videos = _render_video(part)
    assert message["content"] == [{"type": "video"}]
    assert videos == [
        {
            "url": frames,
            "fps": 1.0,
            "num_frames": 2,
            "resized_height": 64,
            "resized_width": 96,
            "max_pixels": 6144,
        }
    ]


@pytest.mark.parametrize("timestamp", [0, 1, 1.5, "1.500s"])
def test_video_api_preserves_native_timestamp_types(timestamp):
    part = _video_part(
        {
            "type": "video_url",
            "video_url": {"url": [{"image": "frame.png", "timestamp": timestamp}]},
        }
    )
    _, videos = _render_video(part)
    actual = videos[0]["url"][0]["timestamp"]
    assert type(actual) is type(timestamp)
    assert actual == timestamp


@pytest.mark.parametrize("max_dynamic_patch", [None, 8])
def test_existing_video_url_contract_is_preserved(max_dynamic_patch):
    part = _video_part(
        {
            "type": "video_url",
            "video_url": {"url": "clip.mp4", "max_dynamic_patch": max_dynamic_patch},
        }
    )
    _, videos = _render_video(part)
    assert videos == (
        ["clip.mp4"]
        if max_dynamic_patch is None
        else [{"url": "clip.mp4", "max_dynamic_patch": 8}]
    )


def _loader(monkeypatch, settings):
    from transformers.processing_utils import ProcessorMixin

    read = Mock(return_value=(settings, {}))
    monkeypatch.setattr(ProcessorMixin, "get_processor_dict", read)
    image_class = SimpleNamespace(from_dict=Mock(side_effect=copy.deepcopy))
    video_class = SimpleNamespace(from_dict=Mock(side_effect=copy.deepcopy))
    classes = {
        "processing.Native": SimpleNamespace,
        "image.Native": image_class,
        "video.Native": video_class,
    }
    resolve = Mock(side_effect=lambda reference, *a, **kw: classes[reference])
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", resolve
    )
    tokenizer = SimpleNamespace(chat_template="checkpoint chat template")
    tokenizer_class = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    namespace = _definitions(
        SRT / "utils/hf_transformers/processor.py",
        {"_build_welm_vlm_nested_processor"},
        {"AutoTokenizer": tokenizer_class},
    )
    return namespace["_build_welm_vlm_nested_processor"], read, resolve, tokenizer_class


@pytest.fixture
def nested_settings():
    return {
        "auto_map": {"AutoProcessor": "processing.Native"},
        "image_processor": {
            "auto_map": {"AutoImageProcessor": "image.Native"},
            "patch_size": 16,
            "temporal_patch_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "size": {"shortest_edge": 1024, "longest_edge": 16777216},
        },
        "video_processor": {
            "auto_map": {"AutoVideoProcessor": "video.Native"},
            "fps": 2,
            "max_frames": 768,
        },
    }


def test_nested_loader_keeps_checkpoint_components_and_access_options(
    monkeypatch, nested_settings
):
    original = copy.deepcopy(nested_settings)
    load, read, resolve, tokenizer_class = _loader(monkeypatch, nested_settings)
    config = SimpleNamespace(model_type="welmv4_vlm", image_token="<|welm_image_pad|>")
    processor = load("model", config, True, "revision", local_files_only=True)
    assert processor.image_processor == original["image_processor"]
    assert processor.video_processor == original["video_processor"]
    assert nested_settings == original
    assert processor.chat_template == "checkpoint chat template"
    for call in resolve.call_args_list:
        assert call.kwargs == {"revision": "revision", "local_files_only": True}
    tokenizer_class.from_pretrained.assert_called_once_with(
        "model", trust_remote_code=True, revision="revision", local_files_only=True
    )


@pytest.mark.parametrize(
    "model_type,image_token,trust",
    [
        ("welmv4_vlm", "<|welm_image_pad|>", False),
        ("welmv4_vlm", "<|image_pad|>", True),
        ("welm_moonvit_vl", "<|welm_image_pad|>", True),
    ],
)
def test_nested_loader_does_not_execute_untrusted_or_other_models(
    monkeypatch, nested_settings, model_type, image_token, trust
):
    load, read, resolve, _ = _loader(monkeypatch, nested_settings)
    config = SimpleNamespace(model_type=model_type, image_token=image_token)
    assert load("model", config, trust, None) is None
    read.assert_not_called()
    resolve.assert_not_called()


def test_nested_loader_rejects_missing_components(monkeypatch, nested_settings):
    del nested_settings["video_processor"]
    load, _, _, _ = _loader(monkeypatch, nested_settings)
    config = SimpleNamespace(model_type="welmv4_vlm", image_token="<|welm_image_pad|>")
    with pytest.raises(ValueError, match="missing video_processor"):
        load("model", config, True, None)


def test_nested_loader_honors_explicit_template(monkeypatch, nested_settings):
    load, _, _, _ = _loader(monkeypatch, nested_settings)
    config = SimpleNamespace(model_type="welmv4_vlm", image_token="<|welm_image_pad|>")
    processor = load("model", config, True, None, chat_template="explicit template")
    assert processor.chat_template == "explicit template"


def test_nested_loader_rejects_missing_component_class(monkeypatch, nested_settings):
    nested_settings["image_processor"].pop("auto_map")
    load, _, _, _ = _loader(monkeypatch, nested_settings)
    config = SimpleNamespace(model_type="welmv4_vlm", image_token="<|welm_image_pad|>")
    with pytest.raises(ValueError, match="missing AutoImageProcessor auto_map"):
        load("model", config, True, None)


@pytest.mark.parametrize(
    "videos",
    [
        "one.mp4",
        ["one.mp4"],
        ["one.mp4", "two.mp4"],
        [["one.mp4", "two.mp4"]],
        [{"url": [{"image": "frame.png"}]}, "two.mp4"],
    ],
)
def test_parallel_samples_keep_all_videos_of_a_single_prompt(videos):
    path = SRT / "managers/io_struct.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerateReqInput"
    )
    nodes = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_handle_parallel_sampling", "_normalize_video_data")
    ]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    request = SimpleNamespace(
        sampling_params={"n": 2},
        is_single=True,
        batch_size=1,
        text=None,
        input_ids=[1, 2],
        input_embeds=None,
        image_data=None,
        video_data=copy.deepcopy(videos),
    )
    scope["_handle_parallel_sampling"](request)
    scope["_normalize_video_data"](request, 2)
    expected = videos
    if isinstance(videos, list) and videos and isinstance(videos[0], list):
        expected = videos[0]
    assert request.video_data == [expected, expected]


def test_parallel_video_batch_retains_prompt_association():
    path = SRT / "managers/io_struct.py"
    node = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_normalize_video_data"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    videos = [["a.mp4", "b.mp4"], [{"url": "c.mp4", "fps": 1}]]
    request = SimpleNamespace(video_data=copy.deepcopy(videos), parallel_sample_num=2)
    scope["_normalize_video_data"](request, 4)
    assert request.video_data == videos * 2


@pytest.mark.parametrize(
    "images",
    ["one.png", ["one.png"], ["one.png", "two.png"], [["one.png", "two.png"]]],
)
def test_parallel_samples_keep_images_and_videos_in_the_same_prompt(images):
    path = SRT / "managers/io_struct.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerateReqInput"
    )
    nodes = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in (
            "_handle_parallel_sampling",
            "_normalize_image_data",
            "_normalize_video_data",
        )
    ]
    scope = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    videos = [{"url": [{"image": "frame.png", "timestamp": 1.5}]}, "two.mp4"]
    request = SimpleNamespace(
        sampling_params={"n": 2},
        is_single=True,
        batch_size=1,
        text="Inspect the images and videos",
        input_ids=None,
        input_embeds=None,
        image_data=copy.deepcopy(images),
        video_data=copy.deepcopy(videos),
    )
    scope["_handle_parallel_sampling"](request)
    scope["_normalize_image_data"](request, 2)
    scope["_normalize_video_data"](request, 2)
    expected = [images] if isinstance(images, str) else images
    if isinstance(expected[0], list):
        expected = expected[0]
    assert request.image_data == [expected, expected]
    assert request.video_data == [videos, videos]
