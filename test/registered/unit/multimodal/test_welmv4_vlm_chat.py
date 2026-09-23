"""Check actual chat conversion methods without importing model workers.

Request/tokenizer boundaries are faked. Full runtime import coverage lives in
unit/entrypoints/openai/test_serving_chat.py.
"""

import ast
import copy
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _chat_methods():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/entrypoints/openai/serving_chat.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingChat"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_convert_to_internal_request", "_apply_jinja_template")
    ]
    namespace = {
        "copy": copy,
        "GenerateReqInput": lambda **kw: SimpleNamespace(
            **({"text": None, "input_ids": None} | kw)
        ),
        "MessageProcessingResult": lambda **kw: SimpleNamespace(**kw),
        "_extract_max_dynamic_patch": lambda req: (None, None),
        "envs": SimpleNamespace(
            SGLANG_DEFAULT_THINKING=SimpleNamespace(get=lambda: False)
        ),
        "ThinkingMode": SimpleNamespace(THINKING="thinking", CHAT="chat"),
        "normalize_assistant_tool_call_arguments": lambda *a, **kw: None,
        "normalize_tool_content": lambda role, content: content,
        "process_content_for_template_format": lambda msg, *args, **kw: msg,
    }
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0]] + methods,
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("prompt_input_type", ["token_ids", "text"])
def test_template_and_internal_request_preserve_processor_input_contract(
    prompt_input_type,
):
    methods = _chat_methods()
    request = Mock(
        chat_template_kwargs={},
        stream=False,
        input_ids=None,
        top_logprobs=0,
        return_prompt_token_ids=False,
        return_token_ids=False,
        reasoning_effort=None,
        stop=None,
    )
    request.messages = [
        SimpleNamespace(model_dump=lambda: {"role": "user", "content": "inspect"})
    ]
    ids = [1, 154752, 2]
    tokenizer = Mock()
    tokenizer.apply_chat_template.return_value = "native prompt"
    tokenizer.encode.return_value = ids
    tokenizer.decode.return_value = "decoded prompt"
    serving = SimpleNamespace(
        is_gpt_oss=False,
        tokenizer_manager=SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=True), tokenizer=tokenizer
        ),
        template_manager=SimpleNamespace(
            jinja_template_content_format="openai", reasoning_config=None
        ),
        default_sampling_params={},
        chat_encoding_spec=None,
        mm_prompt_input_type=prompt_input_type,
        extract_custom_labels=lambda req: None,
        extract_routed_dp_rank_from_header=lambda *args: None,
        _resolve_lora_path=lambda *args: None,
        _compute_extra_key=lambda req: None,
        extract_routing_key=lambda req: None,
        _encode_messages=lambda *args, **kw: None,
        _handle_last_assistant_message=lambda messages, req: (messages, None),
        _tokenizer_auto_adds_specials=True,
    )
    result = methods["_apply_jinja_template"](serving, request, None, True)
    assert result.prompt_ids is ids
    serving._process_messages = lambda *args: SimpleNamespace(
        **(vars(result) | {"tool_call_constraint": None, "require_reasoning": False})
    )
    adapted, returned_request = methods["_convert_to_internal_request"](
        serving, request
    )
    assert returned_request is request
    if prompt_input_type == "token_ids":
        tokenizer.decode.assert_not_called()
        assert adapted.input_ids is ids
        assert adapted.text is None
    else:
        tokenizer.decode.assert_called_once_with(ids)
        assert adapted.text == "decoded prompt"
        assert adapted.input_ids is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
