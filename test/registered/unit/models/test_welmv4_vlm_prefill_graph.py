"""CPU contracts for VL eager embeddings followed by the real graph runner.

Run production wrapper/OE/cache/runner methods. Only device capture and the
large transformer layers are replaced: these tests do not validate NPU kernels.
"""

import ast
import copy
import importlib.util
import logging
import runpy
import sys
from bisect import bisect_left
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from test_welmv4_vlm_backbone import _load_forward
from test_welmv4_vlm_vision import (
    _assert_retained_cpu_pixels,
    _cache_test_image,
    _install_real_oe,
    _make_chunk_cache_wrapper,
)

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SRT = ROOT / "python/sglang/srt"
RUNNER = SRT / "model_executor/runner/prefill_cuda_graph_runner.py"


def _load_method(path, cls_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if getattr(node, "name", None) == cls_name)
    method = next(
        node for node in cls.body if getattr(node, "name", None) == method_name
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], method],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


def _adapter_type():
    path = SRT / "model_executor/runner/welm_prefill_graph.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    shape_path = SRT / "model_executor/runner/shape_key.py"
    spec = importlib.util.spec_from_file_location(
        "_vl_graph_test_shape_key", shape_path
    )
    shape = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = shape
    spec.loader.exec_module(shape)
    namespace = dict(
        torch=torch,
        copy=copy,
        bisect_left=bisect_left,
        logger=logging.getLogger(__name__),
        ShapeKey=shape.ShapeKey,
        ForwardMode=SimpleNamespace(EXTEND=1, MIXED=3),
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["WelmPrefillGraphAdapter"]


Adapter = _adapter_type()


@pytest.mark.parametrize("vl", [False, True])
def test_adapter_replay_preserves_text_hashing_and_skips_vl_placeholders(vl):
    adapter = Adapter.__new__(Adapter)
    adapter.eager_input_embeddings = vl
    adapter.capture_phase = "prompt"
    adapter.prune = False
    adapter.pad_mirror = False
    adapter.oe_ids = torch.full((4, 8), 9, dtype=torch.int32)
    adapter.valid_rows = torch.ones(8, dtype=torch.bool)
    adapter.local_valid_rows = torch.ones(1, dtype=torch.int32)
    adapter.full_write_locs = torch.full((8,), 900, dtype=torch.int64)
    adapter.swa_write_locs = torch.full((8,), 800, dtype=torch.int64)
    adapter.rope_tiles = {}
    hasher = Mock(side_effect=lambda ids, batch: ids.unsqueeze(0).expand(4, -1))
    adapter.model = SimpleNamespace(
        oe_grams=[2, 2, 3, 3], _compute_oe_hashed_ids=hasher
    )
    adapter.backend = SimpleNamespace(
        use_sliding_window_kv_pool=True,
        token_to_kv_pool=SimpleNamespace(
            translate_loc_from_full_to_swa=lambda ids: ids + 100
        ),
        prepare_welm_prefill_graph_metadata=Mock(),
    )
    adapter.flash_inputs = object()
    adapter.flash_metadata = {adapter.key(8): object()}
    # Pinned staging and Flash hardware planning are separate upstream tests.
    adapter._prepare_flash_offsets = Mock()
    adapter._prepare_tiles = Mock()
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (adapter.oe_ids, adapter.valid_rows, adapter.full_write_locs)
    )
    for real_rows in (7, 2, 6):
        ids = torch.arange(real_rows) + (100001 if vl else 1)
        live = SimpleNamespace(
            batch_size=1,
            extend_seq_lens_cpu=[real_rows],
            input_ids=ids,
            ngram_embedding_info=object(),
            global_num_tokens_cpu=None,
            out_cache_loc=torch.arange(real_rows) + 20,
        )
        static = SimpleNamespace(
            num_token_non_padded=None, positions=torch.full((8,), 55)
        )
        adapter.prepare_replay(live, static, 8)
        assert static.ngram_embedding_info is live.ngram_embedding_info
        if vl:
            assert static.welm_prefill_oe_ids is None
        else:
            assert static.welm_prefill_oe_ids.data_ptr() == adapter.oe_ids.data_ptr()
        assert adapter.valid_rows.tolist() == [True] * real_rows + [False] * (
            8 - real_rows
        )
        assert adapter.full_write_locs[real_rows:].tolist() == [-1] * (8 - real_rows)
        assert adapter.swa_write_locs[real_rows:].tolist() == [-1] * (8 - real_rows)
        assert static.positions[real_rows:].tolist() == [0] * (8 - real_rows)
        if not vl:
            torch.testing.assert_close(
                adapter.oe_ids[:, :real_rows], ids.expand(4, -1).int()
            )
        assert addresses == tuple(
            tensor.data_ptr()
            for tensor in (adapter.oe_ids, adapter.valid_rows, adapter.full_write_locs)
        )
    assert hasher.call_count == (0 if vl else 3)


class _TinyBackbone(nn.Module):
    # Real model entry/OE/padding logic; no 80B layers or device runtime imports.
    forward = _load_forward("Qwen2MoeModel")

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.oe_grams = model.oe_grams
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.start_layer = self.end_layer = self.scale_seq_times = 0
        self.layers_to_capture = []
        self.norm = lambda values: (values, None)
        self._restore_npu_prefill_deepep_output_layout = lambda values, aux, batch: (
            values,
            aux,
        )
        _install_real_oe(self)


def _runner(model, adapter, embedding_buffer, wrapper=None):
    slot = SimpleNamespace(slice_for=lambda bs, rows: embedding_buffer[:rows])
    runner = SimpleNamespace(
        welm_adapter=adapter,
        layer_model=model,
        _is_full_backend=False,
        _input_embeds_arg_idx=3,
        buffer_registry=SimpleNamespace(
            has_slot=lambda name: name == "input_embeds", get_slot=lambda name: slot
        ),
        _uses_eager_prefill_tail=lambda: True,
        _get_layer_model_positions=lambda batch: batch.positions,
        _prefill_forward_context=lambda *args, **kwargs: nullcontext(),
        model_runner=SimpleNamespace(model=wrapper),
    )
    run_forward = _load_method(
        RUNNER,
        "PrefillCudaGraphRunner",
        "_run_forward",
        dict(
            torch=torch,
            set_dp_buffer_len=lambda *args: None,
            set_is_extend_in_batch=lambda value: None,
        ),
    )
    runner._run_forward = MethodType(run_forward, runner)
    execute = _load_method(
        RUNNER,
        "PrefillCudaGraphRunner",
        "_execute_body_capture",
        dict(_slice_output_rows=lambda rows, n: rows[:n]),
    )
    runner._execute_body_capture = MethodType(execute, runner)
    return runner


def _batch(tokens, lengths, *, items=None, prefixes=None, capacity=8, table=None):
    items = items or [None] * len(lengths)
    prefixes = prefixes or [0] * len(lengths)
    ids = torch.zeros(capacity, dtype=torch.long)
    ids[: len(tokens)] = torch.tensor(tokens)
    positions = torch.zeros(capacity, dtype=torch.long)
    positions[: len(tokens)] = torch.cat(
        [
            torch.arange(prefix, prefix + length)
            for prefix, length in zip(prefixes, lengths)
        ]
    )
    starts = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)[:-1].tolist()))
    if table is None:
        table = torch.zeros((len(lengths), 30), dtype=torch.int32)
        offset = 0
        for row, (prefix, length) in enumerate(zip(prefixes, lengths)):
            logical = ids[offset : offset + length].clone()
            logical[logical >= 100000] = 77
            table[row, prefix : prefix + length] = logical
            offset += length
    batch = SimpleNamespace(
        input_ids=ids,
        positions=positions,
        input_embeds=None,
        mm_inputs=[
            SimpleNamespace(mm_items=value) if value else None for value in items
        ],
        ngram_embedding_info=SimpleNamespace(
            token_table=table,
            req_lens=torch.tensor(lengths, dtype=torch.int32),
            column_starts=torch.tensor(prefixes, dtype=torch.int32),
        ),
        batch_size=len(lengths),
        req_pool_indices=torch.arange(len(lengths)),
        num_token_non_padded_cpu=len(tokens),
        extend_start_loc=starts,
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens_cpu=lengths,
        forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_target_verify=lambda: False
        ),
        welm_prefill_graph=None,
        welm_prefill_oe_ids=None,
        welm_prefill_graph_phase="prompt",
        can_run_tbo=False,
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        global_dp_buffer_len=None,
        global_num_tokens_cpu=None,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda: False),
    )
    batch.contains_image_inputs = lambda: any(
        mm is not None and mm.mm_items for mm in (batch.mm_inputs or [])
    )
    return batch


def test_capture_skips_oe_only_for_vl_and_zeros_nan_padding():
    wrapper, _, _, _, _ = _make_chunk_cache_wrapper()
    model = _TinyBackbone(wrapper.model)
    embeddings = torch.full((8, 4), float("nan"))
    embeddings[:3] = 123
    adapter = SimpleNamespace(eager_input_embeddings=True, prune=False)
    runner = _runner(model, adapter, embeddings)
    batch = _batch([1, 2, 3], [3])
    batch.input_embeds = embeddings
    batch.welm_prefill_graph = adapter
    batch.welm_prefill_token_mask = torch.arange(8) < 3
    actual = runner._run_forward(batch, 8)
    model._compute_oe_embedding.assert_not_called()
    torch.testing.assert_close(actual[:3], embeddings[:3])
    assert torch.equal(actual[3:], torch.zeros(5, 4))
    # The text adapter still records OE in its body using prepared IDs.
    adapter.eager_input_embeddings = False
    batch.welm_prefill_oe_ids = torch.zeros(4, 8, dtype=torch.int32)
    runner._run_forward(batch, 8)
    model._compute_oe_embedding.assert_called_once()


def test_real_runner_replays_images_text_and_mixed_rows_without_double_oe(monkeypatch):
    wrapper, encoder, _, _, _ = _make_chunk_cache_wrapper()
    model = _TinyBackbone(wrapper.model)
    wrapper.model = model
    wrapper.pp_group = SimpleNamespace(is_last_rank=False)
    # Execute the actual top-level text forward below the real VL wrapper.
    monkeypatch.setattr(
        type(wrapper).__mro__[1], "forward", _load_forward("WeLMV4MoeForCausalLM")
    )
    static_embeddings = torch.full((8, 4), float("nan"))
    adapter = SimpleNamespace(
        eager_input_embeddings=True, prune=False, pad_mirror=False
    )
    runner = _runner(model, adapter, static_embeddings, wrapper)
    captured_forward = model.forward
    adapter.replay = lambda key, batch, **kwargs: captured_forward(
        batch.input_ids,
        batch.positions,
        batch,
        batch.input_embeds,
        skip_oe_fusion=True,
    )
    first = _cache_test_image(100001, 1, 3, 100)
    # Image, shorter pure text, prefix inside the same image plus a one-token
    # decode row, then image again: one graph slot must not retain old values.
    scenarios = [
        (
            [1, first.pad_value, first.pad_value, first.pad_value, 2],
            [5],
            [[first]],
            [0],
        ),
        ([4, 5], [2], [None], [0]),
        ([first.pad_value, first.pad_value, 2, 9], [3, 1], [[first], None], [2, 6]),
        (
            [1, first.pad_value, first.pad_value, first.pad_value, 2],
            [5],
            [[first]],
            [0],
        ),
    ]
    for tokens, lengths, items, prefixes in scenarios:
        history = torch.zeros((len(lengths), 30), dtype=torch.int32)
        if items[0]:
            history[0, :5] = torch.tensor([1, 77, 77, 77, 2])
        else:
            history[0, :2] = torch.tensor([4, 5])
        if len(lengths) > 1:
            history[1, :7] = torch.tensor([3, 4, 5, 6, 7, 8, 9])
        batch = _batch(tokens, lengths, items=items, prefixes=prefixes, table=history)
        # Reference uses the same eager wrapper and real OE, without graphs.
        reference = wrapper.forward(batch.input_ids, batch.positions, batch)
        batch = _batch(tokens, lengths, items=items, prefixes=prefixes, table=history)
        batch.input_embeds = static_embeddings
        batch.welm_prefill_graph = adapter
        batch.welm_prefill_token_mask = torch.arange(8) < len(tokens)
        model._compute_oe_embedding.reset_mock()
        actual = runner._execute_body_capture(batch, batch, 8, len(tokens), object())
        assert model._compute_oe_embedding.call_count == 1
        torch.testing.assert_close(
            actual[: len(tokens)], reference[: len(tokens)], atol=0, rtol=0
        )
        assert torch.equal(actual[len(tokens) :], torch.zeros(8 - len(tokens), 4))
        assert batch.input_embeds.data_ptr() == static_embeddings.data_ptr()
        assert model.forward == captured_forward  # monkey-patch always restored
        _assert_retained_cpu_pixels(first)
    assert len(encoder.calls) == 1  # Actual embedding cache reused across chunks.


def test_mirror_capacity_output_is_trimmed_before_outer_model_tail():
    seen = []
    model = SimpleNamespace(forward=lambda *args: None)
    original = model.forward
    adapter = SimpleNamespace(pad_mirror=True)
    adapter.replay = lambda key, batch, **kwargs: torch.arange(16).reshape(4, 4)
    outer = SimpleNamespace(
        forward=lambda ids, pos, batch, **kwargs: seen.append(
            model.forward(ids, pos, batch)
        )
    )
    runner = _runner(model, adapter, torch.zeros(8, 4), outer)
    batch = _batch([1, 2, 3, 4, 5], [3, 1, 1])
    runner._execute_body_capture(batch, batch, 8, 5, object())
    assert seen[0].shape == (3, 4)  # Breal=3, Bcap=4, Treal=5 are distinct.
    assert model.forward is original


@pytest.mark.parametrize("welm,allowed", [(False, False), (True, True)])
def test_large_padding_relaxation_is_scoped_to_welm(welm, allowed):
    namespace = dict(
        Backend=SimpleNamespace(BREAKABLE="breakable"),
        _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR=2,
        _MAX_WELM_PREFILL_GRAPH_PADDING_FACTOR=64,
    )
    can_replay = _load_method(
        RUNNER, "PrefillCudaGraphRunner", "can_replay_locally", namespace
    )
    runner = SimpleNamespace(
        _is_full_backend=False,
        prefill_backend_name="breakable",
        has_mha_companion_layers=False,
        max_num_tokens=16,
        welm_adapter=object() if welm else None,
        _pad_to_bucket=lambda tokens, sizes: 16,
        capture_num_tokens=[16],
    )
    assert (
        can_replay(
            runner,
            batch_size=1,
            num_tokens=5,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=[0],
            is_target_verify=False,
            capture_hidden_mode=None,
            return_logprob=False,
        )
        is allowed
    )


@pytest.mark.parametrize(
    "megamoe,sync,dump,error",
    [
        (True, True, False, "SYNC_AFTER_OP"),
        (False, False, True, "MOE_STAGE_DUMP"),
        (True, False, False, None),
        (False, True, False, None),
    ],
)
def test_graph_rejects_host_diagnostics_before_allocating(
    monkeypatch, megamoe, sync, dump, error
):
    flags = SimpleNamespace(
        WELM_NPU_USE_MEGAMOE=SimpleNamespace(get=lambda: megamoe),
        SGLANG_NPU_MEGAMOE_SYNC_AFTER_OP=SimpleNamespace(get=lambda: sync),
        SGLANG_NPU_WELMV4_MOE_STAGE_DUMP=SimpleNamespace(get=lambda: dump),
    )
    monkeypatch.setitem(
        Adapter._validate_capture_diagnostics.__globals__, "envs", flags
    )
    if error:
        with pytest.raises(ValueError, match=error):
            Adapter(SimpleNamespace())  # No model/buffers touched before rejection.
    else:
        Adapter._validate_capture_diagnostics()
