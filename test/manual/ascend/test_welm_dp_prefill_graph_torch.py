"""Actual CPU-torch storage tests for DP prefill adapters, without NPU imports.

Uses production AST methods, including Ascend Flash metadata preparation.
Only pinned allocation/NPU availability and model/graph execution are mocked.
This checks tensor aliasing and staging, not device graphs, Flash, or HCCL.
"""

import copy
from contextlib import nullcontext
import dataclasses
import logging
import sys
import types
import unittest
from bisect import bisect_left
from unittest.mock import patch

import torch

from test_welmv4_prefill_graph_contracts import ShapeKey, SRT, load_method, load_nodes


class CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def tensor(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch.tensor(*args, **kwargs)


T = CpuTorch()
MAX_LEN = types.SimpleNamespace(is_max_len=lambda: True)
rows = load_nodes(
    SRT / "model_executor/forward_batch_info.py",
    [
        "WelmDpRowView",
        "WelmDpMoeRowLayout",
        "_welm_slot_rows",
        "_welm_build_row_view",
        "build_welm_dp_moe_row_layout",
    ],
    {
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "torch": T,
        "ceil_align": lambda x, a: ((x + a - 1) // a) * a,
    },
)
base = load_nodes(
    SRT / "model_executor/runner/welm_prefill_graph.py",
    [
        "parse_capture_batch_sizes",
        "capture_request_lengths",
        "padded_flash_cu_seqlens",
        "padded_rope_tiles",
        "WelmPrefillGraphAdapter",
    ],
    {
        "torch": T,
        "copy": copy,
        "ShapeKey": ShapeKey,
        "bisect_left": bisect_left,
        "ForwardMode": types.SimpleNamespace(EXTEND=1, MIXED=3),
        "logger": logging.getLogger(__name__),
    },
)
namespace = load_nodes(
    SRT / "model_executor/runner/welm_dp_prefill_graph.py",
    ["validate_dp_graph_rows", "WelmDpPrefillGraphAdapter"],
    {
        "torch": T,
        "WelmPrefillGraphAdapter": base["WelmPrefillGraphAdapter"],
        "copy": copy,
        "ShapeKey": ShapeKey,
        "logger": logging.getLogger(__name__),
        "ForwardMode": types.SimpleNamespace(EXTEND=1, IDLE=4),
        "nullcontext": nullcontext,
        "DpPaddingMode": types.SimpleNamespace(MAX_LEN=MAX_LEN),
        "build_welm_dp_moe_row_layout": rows["build_welm_dp_moe_row_layout"],
    },
)
Adapter = namespace["WelmDpPrefillGraphAdapter"]
BACKEND = SRT / "hardware_backend/npu/attention/ascend_backend.py"
backend_ns = {
    "torch": T,
    "ForwardMetadata": types.SimpleNamespace,
    "torch_npu": types.SimpleNamespace(npu_scatter_pa_kv_cache=object()),
}
# The class name is a source fact, not an imported runtime dependency.
import ast

backend_class = next(
    node.name
    for node in ast.parse(BACKEND.read_text(encoding="utf8")).body
    if isinstance(node, ast.ClassDef)
    and any(
        isinstance(m, ast.FunctionDef)
        and m.name == "create_welm_prefill_graph_metadata"
        for m in node.body
    )
)


class CpuBackend:
    create_welm_prefill_graph_metadata = load_method(
        BACKEND, backend_class, "create_welm_prefill_graph_metadata", backend_ns
    )
    prepare_welm_prefill_graph_metadata = load_method(
        BACKEND, backend_class, "prepare_welm_prefill_graph_metadata", backend_ns
    )

    def __init__(self):
        self.device = "cpu"
        self.max_context_len = 512
        self.page_size = 16
        self.is_hybrid_swa = True
        self.use_sliding_window_kv_pool = True
        self.req_to_token_pool = types.SimpleNamespace(
            req_to_token=torch.arange(32 * 512).reshape(32, 512)
        )
        self.full_to_swa_index_mapping = torch.arange(32 * 512) + 32
        self.token_to_kv_pool = types.SimpleNamespace(
            translate_loc_from_full_to_swa=lambda loc: loc + 32
        )


def make_adapter(dp_rank):
    obj = Adapter.__new__(Adapter)
    obj.active_mask = 3
    obj.active_masks = (3,)
    obj.dp_size, obj.dp_rank, obj.attn_tp_size, obj.attn_tp_rank = 2, dp_rank, 2, 0
    obj.ep, obj.ep_size, obj.prune, obj.pad_mirror = True, 4, True, True
    obj.max_tokens, obj.max_requests, obj.device = 128, 8, "cpu"
    obj.eager_input_embeddings, obj.mixed_chunk = True, False
    obj.batch_sizes = (1, 2, 4, 8)
    obj.capture_keys, obj.flash_metadata, obj.captured_states = set(), {}, {}
    obj._row_layouts, obj._capture_templates, obj.rope_tiles, obj.mtp_kv = (
        {},
        {},
        {},
        {},
    )
    obj._token_rows, obj._request_rows = (0, 0), (0, 0)
    obj._creating_capture, obj._warned, obj.capture_phase = False, set(), "prompt"
    obj.prompt_real_rows_gpu = torch.zeros(2, dtype=torch.int32)
    obj.request_real_rows_gpu = torch.zeros(2, dtype=torch.int32)
    obj.prompt_local_valid_rows = torch.zeros(1, dtype=torch.int32)
    obj.request_local_valid_rows = torch.zeros(1, dtype=torch.int32)
    obj.local_valid_rows = torch.zeros(1, dtype=torch.int32)
    obj.valid_rows = torch.zeros(128, dtype=torch.bool)
    obj.oe_ids = torch.zeros((0, 128), dtype=torch.int32)
    obj.full_write_locs = torch.zeros(128, dtype=torch.int64)
    obj.swa_write_locs = torch.zeros(128, dtype=torch.int64)
    obj.tail_indices = torch.zeros(8, dtype=torch.int64)
    obj.mirror_positions = torch.zeros(8, dtype=torch.int64)
    obj.mirror_q_used = torch.zeros(8, dtype=torch.int32)
    obj.mirror_hidden = obj.mirror_residual = None
    obj.first_mirror = 0
    obj.mirror_residual_is_partial = False

    def handoff(**kwargs):
        index = kwargs["tail_indices"]
        return (
            kwargs["hidden_states"].index_select(0, index),
            kwargs["residual"].index_select(0, index),
            False,
        )

    first = types.SimpleNamespace(
        _welm_dp_executor=types.SimpleNamespace(prepare_graph_mirror_handoff=handoff)
    )
    obj.model = types.SimpleNamespace(
        oe_grams=[], layers_to_capture=[], layers=[first], end_layer=0
    )
    obj.backend = CpuBackend()
    obj.flash_inputs = obj.backend.create_welm_prefill_graph_metadata(8)
    return obj


def make_batch(lengths, capacity=128, *, prefix=0):
    real = sum(lengths)
    lens = torch.tensor(lengths, dtype=torch.int32)
    return types.SimpleNamespace(
        batch_size=len(lengths),
        forward_mode=1,
        enable_kv_mirror=True,
        input_ids=torch.zeros(capacity, dtype=torch.int64),
        positions=torch.cat(
            [
                *[
                    torch.arange(length, dtype=torch.int64) + prefix
                    for length in lengths
                ],
                torch.zeros(capacity - real, dtype=torch.int64),
            ]
        ),
        extend_seq_lens_cpu=list(lengths),
        extend_seq_lens=lens,
        extend_prefix_lens=torch.full_like(lens, prefix),
        extend_start_loc=torch.cumsum(lens, 0) - lens,
        orig_seq_lens=lens + prefix,
        seq_lens_cpu=lens + prefix,
        seq_lens=lens + prefix,
        req_pool_indices=torch.arange(len(lengths), dtype=torch.int64),
        out_cache_loc=torch.arange(1000, 1000 + capacity, dtype=torch.int64),
        global_num_tokens_cpu=[capacity, capacity],
        global_num_tokens_gpu=torch.tensor([capacity, capacity], dtype=torch.int32),
        global_dp_buffer_len=2 * capacity,
        num_token_non_padded_cpu=real,
        num_token_non_padded=torch.tensor([real], dtype=torch.int32),
        ngram_embedding_info=None,
        model_specific_states=None,
    )


def capture_metadata(obj):
    captured = {}
    for phase, capacity, lengths in [
        ("prompt", 128, [128]),
        ("mirror", 128, [16] * 8),
        ("mirror", 128, [32] * 4),
        ("mirror", 128, [64] * 2),
        ("mirror", 128, [128]),
    ]:
        obj.capture_phase = phase
        batch = make_batch(lengths, capacity)
        obj.prepare_capture(batch, capacity)
        key = obj.key(capacity) if phase == "prompt" else obj.mirror_key(len(lengths))
        obj.after_capture(key, batch)
        captured[key] = batch
    obj.capture_phase = "prompt"
    return captured


class TestTorchDpAdapter(unittest.TestCase):
    def test_live_vectors_flash_metadata_and_handoff_survive_large_small_large(self):
        for rank in (0, 1):
            obj = make_adapter(rank)
            captured = capture_metadata(obj)
            pointers = {
                name: getattr(obj, name).data_ptr()
                for name in (
                    "prompt_real_rows_gpu",
                    "request_real_rows_gpu",
                    "tail_indices",
                    "mirror_q_used",
                    "full_write_locs",
                    "swa_write_locs",
                )
            }
            prompt_meta = obj.flash_metadata[obj.key(128)]
            mirror_meta = obj.flash_metadata[obj.mirror_key(4)]
            self.assertEqual(
                prompt_meta.welm_flash_seqused_kv.data_ptr(),
                mirror_meta.welm_flash_seqused_kv.data_ptr(),
            )
            self.assertEqual(
                mirror_meta.welm_flash_seqused_q.data_ptr(),
                obj.mirror_q_used.data_ptr(),
            )
            for step, pair in enumerate(
                (([41, 41, 41], [61]), ([3], [2]), ([30, 31, 32], [28, 29]))
            ):
                lengths = pair[rank]
                tokens = tuple(map(sum, pair))
                requests = tuple(map(len, pair))
                bcap = 1 if max(requests) == 1 else 4
                live = make_batch(lengths, prefix=17 + step)
                live.welm_prefill_graph_ticket = types.SimpleNamespace(
                    allowed=True,
                    token_capacity=128,
                    request_capacity=bcap,
                    token_rows=tokens,
                    request_rows=requests,
                )
                static = make_batch(lengths, prefix=17 + step)
                self.assertTrue(obj.local_can_run(live, 128, bcap), obj._warned)
                obj.prepare_replay(live, static, 128)
                self.assertEqual(obj.prompt_real_rows_gpu.tolist(), list(tokens))
                self.assertEqual(obj.request_real_rows_gpu.tolist(), list(requests))
                self.assertEqual(obj.current_flash_metadata, prompt_meta)
                self.assertEqual(
                    prompt_meta.welm_flash_seqused_q[: len(lengths)].tolist(), lengths
                )
                active_mirror = obj.flash_metadata[obj.mirror_key(bcap)]
                self.assertEqual(
                    active_mirror.welm_flash_seqused_q.tolist(),
                    [1] * len(lengths) + [0] * (bcap - len(lengths)),
                )
                self.assertEqual(
                    active_mirror.welm_flash_seqused_kv[: len(lengths)].tolist(),
                    [x + 17 + step for x in lengths],
                )
                self.assertEqual(
                    obj.full_write_locs[sum(lengths) :].tolist(),
                    [-1] * (128 - sum(lengths)),
                )
                self.assertEqual(
                    obj.swa_write_locs[: sum(lengths)].tolist(),
                    (live.out_cache_loc[: sum(lengths)] + 32).tolist(),
                )
                # Produce a distinct prompt output each round. The mirror
                # reads fixed handoff storage after its latest prompt copy.
                hidden = (
                    torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)
                    + step * 1000
                ).to(torch.bfloat16)
                hidden[0].fill_(float("nan"))  # dummy tails select row zero safely
                residual = hidden.float() + 0.25
                module = types.SimpleNamespace(
                    KVMirrorManager=object(), WELMV4_MTP_MIRROR_STATES_KEY="mtp"
                )
                with patch.dict(sys.modules, {"sglang.srt.models.welmv4": module}):
                    obj.finish_prompt(hidden, residual, static.positions, static)
                handed, resid, positions = obj.begin_mirror(
                    captured[obj.mirror_key(bcap)]
                )
                tails = torch.cumsum(torch.tensor(lengths), 0) - 1
                torch.testing.assert_close(handed[: len(lengths)], hidden[tails])
                torch.testing.assert_close(resid[: len(lengths)], residual[tails])
                self.assertEqual(
                    positions[: len(lengths)].tolist(),
                    [length - 1 + 17 + step for length in lengths],
                )
                self.assertEqual(handed[len(lengths) :].count_nonzero().item(), 0)
                handed.zero_()  # mirror norm/attention may mutate their clone
                torch.testing.assert_close(
                    obj.mirror_hidden[: len(lengths)], hidden[tails]
                )
                self.assertEqual(
                    pointers, {name: getattr(obj, name).data_ptr() for name in pointers}
                )

    def test_preflight_rejects_shapes_that_would_fail_staging_before_the_vote(self):
        obj = make_adapter(0)
        capture_metadata(obj)
        for name, size in (
            ("out_cache_loc", 4),
            ("positions", 4),
            ("extend_seq_lens", 1),
            ("extend_prefix_lens", 1),
            ("extend_start_loc", 1),
            ("orig_seq_lens", 1),
            ("seq_lens", 1),
            ("seq_lens_cpu", 0),
            ("req_pool_indices", 1),
        ):
            batch = make_batch([30, 31])
            setattr(batch, name, torch.zeros(size, dtype=torch.int64))
            self.assertFalse(obj.local_can_run(batch, 128, 4), name)
        for name in (
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "orig_seq_lens",
            "seq_lens",
            "seq_lens_cpu",
            "req_pool_indices",
            "out_cache_loc",
            "positions",
        ):
            batch = make_batch([30, 31])
            setattr(batch, name, getattr(batch, name).unsqueeze(0))
            self.assertFalse(obj.local_can_run(batch, 128, 4), name)
        for name in ("extend_prefix_lens", "extend_start_loc"):
            batch = make_batch([30, 31])
            setattr(batch, name, None)
            self.assertFalse(obj.local_can_run(batch, 128, 4), name)
        batch = make_batch([30, 31])
        del obj._row_layouts[("mirror", 4, 3)]
        self.assertFalse(obj.local_can_run(batch, 128, 4))

    def test_preflight_accepted_valid_batches_stage_with_real_torch(self):
        obj = make_adapter(1)
        capture_metadata(obj)
        for local_lengths, other_lengths in (
            ([1], [2]),
            ([13], [35, 50, 22]),
            ([14, 19], [2]),
        ):
            batch = make_batch(local_lengths, prefix=20)
            bcap = 1 if max(len(local_lengths), len(other_lengths)) == 1 else 4
            batch.welm_prefill_graph_ticket = types.SimpleNamespace(
                allowed=True,
                token_capacity=128,
                request_capacity=bcap,
                token_rows=(sum(other_lengths), sum(local_lengths)),
                request_rows=(len(other_lengths), len(local_lengths)),
            )
            self.assertTrue(obj.local_can_run(batch, 128, bcap), obj._warned)
            obj.prepare_replay(batch, make_batch(local_lengths, prefix=20), 128)


if __name__ == "__main__":
    unittest.main()
