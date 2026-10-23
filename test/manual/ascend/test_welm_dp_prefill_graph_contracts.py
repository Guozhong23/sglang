"""Host-only DP graph storage/admission tests; no NPU/HCCL claim.

Run directly with Python and numpy. Production methods are loaded from AST
so torch and an initialized distributed runtime are not required.
"""

import copy
from contextlib import nullcontext
import dataclasses
import logging
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from test_welmv4_prefill_graph_contracts import Adapter, ShapeKey, SRT, load_nodes
from test_welmv4_prefill_graph_split_math import Tensor

TORCH = types.SimpleNamespace(
    tensor=lambda data, dtype=None, **kwargs: Tensor(np.asarray(data, dtype=dtype)),
    int32=np.int32,
    Tensor=Tensor,
)
MAX_LEN = types.SimpleNamespace(is_max_len=lambda: True)
ROWS = load_nodes(
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
        "torch": TORCH,
        "ceil_align": lambda x, a: ((x + a - 1) // a) * a,
    },
)
DP = load_nodes(
    SRT / "model_executor/runner/welm_dp_prefill_graph.py",
    ["validate_dp_graph_rows", "WelmDpPrefillGraphAdapter"],
    {
        "WelmPrefillGraphAdapter": Adapter,
        "ShapeKey": ShapeKey,
        "copy": copy,
        "torch": TORCH,
        "logger": logging.getLogger(__name__),
        "ForwardMode": types.SimpleNamespace(EXTEND=1, IDLE=4),
        "nullcontext": nullcontext,
        "DpPaddingMode": types.SimpleNamespace(MAX_LEN=MAX_LEN),
        "build_welm_dp_moe_row_layout": ROWS["build_welm_dp_moe_row_layout"],
    },
)
DpAdapter = DP["WelmDpPrefillGraphAdapter"]


def ticket(**kwargs):
    values = dict(
        allowed=True,
        token_capacity=128,
        request_capacity=4,
        token_rows=(123, 61),
        request_rows=(3, 1),
    )
    values.update(kwargs)
    return types.SimpleNamespace(**values)


def adapter():
    obj = DpAdapter.__new__(DpAdapter)
    obj.active_mask = 3
    obj.active_masks = (3,)
    obj.dp_size = 2
    obj.dp_rank = 1
    obj.attn_tp_size = 2
    obj.attn_tp_rank = 0
    obj.max_requests = 8
    obj.device = "cpu"
    obj.prune = True
    obj._warned = set()
    obj._row_layouts = {}
    obj._token_rows = (0, 0)
    obj._request_rows = (0, 0)
    obj.prompt_real_rows_gpu = Tensor(np.zeros(2, dtype=np.int32))
    obj.request_real_rows_gpu = Tensor(np.zeros(2, dtype=np.int32))
    obj.prompt_local_valid_rows = Tensor(np.zeros(1, dtype=np.int32))
    obj.request_local_valid_rows = Tensor(np.zeros(1, dtype=np.int32))
    obj.capture_keys = {obj.key(128), obj.mirror_key(4)}
    return obj


class TestDpGraphContracts(unittest.TestCase):
    def test_common_buckets_accept_unequal_real_rows(self):
        self.assertEqual(
            DP["validate_dp_graph_rows"]([12000, 6000], [3, 1], 16384, 4),
            ((12000, 6000), (3, 1)),
        )
        for tokens, requests, tc, bc in [
            ([0, 0], [0, 0], 128, 4),
            ([1, 2], [1, 3], 128, 4),
            ([129, 2], [1, 1], 128, 4),
            ([2, 2], [1, 1], 127, 4),
            ([2], [1], 128, 4),
        ]:
            with self.assertRaises(ValueError):
                DP["validate_dp_graph_rows"](tokens, requests, tc, bc)

    def test_can_run_only_reads_unanimous_ticket_and_keys(self):
        obj = adapter()
        batch = types.SimpleNamespace(welm_prefill_graph_ticket=ticket())
        self.assertTrue(obj.can_run(batch, 128))
        self.assertFalse(obj.can_run(batch, 256))
        batch.welm_prefill_graph_ticket = ticket(allowed=False)
        self.assertFalse(obj.can_run(batch, 128))
        batch.welm_prefill_graph_ticket = ticket()
        obj.capture_keys.remove(obj.mirror_key(4))
        self.assertFalse(obj.can_run(batch, 128))

    def test_missing_mirror_never_executes_prompt(self):
        obj = adapter()
        calls = []
        obj.runner = types.SimpleNamespace(
            backend=types.SimpleNamespace(
                replay=lambda key, batch, **kwargs: calls.append(key)
            )
        )
        batch = types.SimpleNamespace(welm_prefill_graph_ticket=ticket())
        obj.capture_keys.remove(obj.mirror_key(4))
        with self.assertRaises(RuntimeError):
            obj.replay(obj.key(128), batch)
        self.assertEqual(calls, [])

    def test_common_mirror_capacity_not_local_batch_size(self):
        obj = adapter()
        calls = []
        obj.runner = types.SimpleNamespace(
            backend=types.SimpleNamespace(
                replay=lambda key, batch, **kwargs: calls.append(key) or key
            )
        )
        obj.publish_state = lambda key, batch: calls.append("publish")
        batch = types.SimpleNamespace(batch_size=1, welm_prefill_graph_ticket=ticket())
        result = obj.replay(obj.key(128), batch)
        self.assertEqual(result, obj.mirror_key(4))
        self.assertEqual(calls, [obj.key(128), obj.mirror_key(4), "publish"])

    def test_live_vectors_and_bucket_views_keep_storage_on_repeated_replay(self):
        obj = adapter()
        obj._copy_live_rows((123, 61), (3, 1))
        prompt = obj._layout("prompt", 128, create=True)
        mirror = obj._layout("mirror", 4, create=True)
        addresses = (
            id(obj.prompt_real_rows_gpu.data),
            id(obj.request_real_rows_gpu.data),
        )
        self.assertIs(prompt.current.real_rows_gpu, obj.prompt_real_rows_gpu)
        self.assertIs(prompt.request.real_rows_gpu, obj.request_real_rows_gpu)
        self.assertIs(mirror.current.real_rows_gpu, obj.request_real_rows_gpu)
        self.assertEqual(prompt.current.slot_rows_cpu, [128, 128])
        self.assertEqual(prompt.request.slot_rows_cpu, [8, 8])
        self.assertEqual(mirror.request.slot_rows_cpu, [4, 4])
        for tokens, requests in [
            ((120, 62), (4, 2)),
            ((13, 3), (2, 1)),
            ((123, 61), (3, 1)),
        ]:
            obj._copy_live_rows(tokens, requests)
            self.assertIs(obj._layout("prompt", 128), prompt)
            self.assertIs(obj._layout("mirror", 4), mirror)
            self.assertEqual(prompt.current.real_rows_cpu, list(tokens))
            self.assertEqual(prompt.request.real_rows_cpu, list(requests))
            self.assertEqual(mirror.current.real_rows_cpu, list(requests))
            np.testing.assert_array_equal(obj.prompt_real_rows_gpu.data, tokens)
            np.testing.assert_array_equal(obj.request_real_rows_gpu.data, requests)
            self.assertEqual(prompt.current.global_slot_rows, 256)
            self.assertEqual(mirror.current.global_slot_rows, 8)
            self.assertEqual(
                addresses,
                (id(obj.prompt_real_rows_gpu.data), id(obj.request_real_rows_gpu.data)),
            )

    def test_capture_head_never_fills_real_vectors_with_slot_or_first_dp_count(self):
        obj = adapter()
        obj._copy_live_rows((123, 61), (3, 1))
        obj._layout("prompt", 128, create=True)
        obj._layout("mirror", 4, create=True)
        obj.flash_metadata = {
            key: types.SimpleNamespace(welm_flash_schedules={})
            for key in obj.capture_keys
        }
        kv = types.SimpleNamespace(activations_dict_kv={})
        module = types.SimpleNamespace(KVMirrorManager=kv)
        for phase, rows, expected_scalar in [
            ("prompt", 128, 61),
            ("mirror", 4, 1),
            ("prompt", 128, 61),
        ]:
            batch = types.SimpleNamespace(
                welm_prefill_graph_phase=phase,
                batch_size=4,
                positions=Tensor(np.zeros(rows)),
                global_num_tokens_gpu=Tensor(np.zeros(2)),
                num_token_non_padded=Tensor(np.zeros(1)),
                model_specific_states=None,
            )
            with patch.dict(sys.modules, {"sglang.srt.models.welmv4": module}):
                obj.before_capture_forward(batch)
            np.testing.assert_array_equal(obj.prompt_real_rows_gpu.data, [123, 61])
            np.testing.assert_array_equal(obj.request_real_rows_gpu.data, [3, 1])
            np.testing.assert_array_equal(
                batch.global_num_tokens_gpu.data, [rows, rows]
            )
            self.assertEqual(batch.num_token_non_padded.item(), expected_scalar)

    def test_mirror_capacity_one_needs_no_scattered_alignment(self):
        obj = adapter()
        obj._copy_live_rows((2, 2), (1, 1))
        mirror = obj._layout("mirror", 1, create=True)
        self.assertEqual(mirror.request.slot_rows_cpu, [1, 1])
        with self.assertRaises(ValueError):
            obj._layout("prompt", 3, create=True)

    def test_mirror_capture_uses_bcap_even_with_max_token_buffers(self):
        obj = adapter()
        obj.capture_phase = "mirror"
        obj._creating_capture = True
        obj._copy_live_rows((128, 128), (3, 3))
        batch = types.SimpleNamespace(batch_size=3)
        with patch.object(Adapter, "_bind", lambda self, batch, capacity: None):
            obj._bind(batch, 128)
        self.assertIn(("mirror", 3, 3), obj._row_layouts)
        self.assertNotIn(("mirror", 128), obj._row_layouts)
        self.assertEqual(batch.welm_dp_moe_row_layout.current.slot_rows_cpu, [3, 3])
        self.assertEqual(batch.welm_dp_raw_rows_cpu, [3, 3])
        self.assertEqual(batch.global_num_requests_cpu, [3, 3])

    def test_pure_tp_graph_keys_remain_distinct(self):
        obj = adapter()
        plain = Adapter.__new__(Adapter)
        plain.prune = True
        self.assertNotEqual(obj.key(128), plain.key(128))
        self.assertNotEqual(obj.mirror_key(4), plain.mirror_key(4))


if __name__ == "__main__":
    unittest.main()
