"""CPU behavior tests for WeLM metadata staging and Ascend graph producers.

Run from the repository root:
    python -m pytest -q test/manual/ascend/test_welm_metadata_staging.py

Production methods run on real CPU tensors. Only pinned allocation requests and
platform detection are mocked; these tests do not establish NPU DMA lifetime,
kernel performance or graph-capture compatibility.
"""

from __future__ import annotations

import ast
import copy
import logging
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"
FORWARD = SRT / "model_executor/forward_batch_info.py"
ASCEND = SRT / "hardware_backend/npu/attention/ascend_backend.py"


def load_nodes(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text())
    body = (
        tree.body
        if class_name is None
        else next(n.body for n in tree.body if getattr(n, "name", "") == class_name)
    )
    nodes = [copy.deepcopy(n) for n in body if getattr(n, "name", "") in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing production definitions: {names}")
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    namespace.setdefault("__name__", __name__)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return NS(**namespace)


class StagingTorch:
    """Record pin requests while retaining real CPU tensor arithmetic."""

    def __init__(self):
        self.allocations = []

    def __getattr__(self, name):
        return getattr(torch, name)

    def tensor(self, data, **kwargs):
        requested = dict(kwargs)
        kwargs.pop("pin_memory", None)
        tensor = torch.tensor(data, **kwargs)
        self.allocations.append((tensor, requested))
        return tensor


def row_api(torch_module):
    return load_nodes(
        FORWARD,
        [
            "WelmDpRowView",
            "WelmDpMoeRowLayout",
            "_welm_slot_rows",
            "_welm_build_row_view",
            "build_welm_dp_moe_row_layout",
        ],
        {
            "torch": torch_module,
            "dataclass": dataclass,
            "field": field,
            "ceil_align": lambda value, align: (value + align - 1) // align * align,
        },
    )


PADDING = load_nodes(
    FORWARD,
    ["_pad_tensor_to_size", "_pad_inputs_to_size"],
    {"torch": torch},
    class_name="ForwardBatch",
)
METADATA = load_nodes(
    ASCEND,
    ["ForwardMetadata"],
    {"torch": torch, "dataclass": dataclass, "field": field},
).ForwardMetadata
DEVICE = load_nodes(
    ASCEND,
    [
        "prepare_welm_prefill_graph_metadata_device",
        "_prepare_welm_graph_page_tables_device",
        "_prepare_welm_decode_metadata_device",
        "_init_cuda_graph_metadata",
        "init_forward_metadata_out_graph",
        "init_forward_metadata_in_graph",
    ],
    {"torch": torch, "ForwardMetadata": METADATA},
    class_name="AscendAttnBackend",
)


def backend_and_metadata(*, hybrid=True):
    # Only page heads are legal. Reading unused tokens would index outside the
    # SWA mapping, making an unmasked gather fail rather than silently pass.
    table = torch.full((4, 11), 999, dtype=torch.int64)
    table[0].zero_()
    table[1, [0, 4, 8]] = torch.tensor([4, 8, 12])
    table[2, [0, 4, 8]] = torch.tensor([16, 20, 24])
    table[3, [0, 4, 8]] = torch.tensor([28, 32, 36])
    mapping = torch.arange(64, dtype=torch.int64) + 40
    mapping[0] = 0
    mapping[8] = -1
    mapping[20] = -5
    metadata = METADATA(
        block_tables=torch.full((4, 4), 777, dtype=torch.int64),
        block_tables_swa=torch.full((4, 4), 777, dtype=torch.int64) if hybrid else None,
        welm_flash_seqused_q=torch.full((4,), 777, dtype=torch.int32),
        welm_flash_seqused_kv=torch.full((4,), 777, dtype=torch.int32),
    )
    starts = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
    backend = NS(
        req_to_token=table,
        full_to_swa_index_mapping=mapping,
        page_size=4,
        is_hybrid_swa=hybrid,
        forward_metadata=metadata,
        graph_metadata={
            "welm_decode_page_starts": starts,
            "welm_decode_page_in_bounds": starts < table.shape[1],
        },
        token_to_kv_pool=NS(
            translate_loc_from_full_to_swa=lambda indices: mapping[indices]
        ),
    )
    for name in (
        "_prepare_welm_graph_page_tables_device",
        "_prepare_welm_decode_metadata_device",
    ):
        setattr(backend, name, MethodType(getattr(DEVICE, name), backend))
    return backend, metadata


class TestPinnedRowStaging(unittest.TestCase):
    def test_fresh_host_staging_preserves_fixed_buffers_and_dtype(self):
        proxy = StagingTorch()
        api = row_api(proxy)
        current = torch.empty(4, dtype=torch.int64)
        requests = torch.empty(4, dtype=torch.int64)
        pointers = current.data_ptr(), requests.data_ptr()
        original_copy = torch.Tensor.copy_
        transfers = []

        def record_copy(target, source, *args, **kwargs):
            transfers.append((target, source, kwargs))
            return original_copy(target, source, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", new=record_copy):
            for counts, reqs in (
                ([32768, 0, 7, 1], [2, 0, 1, 1]),
                ([0, 3, 0, 9], [0, 1, 0, 3]),
            ):
                layout = api.build_welm_dp_moe_row_layout(
                    raw_rows_cpu=counts,
                    request_rows_cpu=reqs,
                    local_dp_rank=2,
                    device="cpu",
                    padding_mode=NS(is_max_len=lambda: True),
                    current_real_rows_gpu=current,
                    request_real_rows_gpu=requests,
                    pin_memory=True,
                )
                self.assertEqual(current.tolist(), counts)
                self.assertEqual(requests.tolist(), reqs)
                self.assertEqual((current.data_ptr(), requests.data_ptr()), pointers)
                self.assertIs(layout.current.real_rows_gpu, current)
                self.assertIs(layout.request.real_rows_gpu, requests)
        self.assertEqual(len(transfers), 4)
        self.assertTrue(all(call[2] == {"non_blocking": True} for call in transfers))
        self.assertEqual(len({tensor.data_ptr() for tensor, _ in proxy.allocations}), 4)
        for tensor, kwargs in proxy.allocations:
            self.assertEqual(
                kwargs, dict(dtype=torch.int64, device="cpu", pin_memory=True)
            )
            self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(proxy.allocations[0][0].tolist(), [32768, 0, 7, 1])

    def test_new_destination_transfer_respects_pin_policy(self):
        for pin in (False, True):
            proxy = StagingTorch()
            api = row_api(proxy)
            calls = []
            original_to = torch.Tensor.to

            def record_to(tensor, *args, **kwargs):
                calls.append((args, kwargs))
                return original_to(tensor, *args, **kwargs)

            with patch.object(torch.Tensor, "to", new=record_to):
                result = api._welm_build_row_view(
                    real_rows_cpu=[3, 0, 1, 2],
                    local_dp_rank=1,
                    device="cpu",
                    padding_mode=NS(is_max_len=lambda: False),
                    pin_memory=pin,
                )
            self.assertEqual(result.real_rows_gpu.tolist(), [3, 0, 1, 2])
            self.assertEqual(result.real_rows_gpu.dtype, torch.int32)
            self.assertEqual(
                calls, [((), dict(device="cpu", non_blocking=True))] if pin else []
            )
            self.assertEqual(proxy.allocations[0][1].get("pin_memory", False), pin)

    def test_wrong_fixed_capacity_fails_before_allocating_staging(self):
        proxy = StagingTorch()
        with self.assertRaisesRegex(ValueError, "wrong capacity"):
            row_api(proxy)._welm_build_row_view(
                real_rows_cpu=[3, 0, 1, 2],
                local_dp_rank=0,
                device="cpu",
                padding_mode=NS(is_max_len=lambda: False),
                real_rows_gpu=torch.empty(2, dtype=torch.int32),
                pin_memory=True,
            )
        self.assertEqual(proxy.allocations, [])


class TestUnpaddedAlias(unittest.TestCase):
    def test_only_opted_in_contiguous_unpadded_input_aliases(self):
        tensors = [
            torch.arange(12).reshape(3, 4),
            torch.arange(24).reshape(3, 8)[:, ::2],
        ]
        for tensor in tensors:
            for allow in (False, True):
                for size in (3, 5):
                    for value in (0, -1):
                        result = PADDING._pad_tensor_to_size(
                            None, tensor, size, value=value, allow_unpadded_alias=allow
                        )
                        aliases = allow and tensor.is_contiguous() and size == 3
                        self.assertEqual(result is tensor, aliases)
                        self.assertTrue(result.is_contiguous())
                        torch.testing.assert_close(result[:3], tensor)
                        if size > 3:
                            self.assertTrue(torch.all(result[3:] == value))
                        if not aliases:
                            self.assertNotEqual(result.data_ptr(), tensor.data_ptr())

    def test_input_alias_is_limited_to_read_only_device_fields(self):
        device_fields = {
            "input_ids": torch.arange(3),
            "req_pool_indices": torch.tensor([1, 2]),
            "seq_lens": torch.tensor([5, 8]),
            "out_cache_loc": torch.arange(3),
            "positions": torch.arange(3),
            "extend_seq_lens": torch.tensor([1, 2]),
        }
        copied_fields = {
            "seq_lens_cpu": torch.tensor([5, 8]),
            "encoder_lens": torch.tensor([0, 0]),
            "rids_int": torch.tensor([1, 2]),
            "bootstrap_room_ids_int": torch.tensor([7, 8]),
        }
        batch = NS(
            **device_fields,
            **copied_fields,
            lora_ids=None,
            seq_lens_sum=13,
            mamba_track_indices=None,
            mamba_track_mask=None,
            mamba_track_seqlens=None,
            mrope_positions=None,
            sampling_info=NS(),
            spec_info=None,
        )
        batch._pad_tensor_to_size = MethodType(PADDING._pad_tensor_to_size, batch)
        runner = NS(attn_backend=NS(get_cuda_graph_seq_len_fill_value=lambda: 1))
        PADDING._pad_inputs_to_size(batch, runner, 3, 2, allow_unpadded_alias=True)
        for name, original in device_fields.items():
            self.assertIs(getattr(batch, name), original)
        for name, original in copied_fields.items():
            self.assertIsNot(getattr(batch, name), original)
            torch.testing.assert_close(getattr(batch, name), original)
        self.assertIs(batch.sampling_info.rids_int, batch.rids_int)
        self.assertEqual(batch.seq_lens_sum, 13)


class TestNpuPlatform(unittest.TestCase):
    def test_pin_memory_policy_is_used_by_public_helper(self):
        method = load_nodes(
            SRT / "platforms/npu.py",
            ["is_pin_memory_available"],
            {},
            class_name="NPUDeviceMixin",
        ).is_pin_memory_available
        platform = NS(is_pin_memory_available=lambda device: method(None, device))
        public = load_nodes(
            SRT / "utils/common.py",
            ["is_pin_memory_available"],
            {"current_platform": platform},
        ).is_pin_memory_available
        for device, expected in (
            (None, True),
            ("npu", True),
            ("npu:3", True),
            ("cpu", False),
            (torch.device("cpu"), False),
        ):
            self.assertEqual(public(device), expected)

    def test_platform_discovery_uses_npu_fallback_and_preserves_explicit_cpu(self):
        class Cpu:
            pass

        class Npu:
            pass

        for cpu in (False, True):
            for npu in (False, True):
                api = load_nodes(
                    SRT / "platforms/__init__.py",
                    ["_is_npu_available", "_resolve_platform"],
                    {
                        "torch": NS(npu=NS(is_available=lambda: npu)),
                        "envs": NS(SGLANG_PLATFORM=NS(get=lambda: None)),
                        "load_plugins_by_group": lambda group: {},
                        "PLATFORM_PLUGINS_GROUP": "test",
                        "_is_cpu_available": lambda: cpu,
                        "_is_cuda_available": lambda: False,
                        "_is_rocm_available": lambda: False,
                        "_is_xpu_available": lambda: False,
                        "CpuSRTPlatform": Cpu,
                        "NPUSRTPlatform": Npu,
                        "SRTPlatform": object,
                        "logger": logging.getLogger(__name__),
                    },
                )
                self.assertIs(
                    type(api._resolve_platform()),
                    Cpu if cpu else Npu if npu else object,
                )


class TestAscendDeviceMetadata(unittest.TestCase):
    def test_prefill_masks_stale_rows_before_lookup_and_keeps_negative_swa_pages(self):
        backend, metadata = backend_and_metadata()
        fields = (
            "block_tables",
            "block_tables_swa",
            "welm_flash_seqused_q",
            "welm_flash_seqused_kv",
        )
        pointers = [getattr(metadata, name).data_ptr() for name in fields]
        DEVICE.prepare_welm_prefill_graph_metadata_device(
            backend,
            metadata,
            torch.tensor([1, 2, 999, 0]),
            torch.tensor([9, 25, 999, 4]),
            torch.tensor([2, 3, 0, 4]),
            backend.graph_metadata["welm_decode_page_starts"],
            backend.graph_metadata["welm_decode_page_in_bounds"],
        )
        self.assertEqual(
            metadata.block_tables.tolist(),
            [[1, 2, 3, 0], [4, 5, 6, 0], [0] * 4, [0] * 4],
        )
        self.assertEqual(
            metadata.block_tables_swa.tolist(),
            [[11, -1, 13, 0], [14, -2, 16, 0], [0] * 4, [0] * 4],
        )
        self.assertEqual(metadata.welm_flash_seqused_q.tolist(), [2, 3, 0, 4])
        self.assertEqual(metadata.welm_flash_seqused_kv.tolist(), [9, 25, 0, 4])
        # The next request can have a different owner and no other active rows.
        DEVICE.prepare_welm_prefill_graph_metadata_device(
            backend,
            metadata,
            torch.tensor([-999, -999, 3, -999]),
            torch.tensor([999, 999, 1, 999]),
            torch.tensor([0, 0, 1, 0]),
            backend.graph_metadata["welm_decode_page_starts"],
            backend.graph_metadata["welm_decode_page_in_bounds"],
        )
        self.assertEqual(
            metadata.block_tables.tolist(),
            [[0] * 4, [0] * 4, [7, 0, 0, 0], [0] * 4],
        )
        self.assertEqual(
            [getattr(metadata, name).data_ptr() for name in fields], pointers
        )

    def test_decode_masks_idle_rows_and_write_sentinels_before_translation(self):
        for hybrid in (False, True):
            backend, metadata = backend_and_metadata(hybrid=hybrid)
            metadata.swa_out_cache_loc = (
                torch.full((4,), 999, dtype=torch.int64) if hybrid else None
            )
            batch = NS(
                seq_lens=torch.tensor([5, 0, -1, 9]),
                req_pool_indices=torch.tensor([1, 999, -999, 2]),
                out_cache_loc=torch.tensor([8, -999, 999, 20]),
            )
            DEVICE._prepare_welm_decode_metadata_device(backend, batch)
            self.assertEqual(
                metadata.block_tables.tolist(),
                [[1, 2, 0, 0], [0] * 4, [0] * 4, [4, 5, 6, 0]],
            )
            self.assertEqual(metadata.welm_flash_seqused_q.tolist(), [1, 0, 0, 1])
            self.assertEqual(metadata.welm_flash_seqused_kv.tolist(), [5, 0, 0, 9])
            if hybrid:
                self.assertEqual(metadata.swa_out_cache_loc.tolist(), [-1, 0, 0, -5])
            batch.seq_lens.zero_()
            batch.req_pool_indices.fill_(999)
            batch.out_cache_loc.fill_(-999)
            DEVICE._prepare_welm_decode_metadata_device(backend, batch)
            self.assertFalse(torch.any(metadata.block_tables))
            self.assertFalse(torch.any(metadata.welm_flash_seqused_q))
            if hybrid:
                self.assertFalse(torch.any(metadata.block_tables_swa))
                self.assertFalse(torch.any(metadata.swa_out_cache_loc))

    def test_decode_capture_and_replay_use_device_metadata(self):
        backend, _ = backend_and_metadata()
        backend.use_welm_flash_attn = True
        backend.is_welm_v4 = True
        backend.use_sliding_window_kv_pool = True
        backend.swa_out_cache_loc_buf = torch.zeros(4, dtype=torch.int64)
        backend.q_head_num_padding = None
        backend.speculative_num_draft_tokens = 0
        backend.graph_metadata["block_tables"] = torch.zeros(4, 4, dtype=torch.int64)
        backend.graph_metadata["block_tables_swa"] = torch.zeros(
            4, 4, dtype=torch.int64
        )
        mode = NS(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_dllm_extend=lambda: False,
            is_decode_or_idle=lambda: True,
        )
        seq_lens = torch.tensor([3, 0, 5, 0])
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host read")):
            metadata = DEVICE._init_cuda_graph_metadata(
                backend,
                4,
                mode,
                seq_lens,
                torch.zeros(4, dtype=torch.int64),
                welm_decode_metadata_in_graph=True,
            )
        self.assertIs(metadata.seq_lens, seq_lens)
        self.assertIsNone(metadata.seq_lens_cpu_list)
        self.assertEqual(metadata.welm_flash_cu_seqlens_q.tolist(), [0, 1, 2, 3, 4])
        backend._apply_cuda_graph_metadata = Mock(
            side_effect=AssertionError("eager metadata")
        )
        batch = NS(
            batch_size=4,
            seq_lens=seq_lens,
            req_pool_indices=torch.tensor([1, 999, 2, -999]),
            out_cache_loc=torch.tensor([4, -999, 20, -999]),
        )
        DEVICE.init_forward_metadata_out_graph(backend, batch)
        metadata.welm_flash_schedules["stale"] = torch.ones(1)
        metadata.welm_flash_mirror_q_lengths = (torch.ones(1), torch.ones(1))
        DEVICE.init_forward_metadata_in_graph(backend, batch)
        self.assertEqual(metadata.welm_flash_seqused_q.tolist(), [1, 0, 1, 0])
        self.assertEqual(metadata.welm_flash_schedules, {})
        self.assertIsNone(metadata.welm_flash_mirror_q_lengths)
        backend._apply_cuda_graph_metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
