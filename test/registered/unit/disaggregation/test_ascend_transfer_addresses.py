"""CPU tests for Ascend KV transfer preparation (NumPy, no torch/NPU/MF).

Execute production methods from their AST to avoid hardware-dependent imports.
The real grouping, PP mapping, timing and engine wrapper run against fake MF.
"""

import concurrent.futures
import logging
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from test_pd_time_stats import SRT, pd, source_functions

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    pass
else:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


group = source_functions(
    SRT / "disaggregation/common/utils.py",
    {"group_concurrent_contiguous"},
    {"np": np},
)["group_concurrent_contiguous"]
methods = source_functions(
    SRT / "disaggregation/ascend/conn.py",
    {"send_kvcache", "get_mla_kv_ptrs_with_pp"},
    {
        "concurrent": concurrent,
        "group_concurrent_contiguous": group,
        "timed_component": pd.timed_component,
    },
    "AscendKVManager",
)
mha = source_functions(
    SRT / "disaggregation/common/conn.py",
    {"get_mha_kv_ptrs_with_pp"},
    {},
    "CommonKVManager",
)["get_mha_kv_ptrs_with_pp"]
batch_transfer_sync = source_functions(
    SRT / "distributed/device_communicators/mooncake_transfer_engine.py",
    {"batch_transfer_sync"},
    {
        "timed_transfer_call": pd.timed_transfer_call,
        "logger": logging.getLogger(__name__),
    },
    "MooncakeTransferEngine",
)["batch_transfer_sync"]


class Manager:
    send_kvcache = methods["send_kvcache"]
    get_mla_kv_ptrs_with_pp = methods["get_mla_kv_ptrs_with_pp"]
    get_mha_kv_ptrs_with_pp = mha

    def __init__(self, custom=False, pp=1, mla=False):
        self.enable_custom_mem_pool = custom
        self.pp_size = pp
        self.is_mla_backend = mla
        self.kv_args = NS(
            kv_data_ptrs=[2**48 + i * 100_000 for i in range(4)],
            kv_item_lens=[64, 128, 256, 512],
            prefill_start_layer=2,
            kv_buf_groups=2,
            total_kv_layers=4,
        )
        self.engine = NS(batch_transfer_sync=Mock(return_value=0))
        # This old tuple/transpose path must never run for Ascend direct KV.
        self._transfer_data = Mock(side_effect=AssertionError("tuple transpose"))


class Engine:
    batch_transfer_sync = batch_transfer_sync

    def __init__(self, native):
        self.engine = NS(batch_transfer_sync_write=native)


class TestAscendTransferAddresses(unittest.TestCase):
    src = [2, 3, 7, 8, 9, 12]
    dst = [20, 21, 40, 42, 43, 50]
    dst_ptrs = [2**49 + i * 200_000 for i in range(8)]

    def send(self, manager, executor=None, src=None, dst=None, **kwargs):
        if executor is None:
            executor = Mock(spec=concurrent.futures.ThreadPoolExecutor)
        return manager.send_kvcache(
            "peer",
            np.array(self.src if src is None else src, dtype=np.int32),
            self.dst_ptrs,
            np.array(self.dst if dst is None else dst, dtype=np.int32),
            executor,
            **kwargs,
        )

    def expected(self, manager, dst_slots):
        # Hand-calculated runs: either source or destination gaps split a run.
        return [
            (
                src_ptr + src_start * size,
                self.dst_ptrs[dst_slot] + dst_start * size,
                pages * size,
            )
            for src_ptr, dst_slot, size in zip(
                manager.kv_args.kv_data_ptrs, dst_slots, manager.kv_args.kv_item_lens
            )
            for src_start, dst_start, pages in (
                (2, 20, 2),
                (7, 40, 1),
                (8, 42, 2),
                (12, 50, 1),
            )
        ]

    def assert_columns(self, call, expected):
        peer, src_addrs, dst_addrs, lengths = call
        self.assertEqual(peer, "peer")
        for column in (src_addrs, dst_addrs, lengths):
            self.assertIs(type(column), list)
            self.assertTrue(all(type(value) is int for value in column))
            self.assertEqual(len(column), len(expected))
        self.assertEqual(list(zip(src_addrs, dst_addrs, lengths)), expected)

    def test_batch_addresses_order_and_pp_mapping(self):
        for pp in (1, 2):
            for mla in (False, True):
                with self.subTest(pp=pp, mla=mla):
                    manager = Manager(pp=pp, mla=mla)
                    executor = Mock()
                    self.assertEqual(self.send(manager, executor), 0)
                    slots = [0, 1, 2, 3] if pp == 1 else [2, 3, 6, 7]
                    manager.engine.batch_transfer_sync.assert_called_once()
                    self.assert_columns(
                        manager.engine.batch_transfer_sync.call_args.args,
                        self.expected(manager, slots),
                    )
                    executor.submit.assert_not_called()
                    manager._transfer_data.assert_not_called()

    def test_concurrent_layers_have_independent_final_lists(self):
        for pp, mla in ((1, False), (2, False), (2, True)):
            with self.subTest(pp=pp, mla=mla):
                manager = Manager(custom=True, pp=pp, mla=mla)
                barrier = threading.Barrier(4)
                calls = []

                def transfer(*args):
                    calls.append(args)  # Keep references, not copies.
                    barrier.wait(timeout=5)
                    return 0

                manager.engine.batch_transfer_sync = transfer
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    self.assertEqual(self.send(manager, executor), 0)
                slots = [0, 1, 2, 3] if pp == 1 else [2, 3, 6, 7]
                expected = self.expected(manager, slots)
                self.assertEqual(len(calls), 4)
                self.assertEqual(
                    len({id(col) for call in calls for col in call[1:]}), 12
                )
                for index, call in enumerate(
                    sorted(calls, key=lambda call: call[1][0])
                ):
                    self.assert_columns(call, expected[4 * index : 4 * (index + 1)])
                manager._transfer_data.assert_not_called()

    def test_empty_indices_or_layers_do_not_call_engine(self):
        for custom in (False, True):
            for src, dst in (([], []), ([], [1]), ([1], [])):
                with self.subTest(custom=custom, src=src, dst=dst):
                    manager = Manager(custom=custom)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        self.assertEqual(self.send(manager, pool, src=src, dst=dst), 0)
                    manager.engine.batch_transfer_sync.assert_not_called()
            manager = Manager(custom=custom)
            manager.kv_args.kv_data_ptrs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                self.assertEqual(self.send(manager, pool), 0)
            manager.engine.batch_transfer_sync.assert_not_called()

    def test_invalid_indices_fail_before_transfer(self):
        manager = Manager()
        with self.assertRaisesRegex(ValueError, "equal-length"):
            self.send(manager, src=[1], dst=[1, 2])
        with self.assertRaisesRegex(NotImplementedError, "device KV indices"):
            self.send(manager, dst_device_kv_indices=np.array([1], dtype=np.int32))
        manager.engine.batch_transfer_sync.assert_not_called()

    def test_batch_return_code_and_exception_are_preserved(self):
        manager = Manager()
        manager.engine.batch_transfer_sync.return_value = -7
        self.assertEqual(self.send(manager), -7)
        manager.engine.batch_transfer_sync.side_effect = RuntimeError("transport")
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.send(manager)

    def test_layer_failure_cancels_pending_futures(self):
        manager = Manager(custom=True)
        manager.engine.batch_transfer_sync.return_value = -7
        futures = []

        def submit(fn, *args):
            future = concurrent.futures.Future()
            if not futures:
                future.set_result(fn(*args))
            futures.append(future)
            return future

        self.assertEqual(self.send(manager, Mock(submit=submit)), -7)
        self.assertEqual(len(futures), 4)
        self.assertTrue(all(future.cancelled() for future in futures[1:]))
        manager.engine.batch_transfer_sync.assert_called_once()

    def test_layer_exception_propagates(self):
        manager = Manager(custom=True)
        manager.engine.batch_transfer_sync.side_effect = RuntimeError("transport")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with self.assertRaisesRegex(RuntimeError, "transport"):
                self.send(manager, pool)

    def test_existing_wrapper_preserves_timing_bytes_and_failures(self):
        for custom in (False, True):
            for failure in (False, True):
                with self.subTest(custom=custom, failure=failure):
                    native = Mock(return_value=0)
                    if failure:
                        native.side_effect = RuntimeError("MF error")
                    manager = Manager(custom=custom)
                    manager.engine = Engine(native)
                    timing = pd.RequestTiming("prefill", 123).new_chunk()
                    records = []
                    with patch.object(
                        pd, "emit", side_effect=lambda p, r: records.append(r)
                    ):
                        token = pd.begin_chunk(timing)
                        try:
                            with concurrent.futures.ThreadPoolExecutor(
                                max_workers=4
                            ) as pool:
                                result = self.send(manager, pd.TimingExecutor(pool))
                            self.assertEqual(result, -1 if failure else 0)
                        finally:
                            pd.end_chunk(token, timing)
                    kv = records[0]["components"]["kv"]
                    expected_bytes = sum(
                        sum(c.args[3]) for c in native.call_args_list
                    )
                    expected_segments = 4 * native.call_count if custom else 16
                    self.assertEqual(kv["engine"]["bytes"], expected_bytes)
                    self.assertEqual(kv["engine"]["count"], native.call_count)
                    self.assertEqual(kv["engine"]["segments"], expected_segments)
                    self.assertEqual(
                        kv["engine"]["failures"], native.call_count if failure else 0
                    )
                    self.assertEqual(kv["host"]["failures"], int(failure))
                    self.assertIsNone(pd._context.get())

    def test_disabled_timing_does_not_read_clock(self):
        manager = Manager()
        manager.engine = Engine(Mock(return_value=0))
        with patch.object(pd.time, "perf_counter_ns", side_effect=AssertionError):
            self.assertEqual(self.send(manager), 0)


if __name__ == "__main__":
    unittest.main()
