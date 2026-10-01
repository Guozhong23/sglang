"""CPU contracts for generic/SWA transfer preparation, without torch or MF.

Execute the production methods using the same AST harness as the Ascend tests.
"""

import concurrent.futures
import logging
import threading
import unittest
from collections import deque
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
from test_ascend_transfer_addresses import Engine, group
from test_pd_time_stats import SRT, pd, source_functions

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    pass
else:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")


entry_pairs = source_functions(
    SRT / "disaggregation/utils.py",
    {"build_transfer_entry_pairs"},
    {"deque": deque},
)["build_transfer_entry_pairs"]
methods = source_functions(
    SRT / "disaggregation/mooncake/conn.py",
    {"_send_kvcache_generic", "_transfer_data"},
    {
        "concurrent": concurrent,
        "group_concurrent_contiguous": group,
        "build_transfer_entry_pairs": entry_pairs,
        "logger": logging.getLogger(__name__),
    },
    "MooncakeKVManager",
)
mapping = source_functions(
    SRT / "disaggregation/common/conn.py",
    {"get_mha_kv_ptrs_with_pp", "get_mla_kv_ptrs_with_pp"},
    {},
    "CommonKVManager",
)


class Manager:
    _send_kvcache_generic = methods["_send_kvcache_generic"]
    get_mha_kv_ptrs_with_pp = mapping["get_mha_kv_ptrs_with_pp"]
    get_mla_kv_ptrs_with_pp = mapping["get_mla_kv_ptrs_with_pp"]

    def __init__(self, custom=False, pp=1, mla=False, hybrid=False):
        self.enable_custom_mem_pool = custom
        self.pp_size = pp
        self.is_mla_backend = mla
        self.is_hybrid_mla_backend = hybrid
        self.kv_args = NS(prefill_start_layer=2)
        self.engine = NS(batch_transfer_sync=Mock(return_value=0))
        self._transfer_data = Mock(side_effect=AssertionError("tuple transpose"))


class TestGenericTransferAddresses(unittest.TestCase):
    src = [2, 3, 7, 8, 9, 12]
    dst = [20, 21, 40, 42, 43, 50]
    src_ptrs = [2**48 + i * 100_000 for i in range(4)]
    dst_ptrs = [2**49 + i * 200_000 for i in range(8)]
    sizes = [64, 128, 256, 512]
    # Independently calculated: either source or target gaps split a run.
    host_runs = [(2, 20, 2), (7, 40, 1), (8, 42, 2), (12, 50, 1)]
    device_runs = [(2, 100, 2), (7, 102, 3), (12, 105, 1)]

    def send(self, manager, executor=None, src=None, dst=None, **kwargs):
        if executor is None:
            executor = Mock(spec=concurrent.futures.ThreadPoolExecutor)
        return manager._send_kvcache_generic(
            "peer",
            kwargs.pop("src_data_ptrs", self.src_ptrs),
            kwargs.pop(
                "dst_data_ptrs",
                self.dst_ptrs[:4] if manager.pp_size == 1 else self.dst_ptrs,
            ),
            kwargs.pop("item_lens", self.sizes),
            np.array(self.src if src is None else src, dtype=np.int32),
            np.array(self.dst if dst is None else dst, dtype=np.int32),
            executor,
            state_type="swa",
            **kwargs,
        )

    def expected(self, slots, device_slots=()):
        return [
            (
                src_ptr + start * size,
                self.dst_ptrs[slot] + target * size,
                pages * size,
            )
            for src_ptr, slot, size in zip(self.src_ptrs, slots, self.sizes)
            for start, target, pages in (
                self.device_runs if slot in device_slots else self.host_runs
            )
        ]

    def assert_columns(self, call, expected):
        peer, src_addrs, dst_addrs, lengths = call
        self.assertEqual(peer, "peer")
        for column in (src_addrs, dst_addrs, lengths):
            self.assertIs(type(column), list)
            self.assertTrue(all(type(value) is int for value in column))
        self.assertEqual(list(zip(src_addrs, dst_addrs, lengths)), expected)

    def test_batch_addresses_layouts_and_pp_order(self):
        for pp in (1, 2):
            for layout in ("mha", "mla", "hybrid", "flat"):
                with self.subTest(pp=pp, layout=layout):
                    manager = Manager(
                        pp=pp, mla=layout == "mla", hybrid=layout == "hybrid"
                    )
                    executor = Mock()
                    self.assertEqual(
                        self.send(manager, executor, force_flat=layout == "flat"), 0
                    )
                    slots = (
                        [0, 1, 2, 3]
                        if pp == 1
                        else ([2, 3, 6, 7] if layout == "mha" else [2, 3, 4, 5])
                    )
                    manager.engine.batch_transfer_sync.assert_called_once()
                    self.assert_columns(
                        manager.engine.batch_transfer_sync.call_args.args,
                        self.expected(slots),
                    )
                    executor.submit.assert_not_called()
                    manager._transfer_data.assert_not_called()

    def test_flat_repeated_layer_ids_keep_occurrence_order(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                manager = Manager(custom=custom, pp=2)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertEqual(
                        self.send(
                            manager,
                            pool,
                            force_flat=True,
                            src_layer_ids=[11, 13, 11, 13],
                            dst_layer_ids=[9, 11, 13, 9, 13, 11],
                            dst_data_ptrs=self.dst_ptrs[:6],
                        ),
                        0,
                    )
                calls = sorted(
                    manager.engine.batch_transfer_sync.call_args_list,
                    key=lambda c: c.args[1][0],
                )
                expected = self.expected([1, 2, 5, 4])
                for i, call in enumerate(calls):
                    self.assert_columns(
                        call.args,
                        expected[4 * i : 4 * (i + 1)] if custom else expected,
                    )
                manager._transfer_data.assert_not_called()

    def test_mixed_host_device_pages_and_worker_owned_lists(self):
        for custom in (False, True):
            with self.subTest(custom=custom):
                manager = Manager(custom=custom)
                calls = []
                barrier = threading.Barrier(4) if custom else None

                def transfer(*args):
                    calls.append(args)  # Retain original lists until all calls finish.
                    if barrier is not None:
                        barrier.wait(timeout=5)
                    return 0

                manager.engine.batch_transfer_sync = transfer
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertEqual(
                        self.send(
                            manager,
                            pool,
                            dst_device_data_indices=np.arange(100, 106, dtype=np.int32),
                            dst_device_data_ptrs={self.dst_ptrs[1], self.dst_ptrs[3]},
                        ),
                        0,
                    )
                expected = self.expected([0, 1, 2, 3], device_slots=(1, 3))
                self.assertEqual(len(calls), 4 if custom else 1)
                self.assertEqual(
                    len({id(col) for call in calls for col in call[1:]}), 3 * len(calls)
                )
                offset = 0
                for call in sorted(calls, key=lambda c: c[1][0]):
                    count = len(call[1])
                    self.assert_columns(call, expected[offset : offset + count])
                    offset += count
                self.assertEqual(offset, len(expected))
                manager._transfer_data.assert_not_called()

    def test_empty_transfer_never_calls_engine(self):
        for custom in (False, True):
            for src, dst in (([], []), ([], [1]), ([1], [])):
                manager = Manager(custom=custom)
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertEqual(self.send(manager, pool, src=src, dst=dst), 0)
                manager.engine.batch_transfer_sync.assert_not_called()
            manager = Manager(custom=custom)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(
                    self.send(
                        manager, pool, src_data_ptrs=[], dst_data_ptrs=[], item_lens=[]
                    ),
                    0,
                )
            manager.engine.batch_transfer_sync.assert_not_called()

    def test_invalid_indices_and_metadata_fail_before_engine(self):
        manager = Manager(pp=2)
        with self.assertRaisesRegex(ValueError, "equal-length"):
            self.send(manager, src=[1], dst=[1, 2])
        with self.assertRaisesRegex(ValueError, "equal-length"):
            self.send(manager, dst_device_data_indices=np.array([1], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            self.send(
                manager,
                force_flat=True,
                src_layer_ids=[1, 2, 3, 9],
                dst_layer_ids=[1, 2, 3, 4],
                dst_data_ptrs=self.dst_ptrs[:4],
            )
        manager.engine.batch_transfer_sync.assert_not_called()

    def test_return_codes_exceptions_and_pending_cancellation(self):
        manager = Manager()
        manager.engine.batch_transfer_sync.return_value = -7
        self.assertEqual(self.send(manager), -7)
        manager.engine.batch_transfer_sync.side_effect = RuntimeError("transport")
        with self.assertRaisesRegex(RuntimeError, "transport"):
            self.send(manager)

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
        self.assertTrue(all(f.cancelled() for f in futures[1:]))
        manager.engine.batch_transfer_sync.assert_called_once()
        manager.engine.batch_transfer_sync.side_effect = RuntimeError("transport")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            with self.assertRaisesRegex(RuntimeError, "transport"):
                self.send(manager, pool)

    def test_engine_wrapper_and_state_timing_are_preserved(self):
        for custom in (False, True):
            for failure in (False, True):
                with self.subTest(custom=custom, failure=failure):
                    manager = Manager(custom=custom)
                    native = Mock(return_value=0)
                    if failure:
                        native.side_effect = RuntimeError("MF error")
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
                                transfer = pd.timed_component("state")(
                                    lambda: self.send(manager, pd.TimingExecutor(pool))
                                )
                                self.assertEqual(transfer(), -1 if failure else 0)
                        finally:
                            pd.end_chunk(token, timing)
                    state = records[0]["components"]["state"]
                    self.assertEqual(
                        state["engine"]["bytes"],
                        sum(sum(c.args[3]) for c in native.call_args_list),
                    )
                    self.assertEqual(
                        state["engine"]["segments"],
                        sum(len(c.args[3]) for c in native.call_args_list),
                    )
                    self.assertEqual(state["engine"]["count"], native.call_count)
                    self.assertEqual(
                        state["engine"]["failures"], native.call_count if failure else 0
                    )
                    self.assertEqual(state["host"]["failures"], int(failure))
                    self.assertIsNone(pd._context.get())

    def test_disabled_timing_does_not_read_clock(self):
        manager = Manager()
        manager.engine = Engine(Mock(return_value=0))
        with patch.object(pd.time, "perf_counter_ns", side_effect=AssertionError):
            self.assertEqual(self.send(manager), 0)

    def test_legacy_tuple_helper_remains_available_to_other_callers(self):
        manager = Manager()
        legacy = methods["_transfer_data"]
        self.assertEqual(legacy(manager, "peer", []), 0)
        manager.engine.batch_transfer_sync.assert_not_called()
        rows = [(101, 201, 64), (102, 202, 128)]
        self.assertEqual(legacy(manager, "peer", rows), 0)
        self.assert_columns(manager.engine.batch_transfer_sync.call_args.args, rows)


if __name__ == "__main__":
    unittest.main()
