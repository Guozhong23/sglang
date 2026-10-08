"""CPU-only contracts for PD diagnostics; runnable without torch/Mooncake.

The poll tests execute the production functions with fake CPU tensors and a
collective, so timing must not change readiness or add metadata reads/reduces.
"""

import ast
import concurrent.futures
import importlib.util
import json
import logging
import queue
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # Also support running this stdlib-only test in a source-only checkout.
    pass
else:
    register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "pd_time_stats_under_test", SRT / "observability/pd_time_stats.py"
)
pd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd)


def source_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    body = tree.body
    if class_name:
        body = next(n for n in body if getattr(n, "name", None) == class_name).body
    selected = [n for n in body if getattr(n, "name", None) in names]
    assert len(selected) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class TestTransferTiming(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.patcher = patch.object(
            pd, "emit", side_effect=lambda prefix, record: self.records.append(record)
        )
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def chunk(self, room=1):
        return pd.RequestTiming("prefill", room, tp_rank=0).new_chunk()

    def test_disabled_path_preserves_values_and_does_not_read_clock(self):
        engine = pd.timed_transfer_call(lambda *args: 7)
        component = pd.timed_component("kv")(lambda: engine(None, "peer", [], [], []))
        send = pd.timed_send_prepare(lambda self, req, **kw: kw)
        req = NS(disagg_kv_sender=NS(pd_timing=None))
        with patch.object(pd.time, "perf_counter_ns", side_effect=AssertionError):
            self.assertEqual(component(), 7)
            self.assertEqual(send(None, req), {"last_chunk": False, "end_idx": None})
            self.assertIsNone(pd.begin_chunk(None))
            pd.end_chunk(None, None)
        self.assertEqual(self.records, [])

    def test_engine_bytes_rc_exceptions_and_context_restoration(self):
        timing = self.chunk()
        native = Mock(side_effect=[0, -5, RuntimeError("transport")])
        engine = pd.timed_transfer_call(native)

        @pd.timed_component("state")
        def transfer():
            return engine(None, "peer", [1, 2], [3, 4], [64, 128])

        token = pd.begin_chunk(timing)
        self.assertEqual(transfer(), 0)
        self.assertEqual(transfer(), -5)
        with self.assertRaisesRegex(RuntimeError, "transport"):
            transfer()
        pd.end_chunk(token, timing)
        self.assertIsNone(pd._context.get())
        stat = self.records[0]["components"]["state"]
        self.assertEqual(stat["engine"]["bytes"], 3 * 192)
        self.assertEqual(stat["engine"]["segments"], 6)
        self.assertEqual(stat["engine"]["failures"], 2)
        self.assertEqual(stat["host"]["failures"], 2)
        self.assertEqual(native.call_count, 3)

    def test_executor_keeps_rooms_and_components_separate(self):
        a, b = self.chunk(1), self.chunk(2)
        barrier = threading.Barrier(2)

        @pd.timed_transfer_call
        def native(self, peer, buffers, addresses, lengths):
            barrier.wait(timeout=5)
            return 0

        with ThreadPoolExecutor(max_workers=2) as pool:
            executor = pd.TimingExecutor(pool)
            futures = []
            for timing, component, size in ((a, "kv", 64), (b, "aux", 8)):
                token = pd._context.set((timing, component))
                try:
                    futures.append(executor.submit(native, None, "p", [], [], [size]))
                finally:
                    pd._context.reset(token)
            self.assertEqual([f.result(timeout=5) for f in futures], [0, 0])
            self.assertIsNone(pool.submit(pd._context.get).result(timeout=5))
        self.assertEqual(set(a.components), {"kv"})
        self.assertEqual(set(b.components), {"aux"})
        self.assertEqual(a.components["kv"]["engine"]["bytes"], 64)
        self.assertEqual(b.components["aux"]["engine"]["bytes"], 8)
        self.assertEqual(a.components["kv"]["executor_wait"]["count"], 1)

    def test_snapshot_is_immutable_and_missing_intervals_are_null(self):
        timing = self.chunk()
        timing.add_interval("kv", "engine", 10, 20, bytes=64)
        timing.finish()
        record = self.records[0]
        timing.add_interval("kv", "engine", 30, 40, bytes=128)
        timing.mark("late_event")
        self.assertEqual(record["components"]["kv"]["engine"]["bytes"], 64)
        self.assertNotIn("late_event", record["events_ns"])
        self.assertIsNone(record["queue_wait_ms"])
        self.assertIsNone(record["send_prepare_ms"])

    def test_deferred_chunk_emits_once_and_restores_context(self):
        timing = self.chunk()
        token = pd.begin_chunk(timing)
        start = timing.events["worker_begin"]
        pd.end_chunk(token, timing, deferred=True)
        self.assertEqual(self.records, [])
        self.assertIsNone(pd._context.get())
        token = pd.begin_chunk(timing)
        pd.end_chunk(token, timing)
        self.assertEqual(len(self.records), 1)
        self.assertEqual(self.records[0]["events_ns"]["worker_begin"], start)

    def test_receiver_first_observation_and_derived_intervals(self):
        timing = pd.RequestTiming("decode", 123)
        with patch.object(
            pd.time,
            "perf_counter_ns",
            side_effect=[1_000_000, 2_000_000, 3_000_000, 4_000_000],
        ):
            timing.mark("metadata_sent")
            timing.notice(0, 2)
            timing.notice(0, 2)
            timing.mark("all_notices_received")
        with patch.object(pd.time, "perf_counter_ns", side_effect=AssertionError):
            timing.mark("all_notices_received")
        timing.finish("success")
        record = self.records[0]
        self.assertEqual(record["notices_ns"], {"0": 2_000_000})
        self.assertEqual(record["durations_ms"]["metadata_sent_to_all_notices"], 3)
        self.assertIsNone(record["durations_ms"]["local_ready_to_tp_ready"])

    def test_failed_room_cleanup_and_room_reuse(self):
        ns = source_functions(
            SRT / "disaggregation/mooncake/conn.py",
            {"finish_request_timing"},
            {"KVPoll": NS(Success=4, Failed=0)},
            "MooncakeKVManager",
        )
        old, current = pd.RequestTiming("decode", 3), pd.RequestTiming("decode", 3)
        mgr = NS(pd_timings={3: current}, request_status={3: 0})
        finish = ns["finish_request_timing"]
        finish(mgr, 3, old)
        self.assertIs(mgr.pd_timings[3], current)
        self.assertEqual(self.records, [])
        finish(mgr, 3, current)
        finish(mgr, 3, current)
        self.assertEqual(mgr.pd_timings, {})
        self.assertEqual(len(self.records), 1)
        self.assertEqual(self.records[0]["status"], "failed")

    def test_send_prepare_classifies_prefix_middle_final(self):
        timing = pd.RequestTiming("prefill", 17)
        req = NS(
            disagg_kv_sender=NS(pd_timing=timing),
            rid="rid",
            prefill_attempt_count=1,
            origin_input_ids=[1] * 10,
            cached_tokens=8,
            time_stats=NS(forward_entry_time=1.0, prefill_finished_time=2.0),
        )

        @pd.timed_send_prepare
        def send(self, req, last_chunk=False, end_idx=None):
            return timing.new_chunk()

        for args, kind in (
            ({"end_idx": 8}, "cached_prefix"),
            ({}, "middle"),
            ({"last_chunk": True}, "final"),
        ):
            chunk = send(None, req, **args)
            self.assertEqual(chunk.fields["chunk_kind"], kind)
            self.assertEqual(chunk.fields["rid"], "rid")
            self.assertEqual(chunk.fields["prefill_result_ready_ns"], 2_000_000_000)

    def test_ascend_flat_and_per_layer_native_calls_are_both_timed(self):
        ns = {
            "timed_component": pd.timed_component,
            "timed_transfer_call": pd.timed_transfer_call,
            "concurrent": concurrent,
            "group_concurrent_contiguous": lambda src, dst: ([src], [dst]),
            "logger": Mock(),
        }
        source_functions(
            SRT / "disaggregation/ascend/conn.py",
            {"send_kvcache"}, ns, "AscendKVManager",
        )
        source_functions(
            SRT / "disaggregation/mooncake/conn.py",
            {"_transfer_data"}, ns, "MooncakeKVManager",
        )
        source_functions(
            SRT / "distributed/device_communicators/mooncake_transfer_engine.py",
            {"batch_transfer_sync"}, ns, "MooncakeTransferEngine",
        )
        native = Mock(return_value=0)
        engine = NS(engine=NS(batch_transfer_sync_write=native))
        engine.batch_transfer_sync = lambda *args: ns["batch_transfer_sync"](
            engine, *args
        )
        manager = NS(
            pp_size=1, kv_args=NS(kv_data_ptrs=[100, 200], kv_item_lens=[16, 16]),
            engine=engine,
        )
        manager._transfer_data = lambda *args: ns["_transfer_data"](manager, *args)
        for use_pool, expected_calls in ((False, 1), (True, 2)):
            with self.subTest(use_pool=use_pool):
                native.reset_mock()
                manager.enable_custom_mem_pool = use_pool
                timing = self.chunk()
                token = pd.begin_chunk(timing)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    result = ns["send_kvcache"](
                        manager, "peer", [0, 1], [300, 400], [0, 1],
                        pd.TimingExecutor(pool),
                    )
                pd.end_chunk(token, timing)
                self.assertEqual(result, 0)
                self.assertEqual(native.call_count, expected_calls)
                stat = self.records[-1]["components"]["kv"]["engine"]
                self.assertEqual(stat["count"], expected_calls)
                self.assertEqual(stat["bytes"], 64)
                self.assertEqual(stat["segments"], 2)


class TestAsyncLogging(unittest.TestCase):
    def test_full_queue_drops_without_blocking(self):
        writer = pd._LogWriter.__new__(pd._LogWriter)
        writer.queue = queue.Queue(maxsize=1)
        writer.dropped = 0
        writer.put("", {"a": 1})
        writer.put("", {"a": 2})
        self.assertEqual(writer.queue.qsize(), 1)
        self.assertEqual(writer.dropped, 1)

    def test_emit_runs_handler_in_background_and_preserves_record(self):
        received = []
        done = threading.Event()

        class Handler(logging.Handler):
            def emit(self, record):
                received.append((threading.get_ident(), record.getMessage()))
                done.set()

        logger = logging.Logger("test-pd", level=logging.INFO)
        logger.addHandler(Handler())
        original = {"bootstrap_room": 19}
        with patch.object(pd, "_logger", logger), patch.object(pd, "_writer", None):
            pd.emit("PDTransferStats ", original)
            self.assertTrue(done.wait(timeout=5))
            pd._drain_at_exit()
        tid, text = received[0]
        self.assertNotEqual(tid, threading.get_ident())
        record = json.loads(text.split(" ", 1)[1])
        self.assertEqual(record["bootstrap_room"], 19)
        self.assertEqual(record["dropped_records_total"], 0)
        self.assertEqual(original, {"bootstrap_room": 19})

    def test_disabled_logger_does_not_create_writer(self):
        with patch.object(pd, "_logger", logging.Logger("off", logging.WARNING)):
            with patch.object(pd, "_LogWriter", side_effect=AssertionError):
                pd.emit("PDTransferStats ", {})


class TestProductionPolling(unittest.TestCase):
    def test_metadata_gate_and_tp_consensus_unchanged(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                timing = pd.RequestTiming("decode", 1) if enabled else None
                receiver = NS(pd_timing=timing, poll=lambda: 4)
                dr = NS(kv_receiver=receiver, req=NS(), metadata_buffer_index=0)
                item = Mock(side_effect=[0, 1, 1])
                room = Mock()
                room.__getitem__ = Mock(return_value=NS(item=item))
                reductions = []
                replies = iter([3, 3, 4])

                def tensor(values, **kwargs):
                    values = list(values)
                    return NS(values=values, tolist=lambda: values.copy())

                def reduce(tensor, **kwargs):
                    reductions.append(tensor.values.copy())
                    tensor.values[0] = next(replies)

                ns = source_functions(
                    SRT / "disaggregation/utils.py",
                    {"_apply_metadata_gate", "poll_and_all_reduce"},
                    {
                        "KVPoll": NS(Success=4, Transferring=3),
                        "_poll_with_failure_injection": lambda pollers: [
                            p.poll() for p in pollers
                        ],
                        "_is_fake_transfer": lambda *args: False,
                        "torch": NS(tensor=tensor, uint8="uint8"),
                        "dist": NS(all_reduce=reduce, ReduceOp=NS(MIN="min")),
                        "mark_poll": pd.mark_poll,
                    },
                )
                poll = ns["poll_and_all_reduce"]
                args = ([receiver], None, [dr], NS(bootstrap_room=room), NS())
                self.assertEqual(poll(*args), [3])
                if enabled:
                    self.assertIn("metadata_wait_observed", timing.events)
                    self.assertNotIn("local_ready", timing.events)
                self.assertEqual(poll(*args), [3])
                if enabled:
                    self.assertIn("local_ready", timing.events)
                    self.assertNotIn("tp_consensus_ready", timing.events)
                self.assertEqual(poll(*args), [4])
                if enabled:
                    self.assertIn("tp_consensus_ready", timing.events)
                self.assertEqual(item.call_count, 3)
                self.assertEqual(reductions, [[3], [4], [4]])


if __name__ == "__main__":
    unittest.main()
