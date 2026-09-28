"""CPU-only GC attribution tests; no torch, NPU, MF, or heap snapshots."""

import ast
import gc
import importlib.util
import logging
import sys
import threading
import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_pd_time_stats import SRT, pd, source_functions

spec = importlib.util.spec_from_file_location(
    "pd_gc_under_test", SRT / "observability/pd_gc_diagnostics.py"
)
diag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag_module)


class TestGCAttribution(unittest.TestCase):
    def setUp(self):
        was_enabled = gc.isenabled()
        gc.disable()  # Deterministic test only; explicit collect still runs.
        self.addCleanup(gc.enable if was_enabled else gc.disable)
        self.diag = diag_module.PDGCDiagnostics(dict(role="prefill", tp_rank=0))
        self.diag.install()
        self.addCleanup(self.diag.uninstall)
        patcher = patch.object(pd, "_gc_diagnostics", self.diag)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.records = []
        patcher = patch.object(
            pd, "emit", side_effect=lambda p, r: self.records.append(r)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def chunk(self):
        return pd.RequestTiming("prefill", 123, tp_rank=0).new_chunk(queue_id=2)

    def test_real_gc_preparation_engine_and_after_return_are_distinct(self):
        timing = self.chunk()

        @pd.timed_transfer_call
        def engine(*args):
            gc.collect(2)
            return 0

        @pd.timed_component("kv")
        def build_descriptors():
            gc.collect(2)
            result = engine(None, "peer", [1], [2], [64])
            gc.collect(2)
            return result

        token = pd.begin_chunk(timing)
        self.assertEqual(build_descriptors(), 0)
        pd.end_chunk(token, timing)
        events = list(self.diag.drain())
        self.assertEqual(
            [e["phase"] for e in events], ["pre_engine", "engine", "post_engine"]
        )
        self.assertEqual(events[0]["callsite"][0]["function"], "build_descriptors")
        self.assertEqual(events[0]["native_tid"], threading.get_native_id())
        self.assertEqual(events[0]["bootstrap_room"], 123)
        prepared = self.records[-1]["gc_preparations"][0]
        self.assertTrue(prepared["supported"])
        self.assertEqual(prepared["gc_count_by_generation"], [0, 0, 1])
        kv = self.records[-1]["components"]["kv"]
        self.assertEqual(prepared["start_ns"], kv["host"]["first_start_ns"])
        self.assertEqual(prepared["end_ns"], kv["engine"]["first_start_ns"])
        self.assertIsNone(self.diag.scope())
        self.assertIsNone(pd._context.get())

    def test_other_thread_gc_has_no_false_send_scope(self):
        timing = self.chunk()

        @pd.timed_component("kv")
        def transfer():
            thread = threading.Thread(target=lambda: gc.collect(2))
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            return pd.timed_transfer_call(lambda *args: 0)(None, "p", [], [], [])

        token = pd.begin_chunk(timing)
        transfer()
        pd.end_chunk(token, timing)
        event, = self.diag.drain()
        self.assertNotEqual(event["native_tid"], threading.get_native_id())
        self.assertEqual(event["phase"], "outside_send")
        self.assertNotIn("bootstrap_room", event)
        self.assertEqual(
            self.records[-1]["gc_preparations"][0]["gc_count_by_generation"],
            [0, 0, 0],
        )

    def test_exception_early_return_and_multiple_calls_mark_unsupported(self):
        native = pd.timed_transfer_call(lambda *args: 0)

        @pd.timed_component("kv")
        def transfer(mode):
            if mode == "raise":
                raise RuntimeError("prepare")
            if mode == "two":
                native(None, "p", [], [], [])
                native(None, "p", [], [], [])
            return 0

        for mode in ("raise", "empty", "two"):
            timing = self.chunk()
            token = pd.begin_chunk(timing)
            if mode == "raise":
                with self.assertRaisesRegex(RuntimeError, "prepare"):
                    transfer(mode)
            else:
                transfer(mode)
            self.assertIsNone(self.diag.scope())
            pd.end_chunk(token, timing)
            self.assertFalse(self.records[-1]["gc_preparations"][0]["supported"])

    def test_native_exception_restores_context_without_hiding_failure(self):
        @pd.timed_transfer_call
        def native(*args):
            raise ValueError("native")

        transfer = pd.timed_component("kv")(lambda: native(None, "p", [], [], []))
        timing = self.chunk()
        token = pd.begin_chunk(timing)
        with self.assertRaisesRegex(ValueError, "native"):
            transfer()
        pd.end_chunk(token, timing)
        self.assertIsNone(self.diag.scope())
        self.assertEqual(
            self.records[-1]["components"]["kv"]["engine"]["failures"], 1
        )

    def test_disabled_has_no_probe_allocation_cpu_clock_or_thread_lookup(self):
        self.diag.uninstall()
        with patch.object(pd, "_gc_diagnostics", None), patch.object(
            diag_module.Preparation, "__init__", side_effect=AssertionError
        ), patch.object(time, "thread_time_ns", side_effect=AssertionError), patch.object(
            threading, "get_native_id", side_effect=AssertionError
        ), patch.object(
            diag_module.PDGCDiagnostics, "_callsite", side_effect=AssertionError
        ):
            timing = self.chunk()
            token = pd.begin_chunk(timing)
            native = pd.timed_transfer_call(lambda *args: 0)
            transfer = pd.timed_component("kv")(lambda: native(None, "p", [], [], []))
            self.assertEqual(transfer(), 0)
            pd.end_chunk(token, timing)
            self.assertFalse(hasattr(timing, "gc_preparations"))
        self.assertNotIn("gc_preparations", self.records[-1])
        self.assertNotIn(self.diag.callback, gc.callbacks)

    def test_install_is_idempotent_and_preserves_foreign_callbacks(self):
        foreign = lambda *args: None
        gc.callbacks.append(foreign)
        self.addCleanup(lambda: gc.callbacks.remove(foreign))
        self.diag.install()
        self.diag.install()
        self.assertEqual(gc.callbacks.count(self.diag.callback), 1)
        self.diag.uninstall()
        self.assertIn(foreign, gc.callbacks)

    def test_bounded_buffer_short_summary_and_unpaired_detection(self):
        self.diag.uninstall()
        diag = diag_module.PDGCDiagnostics({}, capacity=1)
        for gen in (0, 2, 2):
            diag._callback("start", {"generation": gen})
            diag._callback("stop", {"generation": gen, "collected": 0})
        self.assertEqual(diag.count, [1, 0, 2])
        self.assertEqual(diag.filtered, 1)
        self.assertEqual(diag.dropped, 1)
        self.assertEqual(len(list(diag.drain())), 1)
        diag._callback("stop", {"generation": 2})
        self.assertEqual(diag.status()["gc_unpaired"], 1)

    def test_writer_flushes_without_new_pd_messages(self):
        received = []
        ready = threading.Event()

        class Handler(logging.Handler):
            def emit(self, record):
                message = record.getMessage()
                received.append((threading.get_native_id(), message))
                if '"event_id"' in message:
                    ready.set()

        logger = logging.Logger("pd-gc-test", level=logging.INFO)
        logger.addHandler(Handler())
        with patch.object(pd, "_logger", logger):
            writer = pd._LogWriter()
            gc.collect(2)
            self.assertTrue(ready.wait(timeout=3))
        gc_rows = [x for x in received if '"event_id"' in x[1]]
        self.assertTrue(gc_rows)
        self.assertTrue(all(tid != threading.get_native_id() for tid, _ in gc_rows))
        self.assertEqual(writer.dropped, 0)


class TestGCConfiguration(unittest.TestCase):
    def test_disabled_writer_waits_without_polling_or_flushing_gc(self):
        class StopWriter(BaseException):
            pass

        writer = object.__new__(pd._LogWriter)
        writer.queue = NS(get=Mock(side_effect=StopWriter))
        writer._flush_gc = Mock(side_effect=AssertionError)
        with patch.object(pd, "_gc_diagnostics", None):
            with self.assertRaises(StopWriter):
                writer._run()
        writer.queue.get.assert_called_once_with(timeout=None)
        writer._flush_gc.assert_not_called()

    def test_scheduler_switches_enable_enhanced_only_for_prefill_stats(self):
        # Execute the actual startup GC branch without constructing a scheduler
        # or touching torch/NPU. In particular LOG_GC=0 must do no setup at all.
        path = SRT / "managers/scheduler.py"
        tree = ast.parse(path.read_text(encoding="utf8"))
        cls = next(n for n in tree.body if getattr(n, "name", None) == "Scheduler")
        method = next(
            n for n in cls.body
            if getattr(n, "name", None) == "init_watch_dog_memory_saver_input_blocker"
        )
        branch = method.body[-1]
        self.assertIsInstance(branch, ast.If)
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
            str(path), "exec",
        )
        for enabled, mode, stats in (
            (False, "prefill", True), (True, "decode", True),
            (True, "prefill", False), (True, "prefill", True),
        ):
            configure = Mock()
            ns = dict(
                envs=NS(SGLANG_LOG_GC=NS(get=lambda: enabled)),
                get_disagg=lambda: NS(disaggregation_mode=mode),
                self=NS(
                    server_args=NS(enable_request_time_stats_logging=stats),
                    ps=NS(attn_tp_rank=1, attn_dp_rank=0, attn_cp_rank=0, pp_rank=0),
                ),
                configure_gc_logger=configure,
            )
            exec(code, ns)
            if not enabled:
                configure.assert_not_called()
            elif mode == "prefill" and stats:
                self.assertEqual(
                    configure.call_args.kwargs["pd_diagnostics"]["tp_rank"], 1
                )
            else:
                configure.assert_called_once_with(pd_diagnostics=None)

    def test_enabling_initializes_output_and_registers_once(self):
        writer = Mock()
        logger = logging.Logger("test", level=logging.INFO)
        with (
            patch.dict(sys.modules, {
                "sglang.srt.observability.pd_gc_diagnostics": diag_module
            }),
            patch.object(pd, "_ensure_writer", return_value=writer),
            patch.object(pd, "_logger", logger),
            patch.object(pd, "_gc_diagnostics", None),
        ):
            first = pd.enable_gc_diagnostics({"role": "prefill"})
            self.addCleanup(
                lambda: gc.callbacks.remove(first) if first in gc.callbacks else None
            )
            second = pd.enable_gc_diagnostics({"role": "prefill"})
            self.assertEqual(first, second)
            self.assertEqual(gc.callbacks.count(first), 1)
            self.assertEqual(writer.put.call_args.args[0], "PDGCStats ")
            self.assertEqual(writer.put.call_args.args[1]["record"], "gc_enabled")

    def test_legacy_registration_once_and_enhanced_replaces_only_ours(self):
        callbacks = []
        ns = source_functions(
            SRT / "utils/common.py", {"configure_gc_logger"},
            {"gc": NS(callbacks=callbacks), "logger": Mock(), "time": time,
             "_gc_logger_callback": None},
        )
        configure = ns["configure_gc_logger"]
        foreign = lambda *args: None
        callbacks.append(foreign)
        configure()
        configure()
        self.assertEqual(len(callbacks), 2)
        enhanced = lambda *args: None

        def enable(fields):
            callbacks.append(enhanced)
            return enhanced

        fake = NS(enable_gc_diagnostics=enable)
        with patch.dict(sys.modules, {"sglang.srt.observability.pd_time_stats": fake}):
            configure(pd_diagnostics={"role": "prefill"})
        self.assertEqual(callbacks, [foreign, enhanced])


if __name__ == "__main__":
    unittest.main()
