"""Opt-in CPython GC attribution for PD send preparation (no device access).

The callback never logs or takes an application lock. The PD log writer drains
completed events. A bounded deque is used on GIL-enabled CPython; this collector
does not claim to support free-threaded interpreters.
"""

from __future__ import annotations

import gc
import os
import socket
import sys
import threading
import time
from collections import deque

DETAIL_THRESHOLD_NS = 10_000_000
BUFFER_SIZE = 2048
MAX_FRAMES = 6


class Preparation:
    def __init__(self, fields, previous):
        self.previous = previous
        self.native_tid = threading.get_native_id()
        self.fields = {
            key: fields.get(key)
            for key in ("bootstrap_room", "rid", "chunk_seq", "queue_id")
        }
        self.phase = "probe_setup"
        self.start_ns = self.end_ns = None
        self.start_cpu_ns = self.end_cpu_ns = None
        self.engine_calls = 0
        # All generations, including events too short to log individually.
        self.gc_count = [0, 0, 0]
        self.gc_wall_ns = [0, 0, 0]
        self.gc_cpu_ns = [0, 0, 0]

    def start(self, start_ns):
        self.start_ns = start_ns
        self.start_cpu_ns = time.thread_time_ns()
        self.phase = "pre_engine"

    def enter_engine(self, start_ns):
        self.phase = "engine"
        self.engine_calls += 1
        if self.engine_calls == 1:
            self.end_ns = start_ns
            self.end_cpu_ns = time.thread_time_ns()

    def record(self):
        supported = self.engine_calls == 1 and self.end_ns is not None
        return dict(
            self.fields,
            native_tid=self.native_tid,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            start_thread_cpu_ns=self.start_cpu_ns,
            end_thread_cpu_ns=self.end_cpu_ns,
            wall_ms=(self.end_ns - self.start_ns) / 1e6 if supported else None,
            thread_cpu_ms=(self.end_cpu_ns - self.start_cpu_ns) / 1e6
            if supported
            else None,
            same_thread_engine_calls=self.engine_calls,
            supported=supported,
            unsupported_reason=None if supported else "not_one_same_thread_engine",
            gc_count_by_generation=self.gc_count.copy(),
            gc_wall_ns_by_generation=self.gc_wall_ns.copy(),
            gc_thread_cpu_ns_by_generation=self.gc_cpu_ns.copy(),
        )


class PDGCDiagnostics:
    def __init__(self, fields, capacity=BUFFER_SIZE):
        self.pid = os.getpid()
        mono = time.perf_counter_ns()
        self.identity = dict(
            fields,
            schema=1,
            record="gc",
            pid=self.pid,
            host=socket.gethostname(),
            diagnostic_id=f"{self.pid}:{mono}",
            anchor_mono_ns=mono,
            anchor_wall_ns=time.time_ns(),
            python_version=sys.version.split()[0],
            detail_threshold_ns=DETAIL_THRESHOLD_NS,
        )
        self.local = threading.local()
        self.completed = deque(maxlen=capacity)
        self.pending = None
        self.started = self.finished = self.dropped = self.errors = 0
        self.unpaired = self.filtered = 0
        self.count = [0, 0, 0]
        self.wall_ns = [0, 0, 0]
        self.cpu_ns = [0, 0, 0]
        self.callback = self._callback

    def install(self):
        is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
        if not is_gil_enabled():
            raise RuntimeError("PD GC diagnostics requires GIL-enabled CPython")
        if self.callback not in gc.callbacks:
            gc.callbacks.append(self.callback)

    def uninstall(self):
        if self.callback in gc.callbacks:
            gc.callbacks.remove(self.callback)

    def enter(self, fields):
        scope = Preparation(fields, getattr(self.local, "scope", None))
        self.local.scope = scope
        return scope

    def scope(self):
        return getattr(self.local, "scope", None)

    def leave(self, scope):
        self.local.scope = scope.previous
        scope.previous = None

    @staticmethod
    def _callsite():
        # Skip this helper and _callback; keep probe/writer frames visible so
        # allocations made by instrumentation cannot masquerade as model work.
        frame = sys._getframe(2)
        frames = []
        try:
            for _ in range(MAX_FRAMES):
                if frame is None:
                    break
                code = frame.f_code
                frames.append((code.co_filename, code.co_name, frame.f_lineno))
                frame = frame.f_back
            return tuple(frames)
        finally:
            del frame

    def _callback(self, phase, info):
        # Registered only in the scheduler process. Never acquire the writer's
        # queue mutex here: GC can run while that mutex is already held.
        try:
            if os.getpid() != self.pid:
                return
            if phase == "start":
                start = time.perf_counter_ns()
                cpu = time.thread_time_ns()
                tid = threading.get_native_id()
                scope = self.scope()
                stage = scope.phase if scope is not None else "outside_send"
                self.started += 1
                if self.pending is not None:
                    self.unpaired += 1
                self.pending = (
                    self.started,
                    info["generation"],
                    tid,
                    start,
                    cpu,
                    scope,
                    stage,
                    self._callsite(),
                )
            elif phase == "stop":
                end = time.perf_counter_ns()
                cpu_end = time.thread_time_ns()
                pending, self.pending = self.pending, None
                if pending is None:
                    self.unpaired += 1
                    return
                event_id, gen, tid, start, cpu, scope, stage, frames = pending
                if gen != info["generation"] or tid != threading.get_native_id():
                    self.unpaired += 1
                    return
                self.finished += 1
                wall, cpu_used = end - start, cpu_end - cpu
                self.count[gen] += 1
                self.wall_ns[gen] += wall
                self.cpu_ns[gen] += cpu_used
                if scope is not None and stage == "pre_engine":
                    scope.gc_count[gen] += 1
                    scope.gc_wall_ns[gen] += wall
                    scope.gc_cpu_ns[gen] += cpu_used
                if gen != 2 and wall < DETAIL_THRESHOLD_NS:
                    self.filtered += 1
                    return
                if len(self.completed) == self.completed.maxlen:
                    self.dropped += 1
                self.completed.append(
                    (
                        event_id,
                        gen,
                        tid,
                        start,
                        end,
                        cpu,
                        cpu_end,
                        scope.fields if scope is not None else None,
                        stage,
                        frames,
                        info.get("collected", 0),
                        info.get("uncollectable", 0),
                    )
                )
        except Exception:
            # Callback failures must not affect inference or silently imply no GC.
            self.errors += 1
            self.pending = None

    def status(self, record="gc_status"):
        return dict(
            self.identity,
            record=record,
            observed_ns=time.perf_counter_ns(),
            gc_started=self.started,
            gc_completed=self.finished,
            gc_filtered_short=self.filtered,
            gc_buffer_dropped=self.dropped,
            gc_callback_errors=self.errors,
            gc_unpaired=self.unpaired,
            gc_pending=self.pending is not None,
            gc_buffered_events=len(self.completed),
            gc_count_by_generation=self.count.copy(),
            gc_wall_ns_by_generation=self.wall_ns.copy(),
            gc_thread_cpu_ns_by_generation=self.cpu_ns.copy(),
        )

    def drain(self, limit=128):
        """Called by the log writer only; formatting is outside the callback."""
        for _ in range(limit):
            try:
                event = self.completed.popleft()
            except IndexError:
                break
            (
                event_id,
                gen,
                tid,
                start,
                end,
                cpu,
                cpu_end,
                fields,
                stage,
                frames,
                collected,
                uncollectable,
            ) = event
            yield dict(
                self.identity,
                **(fields or {}),
                event_id=event_id,
                generation=gen,
                native_tid=tid,
                phase=stage,
                start_ns=start,
                end_ns=end,
                start_thread_cpu_ns=cpu,
                end_thread_cpu_ns=cpu_end,
                wall_ms=(end - start) / 1e6,
                thread_cpu_ms=(cpu_end - cpu) / 1e6,
                callsite=[dict(file=f, function=n, line=line) for f, n, line in frames],
                collected=collected,
                uncollectable=uncollectable,
                gc_buffer_dropped=self.dropped,
                gc_callback_errors=self.errors,
            )
