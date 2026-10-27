"""CPU behavior checks for WeLM graph read fences and PD resource lifetime.

Run from the repository root with ``python -m pytest -q`` and this file.
The ACL double checks submission/lifetime rules, not NPU event execution.
"""

from __future__ import annotations

import ast
import copy
import sys
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest

SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load_nodes(path, names, namespace=None, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(
            node.body for node in body if getattr(node, "name", None) == class_name
        )
    nodes = [
        copy.deepcopy(node) for node in body if getattr(node, "name", None) in names
    ]
    assert len(nodes) == len(names)
    module = ModuleType(f"welm_sync_test_{len(sys.modules)}")
    sys.modules[module.__name__] = module
    module.__dict__.update(namespace or {})
    tree = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def make_guard():
    released = []
    module = load_nodes(
        SRT / "disaggregation/deferred_release.py",
        ["_PendingRelease", "DecodeDeferredReleaseGuard"],
        dict(
            deque=deque,
            dataclass=dataclass,
            release_kv_cache=lambda req, cache, is_insert: released.append(
                (req, is_insert)
            ),
        ),
    )
    return module.DecodeDeferredReleaseGuard(object()), released


def request(slot=7):
    return NS(req_pool_idx=slot, kv=object(), finished=lambda: True)


def test_deferred_release_keeps_original_fence_and_insert_flag():
    guard, released = make_guard()
    req = request()
    guard.on_submitted(None)
    assert not guard.defer_if_inflight(req, False)
    guard.on_submitted(3)
    guard.on_submitted(8)
    guard.on_completed(3)
    assert guard.defer_if_inflight(req, False)
    guard.on_submitted(11)
    assert guard.defer_if_inflight(req, True)
    assert list(guard.iter_pending_reqs()) == [req]
    assert not released
    guard.on_completed(8)
    assert released == [(req, False)]
    assert not guard.has_pending
    guard.on_completed(11)
    assert not guard.defer_if_inflight(request(9), True)


@pytest.mark.parametrize("field,replacement", [("req_pool_idx", 8), ("kv", object())])
def test_deferred_release_rejects_changed_ownership(field, replacement):
    guard, released = make_guard()
    req = request()
    guard.on_submitted(1)
    assert guard.defer_if_inflight(req, True)
    setattr(req, field, replacement)
    with pytest.raises(AssertionError):
        guard.on_completed(1)
    assert guard.has_pending and not released


def test_deferred_release_orders_multiple_owners_and_checks_execution_ids():
    guard, released = make_guard()
    a, b = request(1), request(2)
    guard.on_submitted(2)
    guard.defer_if_inflight(a, True)
    guard.on_submitted(9)
    guard.defer_if_inflight(b, False)
    guard.on_completed(None)
    guard.on_completed(2)
    assert released == [(a, True)]
    assert list(guard.iter_pending_reqs()) == [b]
    with pytest.raises(AssertionError):
        guard.on_submitted(9)
    with pytest.raises(AssertionError):
        guard.on_completed(10)
    guard.on_completed(9)
    assert released == [(a, True), (b, False)]


def test_result_ready_rejects_unrecorded_or_delayed_completion():
    module = load_nodes(
        SRT / "disaggregation/decode.py",
        ["_on_pd_decode_result_ready"],
        class_name="SchedulerDisaggregationDecodeMixin",
    )
    completed = []
    owner = NS(pd_deferred_release_guard=NS(on_completed=completed.append))
    result = NS(pd_resource_step=2, copy_done=None, delay_sample_func=None)
    with pytest.raises(AssertionError):
        module._on_pd_decode_result_ready(owner, None, result)
    result.copy_done = object()
    result.delay_sample_func = lambda: None
    with pytest.raises(AssertionError):
        module._on_pd_decode_result_ready(owner, None, result)
    assert completed == []
    result.delay_sample_func = None
    module._on_pd_decode_result_ready(owner, None, result)
    assert completed == [2]


def test_reclaim_drain_retires_pending_requests_and_clears_last_batch():
    module = load_nodes(
        SRT / "disaggregation/decode.py",
        ["_pop_and_process_pd_result", "_drain_pd_inflight_results_for_reclaim"],
        class_name="SchedulerDisaggregationDecodeMixin",
    )
    guard, released = make_guard()
    req = request()
    guard.on_submitted(2)
    guard.defer_if_inflight(req, True)
    guard.on_submitted(4)
    owner = NS(
        pd_deferred_release_guard=guard,
        result_queue=deque([(object(), 2), (object(), 4)]),
        last_batch=object(),
        process_batch_result=lambda batch, step: guard.on_completed(step),
    )
    owner._pop_and_process_pd_result = lambda: module._pop_and_process_pd_result(owner)
    module._drain_pd_inflight_results_for_reclaim(owner)
    assert guard.completed == guard.latest_submitted == 4
    assert not guard.has_pending and not owner.result_queue
    assert owner.last_batch is None and released == [(req, True)]


def test_resource_accounting_deduplicates_batches_and_pending_owners():
    module = load_nodes(
        SRT / "managers/scheduler_components/invariant_checker.py",
        ["_iter_resource_owners"],
        class_name="SchedulerInvariantChecker",
    )
    a, b, c = request(), request(), request()
    owner = NS(
        get_last_batch=lambda: NS(reqs=[a, b]),
        get_running_batch=lambda: NS(reqs=[b]),
        get_pending_release_reqs=lambda: [a, c],
    )
    assert [id(req) for req in module._iter_resource_owners(owner)] == [
        id(a), id(b), id(c)
    ]


@pytest.mark.parametrize("kind", ["idle", "prebuilt"])
def test_result_completion_follows_copy_done_inside_free_group(kind):
    events = []
    method = f"process_batch_result_{kind}"
    module = load_nodes(
        SRT / "managers/scheduler_components/batch_result_processor.py",
        [method],
        dict(
            DisaggregationMode=NS(DECODE="decode"),
            get_disagg=lambda: NS(disaggregation_decode_enable_radix_cache=False),
            get_observability=lambda: NS(enable_request_time_stats_logging=False),
        ),
        class_name="SchedulerBatchResultProcessor",
    )
    owner = NS(
        disaggregation_mode="decode",
        token_to_kv_pool_allocator=NS(
            free_group_begin=lambda: events.append("begin"),
            free_group_end=lambda: events.append("end"),
        ),
        on_decode_result_ready=lambda batch, result: events.append("complete"),
        output_streamer=NS(
            _stream_output_generation=lambda *a, **kw: events.append("output"),
            stream_output=lambda *a, **kw: events.append("output"),
        ),
    )
    batch = NS(reqs=[], return_logprob=False)
    result = NS(
        pd_resource_step=5, copy_done=NS(synchronize=lambda: events.append("copy"))
    )
    getattr(module, method)(owner, batch, result)
    assert events == ["copy", "begin", "complete", "end", "output"]
    if kind == "prebuilt":
        events.clear()
        getattr(module, method)(owner, batch, NS(pd_resource_step=None))
        assert events == ["output"]


@pytest.mark.parametrize("enabled,force", [(True, False), (True, True), (False, False)])
def test_scheduler_consumes_ticket_once_or_uses_coarse_fallback(enabled, force):
    module = load_nodes(
        SRT / "managers/scheduler.py",
        ["_apply_war_barrier"],
        dict(envs=NS(SGLANG_FORCE_COARSE_WAR_BARRIER=NS(get=lambda: force))),
        class_name="Scheduler",
    )
    events = []
    ticket = object()
    runner = NS(war_fastpath_read_done_event=ticket)
    owner = NS(
        _war_barrier_enabled=enabled,
        model_worker=NS(war_fastpath_runner=runner),
        forward_stream=object(),
        schedule_stream=NS(
            wait_event=lambda event: events.append(("early", event)),
            wait_stream=lambda stream: events.append(("coarse", stream)),
        ),
    )
    module._apply_war_barrier(owner)
    assert runner.war_fastpath_read_done_event is None
    expected = (
        []
        if not enabled
        else [("coarse", owner.forward_stream)] if force else [("early", ticket)]
    )
    assert events == expected
    events.clear()
    module._apply_war_barrier(owner)
    assert events == ([("coarse", owner.forward_stream)] if enabled else [])


class FakeACLRuntime:
    def __init__(self):
        self.rt = self
        self.calls = []
        self.states = {}
        self.next_event = 0

    @staticmethod
    def check(ret, operation):
        if ret:
            raise RuntimeError(operation)

    def result(self, result, operation):
        value, ret = result
        self.check(ret, operation)
        return value

    def create_event_ex_with_flag(self, flag):
        assert flag == 3
        self.next_event += 1
        self.states[self.next_event] = 0
        return self.next_event, 0

    def record_event(self, event, stream):
        self.calls.append(("record", event, stream))
        return 0

    def stream_wait_event(self, stream, event):
        self.calls.append(("gate", event, stream))
        return 0

    def query_event_status(self, event):
        return self.states[event], 0

    def destroy_event(self, event):
        self.calls.append(("retire", event))
        return 0

    def create_stream(self):
        return 99, 0

    def write(self, address, value, stream):
        self.calls.append(("write", value, stream))

    def wait(self, address, stream):
        self.calls.append(("signal_wait", stream))


def make_graph_owner():
    stream = NS(npu_stream=12)
    capture = NS(active=False)
    fake_torch = NS(
        int64="int64",
        empty=lambda *a, **kw: NS(data_ptr=lambda: 4096),
        npu=NS(
            is_current_stream_capturing=lambda: capture.active,
            current_stream=lambda: stream,
        ),
    )
    module = load_nodes(
        SRT / "hardware_backend/npu/graph_runner/graph_read_done.py",
        ["NpuReadDoneTicket", "_Replay", "NpuGraphReadDone"],
        dict(
            deque=deque,
            dataclass=dataclass,
            contextmanager=contextmanager,
            torch=fake_torch,
            _ACLRuntime=FakeACLRuntime,
            _graph_lifetime_owners=[],
        ),
    )
    return module.NpuGraphReadDone("npu"), capture, stream


def capture_owner(owner, capture):
    owner.begin_capture_forward()
    owner.signal_once()
    assert owner._capture_signals == 0  # Warmup cannot publish a graph signal.
    capture.active = True
    owner.begin_capture_forward()
    owner.signal_once()
    owner.signal_once()
    assert owner._capture_signals == 1
    capture.active = False
    owner.after_capture()


def test_graph_notifications_use_fresh_tickets_and_retire_after_next_gate():
    owner, capture, stream = make_graph_owner()
    capture_owner(owner, capture)
    runtime = owner._runtime
    initial = owner._last_gate.event
    with owner.replay(stream) as first:
        assert not first.armed
        assert not first.query()
        runtime.calls.append(("graph", 1))
    first_id = first.event
    assert first.armed and first_id != initial
    assert runtime.calls[-3:] == [
        ("signal_wait", 99), ("write", 0, 99), ("record", first_id, 99)
    ]
    first.wait(stream)  # Scheduler consumption precedes next submission.
    runtime.states[first_id] = 1
    owner._reap()
    assert ("retire", initial) in runtime.calls
    assert first.event == first_id
    with owner.replay(stream) as second:
        assert second.event != first_id
        runtime.calls.append(("graph", 2))
    assert first.event == first_id  # The second event is still incomplete.
    runtime.states[second.event] = 1
    owner._reap()
    assert first.event is None
    assert second.event is not None
    with pytest.raises(RuntimeError, match="retired"):
        first.wait(stream)


def test_graph_partial_submission_fails_closed():
    owner, capture, stream = make_graph_owner()
    capture_owner(owner, capture)
    with pytest.raises(RuntimeError, match="graph submission"):
        with owner.replay(stream):
            raise RuntimeError("graph submission")
    assert owner._failed and owner._pending is not None
    with pytest.raises(RuntimeError, match="cannot be reused"):
        with owner.replay(stream):
            pass


def test_graph_capture_without_signal_is_rejected():
    owner, capture, _ = make_graph_owner()
    capture.active = True
    owner.begin_capture_forward()
    capture.active = False
    with pytest.raises(RuntimeError, match="exactly one"):
        owner.after_capture()
    assert owner._failed
