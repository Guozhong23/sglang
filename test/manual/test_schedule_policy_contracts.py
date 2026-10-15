"""Host-side scheduling contracts, runnable without torch or accelerator hardware.

Execute unchanged production functions/classes extracted with AST. Only imported
runtime dependencies and resources are replaced with small CPU fixtures. This
does not exercise distributed execution, kernels, or graph replay.

Run: python test/manual/test_schedule_policy_contracts.py
"""

import ast
import logging
import time
import unittest
from array import array
from collections import Counter, defaultdict
from contextlib import nullcontext
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Union


ROOT = Path(__file__).resolve().parents[2]
SRT = ROOT / "python/sglang/srt"


def load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if class_name:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class Cache:
    """Prefix-match fixture with device/host lengths and a simulated waiting tree."""

    def __init__(self, matches=None, disable=False, tail=0):
        self.matches = matches or {}
        self.disable = disable
        self.tail = tail
        self.calls = []
        self.inserted = []

    @staticmethod
    def create_simulated():
        return Cache()

    def reset(self):
        self.inserted.clear()

    def supports_fast_match_prefix(self):
        return False

    def swa_reprefill_tail_tokens(self):
        return self.tail

    def match_prefix(self, params):
        self.calls.append(params)
        key = tuple(params.key.token_ids)
        device, host = self.matches.get(key, (0, 0))
        if key in self.inserted:
            device = len(key)
        limit = getattr(params.key, "limit", None)
        if limit is not None:
            device = min(device, limit)
            host = min(host, max(0, limit - device))
        return NS(
            device_indices=list(range(device)),
            last_device_node="device",
            last_host_node="host",
            best_match_node="best",
            host_hit_length=host,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
        )

    def insert(self, params):
        self.inserted.append(tuple(params.key.token_ids))


POLICY = load_nodes(
    SRT / "managers/schedule_policy.py",
    [
        "match_prefix_for_req",
        "CacheAwarePolicy",
        "CacheAgnosticPolicy",
        "SchedulePolicy",
    ],
    {
        "Enum": Enum,
        "auto": auto,
        "Union": Union,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "RadixCache": Cache,
        "RadixKey": NS,
        "MatchPrefixParams": NS,
        "InsertParams": NS,
        "torch": NS(empty=lambda n, **kw: [False] * n, bool=bool),
        "envs": NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
        "IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD": 32,
        "IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD": 32,
        "get_disagg": lambda: NS(disaggregation_mode="prefill"),
        "logger": logging.getLogger(__name__),
    },
)
SchedulePolicy = POLICY["SchedulePolicy"]


class TimeStats:
    def __init__(self):
        self.entries = 0
        self.retries = 0
        self.wait_queue_entry_time = 0

    def set_wait_queue_entry_time(self):
        self.entries += 1

    def reset_prefill_retry_time(self):
        self.retries += 1


def req(rid, length, arrival=0, matched=0, token=1):
    return NS(
        rid=rid,
        origin_input_ids=[token] * length,
        output_ids=[],
        extra_key=None,
        num_matched_prefix_tokens=matched,
        arrival_processed_tokens=arrival,
        prefix_indices=[],
        host_hit_length=0,
        time_stats=TimeStats(),
        _compute_max_prefix_len=lambda n: n,
    )


def policy(name="hrrn", cache=None):
    return SchedulePolicy(name, cache or Cache(), False, False, False)


def ids(queue):
    return [r.rid for r in queue]


class TestHRRN(unittest.TestCase):
    def test_aging_overtakes_new_short_request(self):
        queue = [req("short", 1, arrival=1000), req("old", 100)]
        policy().calc_priority(queue, processed_tokens=1000)
        self.assertEqual(ids(queue), ["old", "short"])

    def test_zero_work_duplicate_deprioritization_and_rid_ties(self):
        queue = [
            req("z", 10),
            req("a", 10),
            req("zero", 5, matched=8),
            req("duplicate", 1),
        ]
        SchedulePolicy._sort_by_hrrn(queue, {"duplicate"}, 100)
        self.assertEqual(ids(queue), ["zero", "a", "z", "duplicate"])

    def test_zero_or_negative_wait_uses_rid_tie_break(self):
        queue = [req("z", 1, arrival=100), req("a", 100, arrival=110)]
        SchedulePolicy._sort_by_hrrn(queue, set(), 100)
        self.assertEqual(ids(queue), ["a", "z"])

    def test_real_prefix_match_includes_device_and_host_cache(self):
        cache = Cache({(1,) * 100: (60, 30)})
        queue = [req("uncached", 100, token=2), req("cached", 100)]
        policy(cache=cache).calc_priority(queue, processed_tokens=1000)
        self.assertEqual(ids(queue), ["cached", "uncached"])
        self.assertEqual(queue[0].num_matched_prefix_tokens, 90)
        self.assertEqual(queue[0].last_host_node, "host")
        self.assertIs(cache.calls[1].req, queue[0])

    def test_local_swa_reprefill_tail_is_preserved(self):
        cache = Cache({(1,) * 100: (100, 0)}, tail=40)
        queue = [req("swa", 100)]
        policy(cache=cache).calc_priority(queue, processed_tokens=100)
        self.assertEqual(queue[0].num_matched_prefix_tokens, 60)
        self.assertEqual(cache.calls[0].key.limit, 60)

    def test_in_batch_duplicate_is_deprioritized(self):
        queue = [req("z_first", 40), req("a_duplicate", 40)]
        policy().calc_priority(queue, processed_tokens=100)
        self.assertEqual(ids(queue), ["z_first", "a_duplicate"])

    def test_128_sorts_but_129_falls_back_to_fcfs(self):
        for count in (128, 129):
            with self.subTest(count=count):
                queue = [
                    req(f"{i:03}", i + 1, token=i + 1) for i in reversed(range(count))
                ]
                original = ids(queue)
                cache = Cache()
                policy(cache=cache).calc_priority(queue, processed_tokens=1000)
                self.assertEqual(
                    ids(queue), sorted(original) if count == 128 else original
                )
                self.assertEqual(len(cache.calls), 128 if count == 128 else 0)

    def test_disabled_cache_and_default_fcfs_preserve_order(self):
        for name, cache in (("hrrn", Cache(disable=True)), ("fcfs", Cache())):
            with self.subTest(name=name):
                queue = [req("z", 100), req("a", 1)]
                policy(name, cache).calc_priority(queue, processed_tokens=1000)
                self.assertEqual(ids(queue), ["z", "a"])


DisaggregationMode = NS(NULL="null", PREFILL="prefill", DECODE="decode")
SCHEDULER = load_nodes(
    SRT / "managers/scheduler.py",
    ["_add_request_to_queue", "run_batch"],
    {
        "DisaggregationMode": DisaggregationMode,
        "time": time,
        "scheduler_nvtx_method": lambda name: lambda fn: fn,
        "resolve_forward_inputs": lambda batch, future_map: None,
        "logger": logging.getLogger(__name__),
        "_is_hip": False,
    },
    class_name="Scheduler",
)
KVPoll = NS(Failed="failed", Bootstrapping="bootstrapping", WaitingForInput="ready")
BOOTSTRAP = load_nodes(
    SRT / "disaggregation/prefill.py",
    ["pop_bootstrapped"],
    {
        "KVPoll": KVPoll,
        "should_force_retry": lambda req: False,
        "poll_and_all_reduce_attn_cp_tp_group": lambda senders, *groups: senders,
    },
    class_name="PrefillBootstrapQueue",
)
RETRY = load_nodes(
    SRT / "disaggregation/prefill.py",
    ["optimistic_release_and_requeue"],
    {
        "array": array,
        "logger": logging.getLogger(__name__),
        "get_disagg": lambda: NS(optimistic_prefill_attempts=3),
        "maybe_cache_unfinished_req": lambda req, cache: None,
        "release_kv_cache": lambda req, cache: None,
    },
    class_name="SchedulerDisaggregationPrefillMixin",
)


class TestQueueAndLaunchAccounting(unittest.TestCase):
    def test_unified_queue_snapshot_and_rejected_request(self):
        scheduler = NS(
            disaggregation_mode=DisaggregationMode.NULL,
            processed_tokens_counter=345,
            waiting_queue=[],
            _set_or_validate_priority=lambda req: True,
            _abort_on_queued_limit=lambda req: req.rid == "rejected",
            _prefetch_kvcache=lambda req: None,
        )
        accepted, rejected = req("accepted", 10), req("rejected", 10)
        for candidate in (accepted, rejected):
            SCHEDULER["_add_request_to_queue"](scheduler, candidate)
        self.assertEqual(scheduler.waiting_queue, [accepted])
        self.assertEqual(accepted.arrival_processed_tokens, 345)
        self.assertEqual(accepted.time_stats.entries, 1)
        self.assertEqual(rejected.arrival_processed_tokens, 0)

    def test_both_pd_bootstrap_success_branches_snapshot_on_entry(self):
        requests = [req("optimistic", 10), req("ready", 10)]
        for candidate, poll in zip(
            requests, (KVPoll.Bootstrapping, KVPoll.WaitingForInput)
        ):
            candidate.disagg_kv_sender = poll
            candidate.prefill_attempt_count = 0
            candidate.is_retracted = False
        scheduler = NS(
            processed_tokens_counter=678,
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            server_args=NS(optimistic_prefill_attempts=3),
        )
        queue = NS(
            queue=list(requests),
            pp_size=1,
            scheduler=scheduler,
            ensure_metadata_buffer=lambda req: True,
            finalize_bootstrap=lambda req: True,
        )
        self.assertEqual(BOOTSTRAP["pop_bootstrapped"](queue), requests)
        self.assertEqual(queue.queue, [])
        self.assertEqual([r.arrival_processed_tokens for r in requests], [678, 678])
        self.assertEqual([r.time_stats.entries for r in requests], [1, 1])

    def test_optimistic_retry_refreshes_snapshot_only_when_requeued(self):
        for attempts in (1, 3):
            with self.subTest(attempts=attempts):
                candidate = req("retry", 10, arrival=42)
                candidate.prefill_attempt_count = attempts
                candidate.reset_for_retract = lambda: None
                scheduler = NS(
                    processed_tokens_counter=999,
                    tree_cache=None,
                    waiting_queue=[],
                    disagg_prefill_bootstrap_queue=NS(queue=[]),
                    metrics_reporter=NS(enable_metrics=False),
                )
                RETRY["optimistic_release_and_requeue"](scheduler, candidate)
                self.assertEqual(
                    candidate.arrival_processed_tokens, 999 if attempts == 1 else 42
                )
                self.assertEqual(
                    scheduler.waiting_queue, [candidate] if attempts == 1 else []
                )
                self.assertEqual(
                    candidate.time_stats.entries, 1 if attempts == 1 else 0
                )

    def test_dedicated_prefill_launch_counts_once_in_normal_and_overlap(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                observations = []
                result = NS(
                    has_sampled_token_ids=False,
                    extra_keep_alive_refs=[],
                    delay_sample_func=None,
                    copy_to_cpu=lambda **kwargs: None,
                )
                stream = NS(wait_stream=lambda other: None)
                non_spec = NS(is_none=lambda: True)
                scheduler = NS(
                    forward_ct=0,
                    processed_tokens_counter=100,
                    scripted_scheduler_hook=None,
                    profiler_manager=NS(_profile_batch_predicate=lambda batch: None),
                    forward_sleep_time=None,
                    disaggregation_mode=DisaggregationMode.PREFILL,
                    maybe_send_cached_prefix_chunk=lambda req: None,
                    is_generation=True,
                    enable_overlap=overlap,
                    enable_pdmux=False,
                    spec_algorithm=non_spec,
                    future_map=NS(
                        resolve_seq_lens_cpu=lambda batch: None,
                        publish=lambda *args: None,
                    ),
                    _confidence_budget_prepare=None,
                    forward_stream_ctx=nullcontext(),
                    forward_stream=stream,
                    schedule_stream=stream,
                    copy_stream=stream,
                    copy_stream_ctx=nullcontext(),
                    _forward_isolation=lambda *args, **kwargs: nullcontext(),
                    enable_unified_memory=False,
                    device_module=NS(Event=lambda: object()),
                    _relay_forward_payload=lambda *args: None,
                    update_cache_from_scheduler=lambda *args: None,
                    _maybe_report_active_ranks=lambda: None,
                )

                def forward(batch, **kwargs):
                    observations.append(scheduler.processed_tokens_counter)
                    return result

                scheduler.model_worker = NS(forward_batch_generation=forward)
                batch = NS(
                    extend_num_tokens=16512,
                    forward_mode=NS(is_prebuilt=lambda: False),
                    reqs=[req("p", 33024)],
                    spec_algorithm=non_spec,
                    req_pool_indices=[0],
                    seq_lens=16512,
                    return_logprob=False,
                    return_hidden_states=False,
                )
                for _ in range(2):
                    self.assertIs(SCHEDULER["run_batch"](scheduler, batch), result)
                self.assertEqual(observations, [16612, 33124])
                self.assertEqual(scheduler.processed_tokens_counter, 33124)
                self.assertEqual(scheduler.forward_ct, 2)
                self.assertEqual(batch.forward_iter, 2)


if __name__ == "__main__":
    unittest.main()
