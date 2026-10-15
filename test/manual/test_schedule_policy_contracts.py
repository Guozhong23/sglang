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
from contextlib import contextmanager, nullcontext
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


class TestShortestPrefillFirst(unittest.TestCase):
    def test_uncached_work_includes_retracted_output_and_minimum_one(self):
        cached = req("cached", 100, matched=90)
        replay = req("replay", 20, matched=10)
        replay.output_ids = [0] * 50
        zero = req("zero", 10, matched=10)
        queue = [replay, cached, zero]
        SchedulePolicy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(ids(queue), ["zero", "cached", "replay"])
        self.assertEqual(
            [SchedulePolicy._shortest_prefill_work(r) for r in queue], [1, 10, 60]
        )

    def test_equal_work_uses_arrival_then_stable_input_order(self):
        older, newer, equal = req("z", 32), req("a", 32), req("b", 32)
        older.time_stats.wait_queue_entry_time = 1
        newer.time_stats.wait_queue_entry_time = 2
        equal.time_stats.wait_queue_entry_time = 2
        queue = [newer, equal, older]
        SchedulePolicy._sort_by_shortest_prefill(queue, set())
        self.assertEqual(queue, [older, newer, equal])

    def test_calc_priority_matches_cache_and_deprioritizes_duplicate(self):
        cache = Cache({(1,) * 100: (80, 10)})
        queue = [req("long", 90, token=2), req("cached", 100)]
        policy("shortest-prefill-first", cache).calc_priority(queue)
        self.assertEqual(ids(queue), ["cached", "long"])
        duplicate = req("duplicate", 1)
        queue.append(duplicate)
        SchedulePolicy._sort_by_shortest_prefill(queue, {"duplicate"})
        self.assertIs(queue[-1], duplicate)

    def test_cache_disabled_rejected_and_large_queue_still_sorted(self):
        with self.assertRaises(ValueError):
            policy("shortest-prefill-first", Cache(disable=True))
        queue = [req(f"{i:03}", i + 1, token=i + 1) for i in reversed(range(129))]
        policy("shortest-prefill-first").calc_priority(queue)
        self.assertEqual(ids(queue), sorted(ids(queue)))

    def test_page128_reservation_and_continuation_progress(self):
        p = policy("shortest-prefill-first")
        continuation = admission_req("continuation", 33024)
        for work, expected in ((1, 16384), (129, 16256), (16384, 128)):
            with self.subTest(work=work):
                self.assertEqual(
                    p.shortest_prefill_chunk_limit(
                        continuation, [req("short", work)], 16512, 128
                    ),
                    expected,
                )
        self.assertEqual(
            p.shortest_prefill_chunk_limit(
                continuation, [req("a", 512), req("b", 1024)], 16512, 128
            ),
            14976,
        )

    def test_reservation_stops_at_first_nonfitting_candidate(self):
        p = policy("shortest-prefill-first")
        continuation = admission_req("continuation", 33024)
        # The later one-token candidate must not be scanned past the blocker.
        queue = [req("first", 128), req("blocker", 16384), req("later", 1)]
        self.assertEqual(
            p.shortest_prefill_chunk_limit(continuation, queue, 16512, 128), 16384
        )

    def test_no_reservation_for_small_budget_or_nonshorter_work(self):
        continuation = admission_req("continuation", 1024)
        for name, budget, waiting in (
            ("hrrn", 16512, [req("short", 1)]),
            ("shortest-prefill-first", 128, [req("short", 1)]),
            ("shortest-prefill-first", 16512, [req("same", 1024)]),
            ("shortest-prefill-first", 16512, []),
        ):
            with self.subTest(name=name, budget=budget):
                self.assertIsNone(
                    policy(name).shortest_prefill_chunk_limit(
                        continuation, waiting, budget, 128
                    )
                )


class HybridAllocator:
    def available_size(self):
        return 1_000_000

    full_available_size = available_size
    swa_available_size = available_size


class AdmissionCache(Cache):
    def __init__(self):
        super().__init__()
        self.locks = 0
        self.load_calls = 0
        self.host_loaded = 0
        self.sliding_window_size = 4096

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return False

    def evictable_size(self):
        return 0

    full_evictable_size = evictable_size
    swa_evictable_size = evictable_size

    def inc_lock_ref(self, node):
        self.locks += 1
        return NS(swa_uuid_for_lock="swa")

    def dec_lock_ref(self, node):
        self.locks -= 1

    def init_load_back(self, params):
        self.load_calls += 1
        return list(range(self.host_loaded)), "loaded-node"


SCHEDULE = NS(schedule_policy="shortest-prefill-first")
ADMISSION = load_nodes(
    SRT / "managers/schedule_policy.py",
    ["AddReqResult", "PrefillAdder"],
    {
        "Enum": Enum,
        "auto": auto,
        "contextmanager": contextmanager,
        "CLIP_MAX_NEW_TOKENS": 4096,
        "InitLoadBackParams": NS,
        "SWATokenToKVPoolAllocator": HybridAllocator,
        "DeepSeekV4HiSparseTokenToKVPoolAllocator": type("HiSparse", (), {}),
        "PureSWATokenToKVPoolAllocator": type("PureSWA", (), {}),
        "UnifiedMambaTokenToKVPoolAllocator": type("UnifiedMamba", (), {}),
        "is_dsa_prefill_cp_in_seq_split": lambda: False,
        "is_prefill_context_parallel_enabled": lambda: False,
        "get_schedule": lambda: SCHEDULE,
        "torch": NS(cat=lambda values: sum(values, [])),
    },
)
PrefillAdder = ADMISSION["PrefillAdder"]
AddReqResult = ADMISSION["AddReqResult"]


def admission_req(rid, length, *, host=0):
    candidate = req(rid, length)
    candidate.full_untruncated_fill_ids = candidate.origin_input_ids
    candidate.sampling_params = NS(max_new_tokens=1, ignore_eos=False)
    candidate.last_node = "node"
    candidate.best_match_node = "best-node"
    candidate.retracted_stain = False
    candidate.mamba_pool_idx = None
    candidate.swa_host_hit_length = 0
    candidate.host_hit_length = host
    candidate.needs_host_load_back = lambda: host > 0
    candidate.set_extend_range = lambda start, end: setattr(
        candidate, "extend_range", NS(start=start, end=end, length=end - start)
    )
    return candidate


def make_adder(
    *, budget=16512, mixed=0, mixed_kv=None, available=1_000_000, hybrid=False
):
    cache = AdmissionCache()
    allocator = HybridAllocator() if hybrid else NS(available_size=lambda: available)
    adder = PrefillAdder(
        page_size=128,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        running_batch=NS(reqs=[], batch_size=lambda: 0),
        new_token_ratio=1,
        rem_input_tokens=1_000_000,
        rem_chunk_tokens=budget,
        num_mixed_decode_tokens=mixed,
        mixed_decode_kv_tokens=mixed_kv,
    )
    return adder


class TestSPFAdmission(unittest.TestCase):
    def setUp(self):
        SCHEDULE.schedule_policy = "shortest-prefill-first"

    def add(self, adder, candidate, active=True):
        return adder.add_one_req(
            candidate, has_chunked_req=active, truncation_align_size=None
        )

    def test_continuation_and_completed_short_requests_share_batch(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                adder = make_adder(hybrid=hybrid)
                continuation = admission_req("continuation", 33024)
                waiting = [admission_req("a", 512), admission_req("b", 1024)]
                adder.chunked_req_limit = policy(
                    "shortest-prefill-first"
                ).shortest_prefill_chunk_limit(
                    continuation, waiting, adder.rem_chunk_tokens, 128
                )
                self.assertIs(adder.add_chunked_req(continuation), continuation)
                self.assertEqual(continuation.extend_range.length, 14976)
                for candidate in waiting:
                    self.add(adder, candidate)
                self.assertEqual(adder.can_run_list, [continuation, *waiting])
                self.assertIsNone(adder.new_chunked_req)
                self.assertEqual(adder.rem_chunk_tokens, 0)
                self.assertEqual(adder.tree_cache.locks, 2)
                self.assertGreater(adder.rem_total_tokens, 0)

    def test_mixed_compute_and_kv_budgets_remain_distinct(self):
        adder = make_adder(mixed=1, mixed_kv=128)
        self.assertEqual(adder.rem_chunk_tokens, 16511)
        self.assertEqual(adder.rem_total_tokens, 1_000_000 - 128)
        self.assertEqual(adder.rem_swa_token_offset, 128)
        continuation = admission_req("old", 33024)
        limit = policy("shortest-prefill-first").shortest_prefill_chunk_limit(
            continuation, [req("short", 128)], adder.rem_chunk_tokens, 128
        )
        self.assertEqual(limit, 16256)
        adder.chunked_req_limit = limit
        adder.add_chunked_req(continuation)
        self.assertEqual(adder.rem_chunk_tokens, 255)

    def test_second_unfinished_chunk_rejected_before_load_or_delayer(self):
        adder = make_adder(budget=512)
        candidate = admission_req("second", 1024, host=128)
        calls = []
        adder.prefill_delayer_single_pass = NS(
            negotiate_should_allow_prefill=lambda **kwargs: calls.append(kwargs) or True
        )
        self.assertEqual(self.add(adder, candidate), AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [])
        self.assertFalse(hasattr(candidate, "extend_range"))
        self.assertEqual(adder.tree_cache.load_calls, 0)
        self.assertEqual(adder.tree_cache.locks, 0)
        self.assertEqual(calls, [])

    def test_host_miss_rechecked_and_temporary_lock_released(self):
        adder = make_adder(budget=512)
        candidate = admission_req("miss", 1024, host=768)
        self.assertEqual(self.add(adder, candidate), AddReqResult.OTHER)
        self.assertEqual(adder.tree_cache.load_calls, 1)
        self.assertEqual(adder.tree_cache.locks, 0)
        self.assertFalse(hasattr(candidate, "extend_range"))
        self.assertEqual(adder.can_run_list, [])
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(adder.rem_chunk_tokens, 512)

    def test_successful_host_load_can_complete_next_to_old_chunk(self):
        adder = make_adder(budget=512)
        adder.tree_cache.host_loaded = 768
        candidate = admission_req("hit", 1024, host=768)
        self.add(adder, candidate)
        self.assertEqual(candidate.extend_range.length, 256)
        self.assertEqual(adder.can_run_list, [candidate])
        self.assertIsNone(adder.new_chunked_req)
        self.assertEqual(adder.tree_cache.locks, 1)

    def test_kv_and_request_count_admission_gates_are_preserved(self):
        adder = make_adder(available=256)
        self.assertEqual(
            self.add(adder, admission_req("memory", 512)), AddReqResult.NO_TOKEN
        )
        self.assertEqual(adder.tree_cache.locks, 0)
        adder = make_adder()
        adder.prefill_max_requests = 0
        self.assertEqual(
            self.add(adder, admission_req("slots", 512)), AddReqResult.OTHER
        )
        self.assertEqual(adder.can_run_list, [])

    def test_old_chunk_finishes_then_new_chunk_is_allowed(self):
        adder = make_adder()
        old = adder.add_chunked_req(admission_req("old", 512))
        self.assertIsNone(old)
        candidate = admission_req("new", 33024)
        self.add(adder, candidate, active=old is not None)
        self.assertIs(adder.new_chunked_req, candidate)
        self.assertEqual(candidate.extend_range.length, 16000)

    def test_fcfs_keeps_its_previous_admission_behavior(self):
        SCHEDULE.schedule_policy = "fcfs"
        adder = make_adder(budget=512)
        candidate = admission_req("fcfs", 1024)
        self.add(adder, candidate)
        self.assertIs(adder.new_chunked_req, candidate)
        self.assertEqual(candidate.extend_range.length, 512)


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
