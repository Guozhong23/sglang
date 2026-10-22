"""CPU tests of production DP graph admission and fixed-address row metadata.

Four real Gloo ranks exercise the production pre-dispatch method, including
one-rank rejection and differing graph keys. This does not validate NPU capture.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import importlib.util
import logging
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"
CONTRACT_NAME = "sglang.srt.model_executor.runner.welm_dp_graph_contract"
spec = importlib.util.spec_from_file_location(
    CONTRACT_NAME, SRT / "model_executor/runner/welm_dp_graph_contract.py"
)
contract = importlib.util.module_from_spec(spec)
sys.modules[CONTRACT_NAME] = contract
spec.loader.exec_module(contract)


def load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = (
        tree.body
        if class_name is None
        else next(n for n in tree.body if getattr(n, "name", "") == class_name).body
    )
    nodes = [copy.deepcopy(n) for n in body if getattr(n, "name", "") in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


RUNNER = load_nodes(
    SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
    ["prepare_welm_dp_graph_ticket", "_validate_welm_dp_manifest", "can_run_graph"],
    {"torch": torch, "logger": logging.getLogger(__name__)},
    class_name="PrefillCudaGraphRunner",
)


def _gloo_admission_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        dp_rank = rank // 2
        group = NS(world_size=4, cpu_group=dist.group.WORLD, ranks=list(range(4)))
        adapter = NS(
            is_dp=True,
            dp_rank=dp_rank,
            attn_tp_size=2,
            batch_sizes=(1, 2, 4, 8),
            manifest=lambda: {"kind": "test", "kernel_variant": "ordinary"},
        )
        runner = NS(
            welm_adapter=adapter,
            model_runner=NS(tp_group=group),
            capture_num_tokens=(64, 128, 256),
            enable_lora=False,
            can_replay_locally=lambda **kw: True,
        )
        with torch.inference_mode():
            RUNNER["_validate_welm_dp_manifest"](runner)
        cases = [
            "valid",
            "rank_reject",
            "bucket_disagree",
            "counts_disagree",
            "idle",
            "decode",
            "valid_again",
        ]
        for case in cases:
            tokens, requests = (123, 61), (3, 1)
            candidate = True
            runner.capture_num_tokens = (64, 128, 256)
            if case == "idle":
                tokens, requests, candidate = (123, 0), (3, 0), False
            if case == "decode":
                tokens, requests, candidate = (123, 1), (3, 1), False
            reported_tokens = (
                (123, 60) if case == "counts_disagree" and rank == 3 else tokens
            )
            if case == "bucket_disagree" and rank == 3:
                runner.capture_num_tokens = (256,)
            adapter.local_can_run = lambda batch, tcap, bcap: not (
                case == "rank_reject" and rank == 3
            )
            t, b = tokens[dp_rank], requests[dp_rank]
            lengths = ([1] * (b - 1) + [t - b + 1]) if b else []
            batch = NS(
                welm_dp_raw_rows_cpu=reported_tokens,
                global_num_requests_cpu=requests,
                welm_dp_prefill_graph_candidate=candidate,
                extend_seq_lens_cpu=lengths,
                batch_size=b,
                input_ids=torch.zeros(t, dtype=torch.long),
                input_embeds=None,
                replace_embeds=None,
                extend_prefix_lens_cpu=[0] * b,
                forward_mode=NS(is_target_verify=lambda: False),
                capture_hidden_mode=0,
                return_logprob=False,
                welm_prefill_graph_ticket=None,
            )
            with torch.inference_mode():
                RUNNER["prepare_welm_dp_graph_ticket"](runner, batch)
            ticket = batch.welm_prefill_graph_ticket
            expected = case in ("valid", "valid_again")
            assert ticket.allowed == expected, (rank, case, ticket)
            assert RUNNER["can_run_graph"](runner, batch) == expected
            if expected:
                assert (ticket.token_capacity, ticket.request_capacity) == (128, 4)
                assert ticket.token_rows == tokens and ticket.request_rows == requests
        # A manifest mismatch must fail on every rank, before graph capture.
        adapter.manifest = lambda: {"kind": "different" if rank == 3 else "test"}
        try:
            RUNNER["_validate_welm_dp_manifest"](runner)
        except ValueError:
            pass
        else:
            raise AssertionError("rank-local capture manifest was accepted")
    finally:
        dist.destroy_process_group()


class TestAdmission(unittest.TestCase):
    def test_dispatch_votes_before_graph_or_eager_and_skips_pure_decode(self):
        # Execute the actual shared dispatch method: a local decode/idle rank
        # must not bypass the vote when another DP group is prefilling.
        forward = load_nodes(
            SRT / "model_executor/model_runner.py",
            ["_forward_raw"],
            {
                "contextlib": contextlib,
                "has_forward_context": lambda: True,
                "get_global_dwdp_manager": lambda: None,
                "EagerRunner": type("EagerRunner", (), {}),
                "ModelRunnerOutput": NS,
                "device_timer_ctx": lambda *args: contextlib.nullcontext(),
                "_prefill_cuda_graph_allows_context_parallel": lambda *args: True,
            },
            class_name="ModelRunner",
        )["_forward_raw"]
        cases = (
            ("decode", False, True, ["decode"]),
            ("decode", True, True, ["prepare", "vote", "eager"]),
            ("idle", True, True, ["prepare", "vote", "eager"]),
            ("extend", True, True, ["prepare", "vote", "prefill"]),
            ("decode", True, False, ["decode"]),
        )
        for local_mode, has_extend, is_dp, expected in cases:
            with self.subTest(mode=local_mode, has_extend=has_extend, is_dp=is_dp):
                events = []
                batch = NS(
                    forward_mode=NS(
                        is_cuda_graph=lambda: local_mode in ("decode", "idle"),
                        is_decode=lambda: local_mode == "decode",
                        is_split_prefill=lambda: False,
                        is_extend=lambda **kw: local_mode == "extend",
                        is_target_verify=lambda: False,
                    ),
                    is_extend_in_batch=has_extend,
                    global_num_tokens_cpu=None,
                    welm_prefill_graph_ticket=None,
                )

                def vote(live):
                    events.append("vote")
                    live.welm_prefill_graph_ticket = NS(allowed=local_mode == "extend")

                prefill = NS(
                    welm_adapter=NS(is_dp=is_dp),
                    prepare_welm_dp_graph_ticket=vote,
                    can_run_graph=lambda b: b.welm_prefill_graph_ticket.allowed,
                    execute=lambda *a, **kw: events.append("prefill"),
                )
                runner = NS(
                    device="npu",
                    device_timer=None,
                    hisparse_coordinator=None,
                    prefill_cuda_graph_runner=prefill,
                    decode_cuda_graph_runner=NS(
                        can_run_graph=lambda b: True,
                        execute=lambda *a, **kw: events.append("decode"),
                    ),
                    eager_runner=NS(execute=lambda *a, **kw: events.append("eager")),
                    _prepare_eager_forward_batch=lambda b: events.append("prepare"),
                    _maybe_execute_deferred_mamba_cow_and_clear=lambda b: None,
                    _extend_forward_kwargs=lambda *a: {},
                )
                result = forward(runner, batch, None)
                self.assertEqual(events, expected)
                self.assertEqual(result.can_run_graph, expected[-1] != "eager")

    def test_common_capacities_and_unanimous_votes(self):
        ticket = contract.select_welm_dp_prefill_ticket(
            token_rows=[7000, 2100],
            request_rows=[3, 1],
            all_extend=True,
            token_buckets=[4096, 8192],
            request_buckets=[1, 2, 4],
        )
        self.assertEqual((ticket.token_capacity, ticket.request_capacity), (8192, 4))
        votes = [ticket.vote_record(True)] * 4
        self.assertEqual(contract.agree_welm_dp_prefill_ticket(ticket, votes), ticket)
        for rejecting_rank in range(4):
            altered = list(votes)
            altered[rejecting_rank] = ticket.vote_record(False)
            self.assertFalse(
                contract.agree_welm_dp_prefill_ticket(ticket, altered).allowed
            )

    def test_unsupported_and_capacity_cases(self):
        for rows, requests, active, allowed in [
            ([2, 2], [1, 1], True, True),
            ([2, 0], [1, 0], True, False),
            ([2, 1], [1, 1], False, False),
            ([2, 2], [3, 1], True, False),
            ([129, 2], [1, 1], True, False),
            (None, None, True, False),
        ]:
            ticket = contract.select_welm_dp_prefill_ticket(
                token_rows=rows,
                request_rows=requests,
                all_extend=active,
                token_buckets=[64, 128],
                request_buckets=[1, 2, 4],
            )
            self.assertEqual(ticket.allowed, allowed)
            self.assertEqual(len(ticket.vote_record(True)), 7)

    def test_request_storage_is_fixed_and_width_independent(self):
        max_mode = NS(is_max_len=lambda: True)
        namespace = load_nodes(
            SRT / "model_executor/forward_batch_info.py",
            [
                "WelmDpRowView",
                "WelmDpMoeRowLayout",
                "_welm_slot_rows",
                "_welm_build_row_view",
                "build_welm_dp_moe_row_layout",
            ],
            {
                "torch": torch,
                "dataclass": dataclass,
                "field": field,
                "ceil_align": lambda x, a: (x + a - 1) // a * a,
            },
        )
        token_vector, request_vector = torch.empty(2, dtype=torch.int32), torch.empty(
            2, dtype=torch.int32
        )
        builder = namespace["build_welm_dp_moe_row_layout"]
        layout = builder(
            raw_rows_cpu=[7, 3],
            request_rows_cpu=[3, 1],
            local_dp_rank=1,
            device="cpu",
            padding_mode=max_mode,
            current_real_rows_gpu=token_vector,
            current_slot_rows_cpu=[8, 8],
            request_real_rows_gpu=request_vector,
            request_slot_rows_cpu=[4, 4],
        )
        self.assertEqual(
            layout.current.real_rows_gpu.data_ptr(), token_vector.data_ptr()
        )
        self.assertEqual(
            layout.request.real_rows_gpu.data_ptr(), request_vector.data_ptr()
        )
        mask = (torch.arange(4)[None, :] < request_vector[:, None]).reshape(-1)
        self.assertEqual(
            mask.tolist(), [True, True, True, False, True, False, False, False]
        )
        request_vector.copy_(torch.tensor([1, 2], dtype=torch.int32))
        self.assertEqual(layout.current.real_rows_gpu.tolist(), [7, 3])
        self.assertEqual(layout.request.real_rows_gpu.tolist(), [1, 2])
        with self.assertRaises(ValueError):
            builder(
                raw_rows_cpu=[7, 3],
                request_rows_cpu=[3, 1],
                local_dp_rank=1,
                device="cpu",
                padding_mode=max_mode,
                request_slot_rows_cpu=[2, 2],
            )

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_four_rank_production_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_method = (Path(tmp) / "admission-store").as_uri()
            context = mp.spawn(
                _gloo_admission_worker, args=(init_method,), nprocs=4, join=False
            )
            deadline = time.monotonic() + 150
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("four-rank admission did not complete within 150s")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
