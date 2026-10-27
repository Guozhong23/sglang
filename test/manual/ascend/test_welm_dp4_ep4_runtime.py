"""CPU/Gloo validation of WeLM attention-TP1 / DP4 / EP4 contracts.

Run from the repository root with:
    python -m pytest -q test/manual/ascend/test_welm_dp4_ep4_runtime.py

These tests execute production metadata, plans, packing and collectives. They
do not validate HCCL, NPU kernels or graph capture on accelerator hardware.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import Enum, IntEnum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
SRT = ROOT / "python/sglang/srt"


def load_nodes(path, names, namespace, *, class_name=None):
    """Load production functions without importing optional serving packages."""
    tree = ast.parse(path.read_text())
    body = (
        tree.body
        if class_name is None
        else next(n.body for n in tree.body if getattr(n, "name", "") == class_name)
    )
    selected = [copy.deepcopy(n) for n in body if getattr(n, "name", "") in names]
    if len(selected) != len(names):
        raise AssertionError(f"Missing production symbols in {path}: {names}")
    body = [ast.parse("from __future__ import annotations").body[0], *selected]
    namespace.setdefault("__name__", __name__)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return NS(**namespace)


def direct_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PACK = direct_module("sglang.srt.layers.welm_dp_pack", SRT / "layers/welm_dp_pack.py")
POLICY = direct_module(
    "welm_dp4_oproj_policy_test", SRT / "models/welm_oproj_policy.py"
)
VISION = direct_module("welm_dp4_vision_test", SRT / "layers/welmv4_vision_dp.py")
ULYSSES = direct_module(
    "welm_dp4_ulysses_test", SRT / "layers/welmv4_vision_ulysses.py"
)
PADDING = load_nodes(
    SRT / "layers/dp_attention.py",
    ["DpPaddingMode"],
    {"IntEnum": IntEnum, "auto": auto},
).DpPaddingMode
ROWS = load_nodes(
    SRT / "model_executor/forward_batch_info.py",
    [
        "WelmDpRowView",
        "WelmDpMoeRowLayout",
        "_welm_slot_rows",
        "_welm_build_row_view",
        "build_welm_dp_moe_row_layout",
        "ForwardMode",
    ],
    {
        "dataclass": dataclass,
        "field": field,
        "torch": torch,
        "IntEnum": IntEnum,
        "auto": auto,
        "DpPaddingMode": PADDING,
        "ceil_align": lambda n, d: (n + d - 1) // d * d,
    },
)


def make_plan(rank, *, replica=True, replica_ranks=None):
    enabled = NS(get=lambda: False)
    envs = NS(
        SGLANG_WELMV4_DP_FUSED_PACK=enabled,
        SGLANG_WELMV4_DP_PROFILE_RANGES=enabled,
        SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER=enabled,
        SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=NS(get=lambda: 0),
        SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MIN_CHUNK_TOKENS=NS(get=lambda: 1024),
    )
    api = load_nodes(
        SRT / "models/welmv4_dp_attention.py",
        [
            "WelmRunnerRole",
            "WelmMoeTransport",
            "AttentionFinishKind",
            "WelmRunnerParallelPlan",
            "_validate_group_rank_size",
            "build_welm_target_runner_plan",
            "validate_welm_runner_plan",
            "WelmBatchExecutionPlan",
            "WelmDpExecutionPlanner",
        ],
        {
            "dataclass": dataclass,
            "Enum": Enum,
            "auto": auto,
            "envs": envs,
            "WelmOProjPolicy": POLICY.WelmOProjPolicy,
        },
    )
    local = NS(world_size=1, rank_in_group=0, ranks=[rank])
    world = NS(world_size=4, rank_in_group=rank, ranks=list(range(4)))
    replica_group = NS(
        world_size=4,
        rank_in_group=rank,
        ranks=list(range(4)) if replica_ranks is None else replica_ranks,
    )
    plan = api.build_welm_target_runner_plan(
        parallel_state=NS(
            attn_tp_rank=0, attn_tp_size=1, attn_dp_rank=rank, attn_dp_size=4
        ),
        attn_tp_group=local,
        moe_tp_group=local,
        moe_ep_group=world,
        moe_ep_normal_group=world,
        replica_dp_group=replica_group if replica else None,
        normal_allgather_enabled=True,
        local_ep_kernel_available=True,
        shared_expert_ep_replicated=True,
        o_proj_returns_partial=False,
        first_mirror_consumer=40,
    )
    plan = replace(plan, finalized=True)
    api.validate_welm_runner_plan(plan, require_finalized=True)
    return api, plan


def make_layout(rank, rows, requests, *, slots=None):
    return ROWS.build_welm_dp_moe_row_layout(
        raw_rows_cpu=list(rows),
        request_rows_cpu=list(requests),
        local_dp_rank=rank,
        device="cpu",
        padding_mode=PADDING.SUM_LEN,
        current_slot_rows_cpu=None if slots is None else list(slots),
    )


def gloo_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=45),
    )
    try:
        active = torch.ones(4, dtype=torch.int32)
        sync = load_nodes(
            SRT / "managers/scheduler_components/dp_attn.py",
            ["MLPSyncBatchInfo"],
            {
                "torch": torch,
                "dataclass": dataclass,
                "ForwardMode": ROWS.ForwardMode,
                "get_tp_group": lambda: NS(active_ranks_cpu=active),
                "_ENABLE_WELM_MEGAMOE": True,
                "_ENABLE_METRICS_DP_ATTENTION": False,
            },
        )
        modes = ROWS.ForwardMode
        scenarios = [
            ([32768, 17, 1024, 3], [2, 1, 1, 1], [modes.EXTEND] * 4),
            (
                [12, 2, 0, 1],
                [1, 2, 0, 1],
                [modes.EXTEND, modes.DECODE, modes.IDLE, modes.DECODE],
            ),
            (
                [1, 2, 0, 3],
                [1, 2, 0, 3],
                [modes.DECODE, modes.DECODE, modes.IDLE, modes.DECODE],
            ),
            ([0] * 4, [0] * 4, [modes.IDLE] * 4),
        ]
        for owner in range(4):
            scenarios.append(
                (
                    [11 if d == owner else 0 for d in range(4)],
                    [1 if d == owner else 0 for d in range(4)],
                    [modes.EXTEND if d == owner else modes.IDLE for d in range(4)],
                )
            )
        for counts, requests, forward_modes in scenarios:
            for use_reduce in (False, True):
                local_mode = forward_modes[rank]
                info = sync.MLPSyncBatchInfo(
                    dp_size=4,
                    tp_size=1,
                    cp_size=1,
                    num_tokens=counts[rank],
                    num_tokens_for_logprob=requests[rank],
                    num_requests=requests[rank],
                    can_run_decode_cuda_graph=local_mode.is_decode_or_idle(),
                    can_run_prefill_cuda_graph=False,
                    is_extend_in_batch=local_mode.is_extend(),
                    local_can_run_tbo=False,
                    local_forward_mode=local_mode.value,
                )
                info.all_gather("cpu", dist.group.WORLD, use_all_reduce=use_reduce)
                assert info.global_num_tokens == counts
                assert info.global_num_requests == requests
                assert info.welm_dp_forward_modes_cpu == [
                    mode.value for mode in forward_modes
                ]
                assert info.can_run_decode_cuda_graph == all(
                    mode.is_decode_or_idle() for mode in forward_modes
                )
                assert info.is_extend_in_batch == any(
                    mode.is_extend() for mode in forward_modes
                )
                assert not info.welm_dp_prefill_graph_candidate

        gather = load_nodes(
            SRT / "layers/dp_attention.py",
            ["welm_dp_replicate_gather"],
            {"torch": torch},
        ).welm_dp_replicate_gather
        calls = {"sum": 0, "ag": 0}

        def reduce(value):
            calls["sum"] += 1
            dist.all_reduce(value)
            return value

        def all_gather(output, value):
            calls["ag"] += 1
            dist.all_gather_into_tensor(output, value)

        group = NS(
            world_size=4,
            rank_in_group=rank,
            all_reduce=reduce,
            all_gather_into_tensor=all_gather,
        )
        for slots, counts in [
            ((4, 4, 4, 4), (4, 2, 0, 1)),
            ((4, 4, 4, 4), (0, 0, 0, 3)),
            ((4, 0, 2, 1), (3, 0, 1, 1)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
        ]:
            view = make_layout(
                rank, counts, [min(c, 1) for c in counts], slots=slots
            ).current
            n, total = slots[rank], sum(slots)
            source = torch.full((n, 3), float("nan"))
            source[: counts[rank]] = (
                torch.arange(counts[rank] * 3).reshape(-1, 3) + rank * 100 + 1
            )
            expected = torch.zeros(total, 3)
            mask = torch.zeros(total, dtype=torch.bool)
            for d in range(4):
                off, count = view.slot_offsets_cpu[d], counts[d]
                expected[off : off + count] = (
                    torch.arange(count * 3).reshape(-1, 3) + d * 100 + 1
                )
                mask[off : off + count] = True
            view._replicate_gather_buffer = torch.empty(total, 3)
            view._replica_gather_input = torch.empty(n, 3)
            pointer = view._replicate_gather_buffer.data_ptr()
            for fused, replica in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                before = dict(calls)
                actual = gather(
                    source,
                    view,
                    group=group,
                    dp_rank=rank,
                    contribute=True,
                    fused_pack=fused,
                    replica_dp_group=group if replica else None,
                )
                if not fused and not replica:
                    actual.masked_fill_(~mask[:, None], 0)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert actual.data_ptr() == pointer
                equal_slots = len(set(slots)) == 1 and slots[0] > 0
                assert calls["ag"] - before["ag"] == int(replica and equal_slots)
                assert calls["sum"] - before["sum"] == int(
                    total > 0 and not (replica and equal_slots)
                )
    finally:
        dist.destroy_process_group()


class TestDp4Ep4Runtime(unittest.TestCase):
    def test_decode_replay_updates_four_counts_without_replacing_device_storage(self):
        pinned_sources = []

        def host_tensor(values, *, dtype, pin_memory):
            self.assertTrue(pin_memory)
            tensor = torch.tensor(values, dtype=dtype)
            pinned_sources.append(tensor)
            return tensor

        methods = load_nodes(
            SRT / "model_executor/runner/decode_cuda_graph_runner.py",
            ["_welm_target_request_bs", "_update_welm_target_real_rows"],
            {
                "_welm_target_plan": lambda runner: runner.plan,
                "torch": NS(tensor=host_tensor, int32=torch.int32),
            },
            class_name="DecodeCudaGraphRunner",
        )
        runner = NS(
            model_runner=NS(plan=NS(outer_target_dp_size=4)),
            captured_req_width=1,
            welm_dp_real_rows_gpu=torch.empty(4, dtype=torch.int32),
        )
        runner._welm_target_request_bs = lambda batch: methods._welm_target_request_bs(
            runner, batch
        )
        pointer = runner.welm_dp_real_rows_gpu.data_ptr()
        for counts in ([1, 0, 0, 0], [0, 3, 1, 0], [0, 0, 0, 5], [1, 1, 1, 1]):
            batch = NS(global_num_requests_cpu=counts)
            self.assertEqual(runner._welm_target_request_bs(batch), max(counts))
            methods._update_welm_target_real_rows(runner, batch)
            self.assertEqual(runner.welm_dp_real_rows_gpu.tolist(), counts)
            self.assertEqual(runner.welm_dp_real_rows_gpu.data_ptr(), pointer)
        self.assertEqual(len({t.data_ptr() for t in pinned_sources}), 4)
        with self.assertRaisesRegex(RuntimeError, "synchronized"):
            runner._welm_target_request_bs(NS(global_num_requests_cpu=[1, 0]))

    def test_scratch_warmup_covers_all_dp4_roles_with_production_allocators(self):
        scratch = direct_module(
            "welm_dp4_scratch_test", SRT / "models/welm_dp_scratch.py"
        )
        helpers = load_nodes(
            SRT / "layers/dp_attention.py",
            [
                "_welm_validate_scratch_tensor",
                "_welm_prepare_idle_buffers",
                "welm_dp_prepare_full_transport_scratch",
                "welm_dp_prepare_attn_scattered_transport_scratch",
            ],
            {"torch": torch},
        )
        for rank in range(4):
            api, plan = make_plan(rank)
            method = load_nodes(
                SRT / "models/welmv4_dp_attention.py",
                ["prepare_forward_scratch"],
                dict(
                    vars(helpers),
                    WelmMoeTransport=api.WelmMoeTransport,
                    WelmRunnerRole=api.WelmRunnerRole,
                ),
                class_name="WelmDpAttentionExecutor",
            ).prepare_forward_scratch
            pool = scratch.WelmDpScratchPool(
                lambda: NS(
                    record=lambda: None, query=lambda: True, synchronize=lambda: None
                ),
                max_entries=64,
            )
            executor = NS(runner_plan=plan)
            calls = []

            def prepare(**kwargs):
                batch = kwargs["forward_batch"]
                method(executor, **kwargs)
                layout = batch.welm_dp_moe_row_layout
                calls.append(
                    (
                        tuple(layout.current.real_rows_cpu),
                        tuple(layout.request.real_rows_cpu),
                    )
                )
                self.assertEqual(
                    layout.current._attn_scattered_valid_mask.sum().item(),
                    layout.current.local_real_rows,
                )
                self.assertEqual(
                    layout.request._segmented_valid_mask.sum().item(),
                    sum(layout.request.real_rows_cpu),
                )

            executor.prepare_forward_scratch = prepare
            model = NS(
                config=NS(hidden_size=8),
                layers=[NS(_welm_dp_executor=executor)],
                welm_dp_scratch_pool=pool,
            )
            modules = {
                "sglang.srt.layers.dp_attention": NS(DpPaddingMode=PADDING),
                "sglang.srt.model_executor.forward_batch_info": NS(
                    ForwardMode=ROWS.ForwardMode,
                    build_welm_dp_moe_row_layout=ROWS.build_welm_dp_moe_row_layout,
                ),
            }
            with patch.dict(sys.modules, modules):
                scratch.warmup_eager_scratch(
                    model, plan, capacities=(5,), dtype=torch.bfloat16, device="cpu"
                )
            masks = {
                sum((1 << d) for d, rows in enumerate(counts) if rows)
                for counts, _ in calls
            }
            self.assertEqual(masks, set(range(1, 16)))
            self.assertTrue(all(max(counts) == 5 for counts, _ in calls))
            self.assertTrue(
                all(
                    not entry["busy"] and entry["event"].query()
                    for entry in pool.entries
                )
            )

    def test_plans_keep_attention_local_and_switch_moe_transport(self):
        for rank in range(4):
            api, plan = make_plan(rank)
            self.assertEqual(
                (
                    plan.attn_tp_size,
                    plan.attn_tp_rank,
                    plan.outer_target_dp_rank,
                    plan.moe_ep_size,
                ),
                (1, 0, rank, 4),
            )
            planner = api.WelmDpExecutionPlanner(plan)
            for prefill, layer_id, mirror in (
                (True, 0, True),
                (True, 40, True),
                (False, 0, True),
                (True, 45, False),
            ):
                execution = planner.resolve(
                    has_ordinary_prefill=prefill,
                    layer_id=layer_id,
                    enable_kv_mirror=mirror,
                )
                self.assertEqual(
                    execution.attention_finish, api.AttentionFinishKind.IDENTITY
                )
                normal = prefill and (not mirror or layer_id < 40)
                self.assertEqual(
                    execution.moe_transport,
                    api.WelmMoeTransport.DEEPEP_NORMAL_AG_SCATTERED
                    if normal
                    else api.WelmMoeTransport.EXPLICIT_AG_LOCAL_SORT_EP_AR,
                )
            wrong = replace(
                plan,
                moe_ep_group=NS(
                    world_size=4, rank_in_group=(rank + 1) % 4, ranks=list(range(4))
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "rank order"):
                api.validate_welm_runner_plan(wrong, require_finalized=True)
            with self.assertRaisesRegex(ValueError, "replica-DP"):
                make_plan(rank, replica_ranks=[0, 2, 1, 3])
            _, without_replica = make_plan(rank, replica=False)
            self.assertIsNone(without_replica.replica_dp_group)

    def test_four_slot_mirror_and_decode_counts_use_fixed_storage(self):
        for rank in range(4):
            current = torch.empty(4, dtype=torch.int32)
            requests = torch.empty(4, dtype=torch.int32)
            pointers = current.data_ptr(), requests.data_ptr()
            for counts, reqs in (
                ([32768, 0, 17, 1], [3, 0, 2, 1]),
                ([0, 2, 0, 3], [0, 2, 0, 3]),
            ):
                layout = ROWS.build_welm_dp_moe_row_layout(
                    raw_rows_cpu=counts,
                    request_rows_cpu=reqs,
                    local_dp_rank=rank,
                    device="cpu",
                    padding_mode=PADDING.MAX_LEN,
                    current_real_rows_gpu=current,
                    request_real_rows_gpu=requests,
                )
                self.assertEqual((current.data_ptr(), requests.data_ptr()), pointers)
                self.assertEqual(current.tolist(), counts)
                self.assertEqual(requests.tolist(), reqs)
                for layer, extend, expected in (
                    (39, True, layout.current),
                    (40, True, layout.request),
                    (47, False, layout.current),
                ):
                    self.assertIs(
                        layout.for_layer(
                            layer, first_mirror_consumer=40, is_extend_in_batch=extend
                        ),
                        expected,
                    )
                self.assertEqual(layout.current.local_slot_rows, max(counts))
                self.assertEqual(layout.current.global_slot_rows, 4 * max(counts))

    def test_vision_requests_remain_in_singleton_attention_groups(self):
        config = NS(num_attention_heads=16, hidden_size=1152, num_hidden_layers=27)
        for rank in range(4):
            fail = Mock(
                side_effect=AssertionError("Independent DP images must not communicate")
            )
            local = NS(
                world_size=1,
                rank_in_group=0,
                ranks=[rank],
                all_gather_into_tensor=fail,
            )
            modules = {
                "sglang.srt.layers.welmv4_vision_parallel": NS(
                    vision_parallel_mode=lambda enabled: "dp" if enabled else "tp"
                ),
                "sglang.srt.runtime_context": NS(
                    get_mm=lambda: NS(mm_enable_dp_encoder=True)
                ),
                "sglang.srt.utils": NS(is_npu=lambda: True),
                "sglang.srt.distributed": NS(
                    get_attn_tp_group=lambda: local,
                    get_tp_group=lambda: NS(ranks=list(range(4))),
                ),
                "sglang.srt.environ": NS(
                    envs=NS(SGLANG_VIT_ENABLE_CUDA_GRAPH=NS(get=lambda: False))
                ),
            }
            with patch.dict(sys.modules, modules), patch.dict(
                os.environ,
                {
                    "SGLANG_VLM_VISION_USE_ULYSSES": "0",
                    "SGLANG_VLM_VISION_USE_DP": "0",
                },
            ):
                group, mode = ULYSSES.resolve_vision_parallel_group(config)
            self.assertIs(group, local)
            self.assertEqual(mode, "dp")
            images = [
                NS(
                    hash=rank * 10 + i,
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                    feature=torch.full((4, 2), float(rank * 10 + i)),
                )
                for i in range(rank + 1)
            ]
            encoder = Mock(side_effect=lambda pixels, grids: pixels)
            encoder.vision_dp_input_spec = VISION.VisionDPInputSpec((2, 2), (2,), 2)
            encoder.vision_dp_max_patches = 8
            encoder.dtype, encoder.device = torch.float32, "cpu"
            projector = NS(
                forward_images=lambda hidden, grids: hidden.reshape(-1, 4, 2).mean(
                    dim=1
                )
            )
            output = VISION.encode_vision_dp(images, encoder, projector, group)
            self.assertEqual(
                VISION.synchronize_vision_dp_cache(
                    images, [False] * len(images), group=group
                ),
                [True] * len(images),
            )
            for item, actual in zip(images, output):
                torch.testing.assert_close(actual, item.feature[:1], rtol=0, atol=0)
            fail.assert_not_called()

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_four_rank_scheduler_metadata_and_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = mp.spawn(
                gloo_worker,
                args=((Path(tmp) / "store").as_uri(),),
                nprocs=4,
                join=False,
            )
            deadline = time.monotonic() + 120
            try:
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail(
                            "DP4 metadata/transport did not finish in 120 seconds"
                        )
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
