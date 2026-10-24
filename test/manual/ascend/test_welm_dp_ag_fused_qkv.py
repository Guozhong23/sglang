"""CPU behavior tests for attention-TP AG / BF16 fused-QKV pipelining.

Run from the repository root with:
    python -m pytest -q test/manual/ascend/test_welm_dp_ag_fused_qkv.py

Production methods run with real CPU tensors and simulated HCCL/kernel calls.
These tests check scheduling, group ownership and row/slot plumbing, not device
kernel numerics or actual communication/computation overlap.
"""

import copy
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_welm_tp2dp2_admission import SRT, load_nodes


MODEL = SRT / "models/welmv4.py"
DP_MODEL = SRT / "models/welmv4_dp_attention.py"
MODES = NS(EXTEND=1, DECODE=2, MIXED=3, IDLE=4, TARGET_VERIFY=5)


def attention_method(name, namespace=None):
    return load_nodes(
        MODEL,
        [name],
        dict(torch=torch, ForwardMode=MODES, **(namespace or {})),
        class_name="Qwen2MoeAttention",
    )[name]


def make_attention(mirror=False):
    attention = NS(
        num_heads=12,
        num_kv_heads=1,
        head_dim=256,
        qk_rope_head_dim=64,
        q_norm=None,
        k_norm=NS(weight=torch.ones(256, dtype=torch.bfloat16), eps=1e-6),
        q_size=3072,
        kv_size=256,
        kv_mirror_layer_idx=2,
        kv_mirror_layers=[33],
        kv_mirror_imitated_layers=[2] if mirror else [],
        attn=NS(layer_id=2),
        rotary_emb=NS(cos_sin_cache=torch.zeros(8192, 64)),
        qkv_proj=NS(bias=None, weight=torch.zeros(3584, 2048, dtype=torch.bfloat16)),
        use_npu_fused_qkv=True,
        is_nextn=False,
        sliding_window_size=-1,
    )
    if mirror:
        attention.qkv_proj_weight = torch.zeros(4096, 2048, dtype=torch.bfloat16)
    return attention


def _gloo_pipeline_worker(rank, init_method):
    """Two real TP2 groups, then one active and one idle DP replica."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=40),
    )
    try:
        groups = [
            dist.new_group(ranks, timeout=timedelta(seconds=40))
            for ranks in ([0, 1], [2, 3])
        ]
        dp_rank = rank // 2
        local_rows = 7
        slices = [(0, 3), (3, 7)]
        subgroup = NS(world_size=2, device_group=groups[dp_rank])
        owner = make_attention()
        positions = torch.arange(2 * local_rows)
        cache_slots = positions + 5
        caches = [torch.zeros(32, 256) for _ in range(2)]
        backend = NS(
            token_to_kv_pool=NS(get_kv_buffer=lambda layer: tuple(caches)),
            _is_swa_layer=lambda layer: False,
        )

        def compile_program(*args, **kwargs):
            assert kwargs["rank_chunk_layout"]

            def kernel(*runtime):
                received = runtime[0]
                outputs = runtime[6:9]
                full_rows, chunk_rows, start = runtime[-3:]
                for local_rank in range(2):
                    source = received[
                        local_rank * chunk_rows : (local_rank + 1) * chunk_rows, :1
                    ]
                    offset = local_rank * full_rows + start
                    for index, output in enumerate(outputs):
                        output[offset : offset + chunk_rows].copy_(source + index)

            return kernel

        def fail_world_lookup():
            raise AssertionError("DP pipeline used the world-4 group")

        pipeline = attention_method(
            "_npu_prefill_ag_fused_qkv",
            dict(
                is_dp_attention_enabled=lambda: True,
                get_tp_group=fail_world_lookup,
                get_attn_backend=lambda: backend,
                _get_welm_fused_qkv_program=compile_program,
                _WELMV4_FUSED_QKV_TILE_ROWS=256,
            ),
        )
        restore = load_nodes(
            MODEL, ["_restore_welm_prefill_ag_gate"], {"torch": torch}
        )["_restore_welm_prefill_ag_gate"]
        for round_index in range(2):
            dist.barrier()
            if round_index == 0 or dp_rank == 0:
                values = torch.arange(local_rows) + rank * 50 + round_index * 16
                hidden = values.to(torch.bfloat16).view(-1, 1).expand(-1, 2048)
                batch = NS(
                    welm_prefill_graph=None,
                    forward_mode=MODES.EXTEND,
                    batch_size=1,
                    extend_prefix_lens_cpu=[0],
                    out_cache_loc=cache_slots,
                )
                gate_hidden, outputs = pipeline(
                    owner,
                    positions,
                    hidden.contiguous(),
                    batch,
                    slices,
                    all_gather_group=subgroup,
                )
                # Expected data must include only this DP replica's two ranks.
                expected = torch.cat(
                    [
                        torch.arange(local_rows) + peer * 50 + round_index * 16
                        for peer in (dp_rank * 2, dp_rank * 2 + 1)
                    ]
                ).to(torch.bfloat16).view(-1, 1)
                for index, output in enumerate(outputs[:3]):
                    torch.testing.assert_close(
                        output, (expected + index).expand_as(output)
                    )
                gate = gate_hidden[:, :12].unsqueeze(-1)
                torch.testing.assert_close(
                    restore(gate, slices, 2),
                    expected.unsqueeze(-1).expand(-1, 12, 1),
                )
            # Idle DP ranks never issue an attention collective. A wrong world
            # group in the active replica would stall and hit the PG timeout.
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestAgKernelScheduling(TestCase):
    def test_real_tp2_groups_isolate_dp_replicas_and_allow_idle_peer_replica(self):
        with tempfile.TemporaryDirectory(prefix="welm-dp-qkv-ag-") as directory:
            init_method = Path(directory, "init").as_uri()
            mp.spawn(_gloo_pipeline_worker, args=(init_method,), nprocs=4, join=True)

    def run_pipeline(self, *, mirror=False, swa=False, mode=MODES.EXTEND):
        """Two rank-ordered chunks, including an unaligned last chunk."""
        local_rows = 513
        slices = [(0, 256), (256, local_rows)]
        total_rows = 2 * local_rows
        peers = [
            (torch.arange(local_rows) % 61 + rank * 70)
            .to(torch.bfloat16)
            .view(-1, 1)
            .expand(-1, 2048)
            .contiguous()
            for rank in range(2)
        ]
        positions = (torch.arange(total_rows) % 127).to(torch.int32)
        full_slots = torch.arange(total_rows).flip(0) + 17
        swa_slots = torch.arange(total_rows) * 2 + 3
        slots = swa_slots if swa else full_slots
        caches = [torch.full((total_rows * 2 + 17, 256), -1.0) for _ in range(2)]
        backend = NS(
            token_to_kv_pool=NS(get_kv_buffer=lambda layer: tuple(caches)),
            _is_swa_layer=lambda layer: swa,
            forward_metadata=NS(swa_out_cache_loc=swa_slots),
        )
        attention = make_attention(mirror)
        batch = NS(
            welm_prefill_graph=None,
            forward_mode=mode,
            batch_size=1 if mode == MODES.EXTEND else 2,
            extend_prefix_lens_cpu=[0],
            out_cache_loc=full_slots,
        )
        # A world-4 lookup is a failure even if a simulated gather could run.
        subgroup = NS(world_size=2, device_group=object(), ranks=[2, 3])
        world_lookup = Mock(side_effect=AssertionError("DP must use attention TP2"))
        events = []
        issued = []

        def compile_program(device, width, num_slots, max_pos, bulk, **kwargs):
            self.assertTrue(kwargs["rank_chunk_layout"])
            self.assertEqual(width, 4096 if mirror else 3584)
            events.append(("compile", bulk))

            def program(*args):
                (received, weight, gamma, pos, rope, cache_slots) = args[:6]
                outputs = args[6:11]
                k_cache, v_cache, epsilon, full_rows, chunk_rows, start = args[11:]
                events.append(("kernel", start))
                self.assertEqual(full_rows, local_rows)
                self.assertEqual(received.shape[0], 2 * chunk_rows)
                self.assertEqual(pos.dtype, torch.int64)
                torch.testing.assert_close(pos, positions.to(torch.int64))
                torch.testing.assert_close(cache_slots, slots)
                # Stand-in for rank-chunk kernel ABI: output rows and KV slots
                # are global/rank-major while input rows are chunk/rank-major.
                for rank in range(2):
                    input_rows = slice(rank * chunk_rows, (rank + 1) * chunk_rows)
                    output_rows = slice(
                        rank * full_rows + start,
                        rank * full_rows + start + chunk_rows,
                    )
                    values = received[input_rows, :1]
                    for index, output in enumerate(outputs[:5 if mirror else 3]):
                        output[output_rows].copy_(values + index)
                    k_cache[cache_slots[output_rows]] = (values + 1).float()
                    v_cache[cache_slots[output_rows]] = (values + 2).float()

            return program

        def all_gather(receive, send, *, group, async_op):
            chunk = len(issued)
            start, end = slices[chunk]
            self.assertIs(group, subgroup.device_group)
            self.assertTrue(async_op)
            torch.testing.assert_close(send, peers[0][start:end])
            events.append(("ag", start))
            issued.append((send, receive))

            def wait():
                events.append(("wait", start))
                receive.copy_(torch.cat([peer[start:end] for peer in peers]))

            return NS(wait=wait)

        fn = attention_method(
            "_npu_prefill_ag_fused_qkv",
            dict(
                get_tp_group=world_lookup,
                get_attn_backend=lambda: backend,
                _get_welm_fused_qkv_program=compile_program,
                _WELMV4_FUSED_QKV_TILE_ROWS=256,
            ),
        )
        with patch.object(torch.distributed, "all_gather_into_tensor", all_gather):
            gate_hidden, outputs = fn(
                attention,
                positions,
                peers[0],
                batch,
                slices,
                all_gather_group=subgroup,
            )
        world_lookup.assert_not_called()
        runtime_events = [event for event in events if event[0] != "compile"]
        self.assertEqual(
            runtime_events,
            [
                ("ag", 0), ("ag", 256),
                ("wait", 0), ("kernel", 0),
                ("wait", 256), ("kernel", 256),
            ],
        )
        self.assertTrue(
            all(event[0] == "compile" for event in events[:events.index(("ag", 0))])
        )
        expected = torch.cat(peers)[:, :1]
        for index, output in enumerate(outputs[:5 if mirror else 3]):
            torch.testing.assert_close(output, (expected + index).expand_as(output))
        if not mirror:
            self.assertEqual(outputs[3:], (None, None))
        torch.testing.assert_close(
            caches[0][slots], (expected + 1).float().expand(-1, 256)
        )
        torch.testing.assert_close(
            caches[1][slots], (expected + 2).float().expand(-1, 256)
        )
        torch.testing.assert_close(
            gate_hidden,
            torch.cat([peer[start:end] for start, end in slices for peer in peers]),
        )
        return gate_hidden, expected, slices, subgroup

    def test_all_chunks_issued_before_consumption_and_correct_group(self):
        self.run_pipeline()

    def test_mirror_outputs_and_swa_physical_slots(self):
        self.run_pipeline(mirror=True, swa=True, mode=MODES.MIXED)

    def test_dp_missing_attention_group_fails_before_collective(self):
        world_lookup = Mock(side_effect=AssertionError("Unexpected world group"))
        fn = attention_method(
            "_npu_prefill_ag_fused_qkv",
            dict(is_dp_attention_enabled=lambda: True, get_tp_group=world_lookup),
        )
        with patch.object(torch.distributed, "all_gather_into_tensor") as gather:
            with self.assertRaisesRegex(RuntimeError, "attention-TP group"):
                fn(NS(), None, None, None, [(0, 256), (256, 512)])
        world_lookup.assert_not_called()
        gather.assert_not_called()

    def test_gate_restores_rank_order_before_attention_multiply(self):
        gate_hidden, expected, slices, subgroup = self.run_pipeline()
        restore = load_nodes(
            MODEL, ["_restore_welm_prefill_ag_gate"], {"torch": torch}
        )["_restore_welm_prefill_ag_gate"]
        # A deterministic stand-in Gate projection has 12 independent heads.
        head_offsets = torch.arange(12).view(1, -1)
        gate = (gate_hidden[:, :1].float() + head_offsets).unsqueeze(-1)
        result = restore(gate, slices, subgroup.world_size)
        expected_gate = (expected.float() + head_offsets).unsqueeze(-1)
        torch.testing.assert_close(result, expected_gate)
        self.assertEqual(result.shape, (1026, 12, 1))
        self.assertFalse(torch.equal(gate, result))


class TestPipelineEligibility(TestCase):
    def setUp(self):
        self.owner = make_attention()
        self.batch = NS(
            welm_prefill_graph=None,
            forward_mode=MODES.EXTEND,
            _original_forward_mode=None,
        )
        self.hidden = torch.zeros(1025, 2048, dtype=torch.bfloat16)
        self.thresholds = dict(
            min_tokens=512,
            max_chunks=4,
            swa_min_tokens=None,
            swa_max_chunks=None,
        )

        def setting(key):
            return NS(get=lambda: self.thresholds[key])

        envs = NS(
            SGLANG_NPU_PREFILL_AG_FUSED_QKV_MIN_CHUNK_TOKENS=setting("min_tokens"),
            SGLANG_NPU_PREFILL_AG_FUSED_QKV_MAX_CHUNKS=setting("max_chunks"),
            SGLANG_NPU_PREFILL_AG_FUSED_QKV_SWA_MIN_CHUNK_TOKENS=setting(
                "swa_min_tokens"
            ),
            SGLANG_NPU_PREFILL_AG_FUSED_QKV_SWA_MAX_CHUNKS=setting("swa_max_chunks"),
        )
        get_slices = load_nodes(
            MODEL, ["_get_welm_prefill_ag_qkv_slices"],
            {"_WELMV4_FUSED_QKV_TILE_ROWS": 256},
        )["_get_welm_prefill_ag_qkv_slices"]
        self.can_fuse = attention_method("_can_use_npu_fused_qkv")
        self.owner._can_use_npu_fused_qkv = MethodType(self.can_fuse, self.owner)
        self.select = attention_method(
            "get_prefill_ag_fused_qkv_slices",
            {"envs": envs, "_get_welm_prefill_ag_qkv_slices": get_slices},
        )

    def select_slices(self, *, owner=None, batch=None, hidden=None, tp_size=2):
        return self.select(
            self.owner if owner is None else owner,
            self.hidden if hidden is None else hidden,
            self.batch if batch is None else batch,
            tp_size=tp_size,
        )

    def test_q12_producer_and_chunk_thresholds(self):
        expected = [(0, 256), (256, 512), (512, 768), (768, 1025)]
        self.assertEqual(self.select_slices(), expected)
        self.owner.kv_mirror_imitated_layers = [2]
        self.owner.qkv_proj_weight = torch.zeros(4096, 2048, dtype=torch.bfloat16)
        self.assertEqual(self.select_slices(), expected)
        self.thresholds["max_chunks"] = 2
        self.assertEqual(self.select_slices(), [(0, 512), (512, 1025)])
        for maximum in (0, 1):
            self.thresholds["max_chunks"] = maximum
            self.assertIsNone(self.select_slices())

    def test_swa_override_can_disable_pipeline_without_changing_full_layers(self):
        self.thresholds["swa_max_chunks"] = 0
        self.assertIsNotNone(self.select_slices())
        self.owner.sliding_window_size = 1024
        self.assertIsNone(self.select_slices())
        self.thresholds.update(swa_max_chunks=2, swa_min_tokens=4096)
        self.assertIsNone(self.select_slices())
        self.thresholds["swa_min_tokens"] = 512
        self.assertEqual(self.select_slices(), [(0, 512), (512, 1025)])

    def test_graph_decode_verify_idle_and_short_batches_do_not_pipeline(self):
        for mode in (MODES.DECODE, MODES.TARGET_VERIFY, MODES.IDLE):
            batch = copy.copy(self.batch)
            batch.forward_mode = mode
            self.assertIsNone(self.select_slices(batch=batch))
        for original_mode in (MODES.DECODE, MODES.IDLE, MODES.TARGET_VERIFY):
            batch = copy.copy(self.batch)
            batch._original_forward_mode = original_mode
            self.assertIsNone(self.select_slices(batch=batch))
        for graph in (NS(is_dp=True), NS(is_dp=False)):
            batch = copy.copy(self.batch)
            batch.welm_prefill_graph = graph
            self.assertIsNone(self.select_slices(batch=batch))
        for rows in (0, 1, 255, 256, 511):
            self.assertIsNone(self.select_slices(hidden=self.hidden[:rows]))
        self.assertIsNone(self.select_slices(tp_size=1))

    def test_disabled_nextn_and_mirror_consumer_filter_before_kernel(self):
        for name, value in (
            ("use_npu_fused_qkv", False),
            ("is_nextn", True),
            ("kv_mirror_layer_idx", 33),
        ):
            original = getattr(self.owner, name)
            setattr(self.owner, name, value)
            self.assertIsNone(self.select_slices())
            setattr(self.owner, name, original)

    def test_abi_failure_returns_none_before_collective_or_kv_access(self):
        for name, value in (
            ("q_norm", object()),
            ("k_norm", None),
            ("num_kv_heads", 2),
            ("num_heads", 8),
            ("head_dim", 128),
            ("qk_rope_head_dim", 32),
            ("qkv_proj_bias", torch.zeros(3584)),
        ):
            with self.subTest(name=name):
                absent = object()
                original = getattr(self.owner, name, absent)
                setattr(self.owner, name, value)
                self.assertIsNone(self.select_slices())
                if original is absent:
                    delattr(self.owner, name)
                else:
                    setattr(self.owner, name, original)
        for hidden in (self.hidden.float(), self.hidden[:, ::2], self.hidden[:, :1024]):
            self.assertIsNone(self.select_slices(hidden=hidden))
        original = self.owner.qkv_proj.weight
        self.owner.qkv_proj.weight = original.float()
        self.assertIsNone(self.select_slices())
        self.owner.qkv_proj.weight = original.t()
        self.assertIsNone(self.select_slices())

    def run_executor(
        self,
        *,
        output_dtype=torch.bfloat16,
        rows=1025,
        scattered=True,
        graph=False,
        idle=False,
        consumer=False,
        reuse_mxfp8=False,
    ):
        """Run the production DP forward and input-normalization boundary."""
        events = []
        received = []
        subgroup = NS(world_size=2, device_group=object())
        role = NS(TARGET_DP=object(), DRAFT=object())
        layout = NS(EP_SCATTERED=object(), DP_LOCAL_TP_ATTN_FULL=object())
        finish = NS(ATTN_TP_REDUCE_SCATTER=object())
        transport = NS(DEEPEP_NORMAL_AG_SCATTERED=object())

        class Mode(int):
            def is_target_verify(self):
                return self == MODES.TARGET_VERIFY

            def is_extend_without_speculative(self):
                return self in (MODES.EXTEND, MODES.MIXED)

        class Attention(NS):
            def __call__(owner, **kwargs):
                events.append("attention")
                received.append(kwargs)
                return kwargs["hidden_states"]

        attn = Attention(**vars(self.owner))
        attn._can_use_npu_fused_qkv = MethodType(self.can_fuse, attn)
        attn.can_reuse_prefill_mxfp8_input = lambda: reuse_mxfp8
        attn.o_proj = NS(weight=torch.zeros(1, dtype=torch.bfloat16))

        def select(hidden, batch, **kwargs):
            events.append("select")
            return self.select(attn, hidden, batch, **kwargs)

        attn.get_prefill_ag_fused_qkv_slices = select
        view = NS(local_real_rows=0 if idle else rows * 2)
        row_layout = NS(current=view, request=None)
        batch = NS(
            welm_dp_moe_row_layout=row_layout,
            is_extend_in_batch=True,
            enable_kv_mirror=False,
            welm_prefill_graph=(
                NS(is_dp=True, is_idle_role=lambda batch: idle) if graph else None
            ),
            welm_prefill_graph_phase="prompt",
            _original_forward_mode=None,
            forward_mode=Mode(MODES.EXTEND),
        )
        layer = NS(
            layer_id=33 if consumer else 2,
            kv_mirror_layers=[33],
            self_attn=attn,
            is_final_layer=False,
        )
        hidden = torch.zeros(rows, 2048, dtype=torch.float32)
        residual = torch.zeros_like(hidden)

        def normalize(**kwargs):
            events.append("norm")
            return kwargs["hidden_states"].to(output_dtype), kwargs["residual"]

        def all_gather(hidden, *, attn_tp_group):
            events.append("full-ag")
            self.assertIs(attn_tp_group, subgroup)
            return torch.cat((hidden, hidden))

        namespace = dict(
            torch=torch,
            nullcontext=nullcontext,
            WelmRunnerRole=role,
            WelmDpLayout=layout,
            WelmMoeTransport=transport,
            AttentionFinishKind=finish,
            WelmDpLayerState=NS,
            welm_attn_all_gather_rows=all_gather,
            welm_dp_idle_buffers=lambda view: (hidden.clone(), residual.clone()),
        )
        methods = load_nodes(
            DP_MODEL,
            ["forward", "_prepare_attention_input"],
            namespace,
            class_name="WelmDpAttentionExecutor",
        )
        policy_calls = []
        policy = NS(
            select=lambda **kwargs: (policy_calls.append(kwargs) or (0, False))
        )
        executor = NS(
            runner_plan=NS(
                finalized=True,
                role=role.TARGET_DP,
                first_mirror_consumer=None,
                attn_tp_size=2,
                attn_tp_group=subgroup,
                o_proj_returns_partial=True,
                oproj_policy=policy,
            ),
            planner=NS(
                resolve=lambda **kwargs: NS(
                    attention_finish=finish.ATTN_TP_REDUCE_SCATTER,
                    moe_transport=transport.DEEPEP_NORMAL_AG_SCATTERED,
                )
            ),
            _active_row_view=lambda **kwargs: view,
            _input_is_scattered=lambda **kwargs: scattered,
            _is_first_target_mirror_transition=lambda **kwargs: False,
            _normalize_input=normalize,
            _finish_attention_and_norm=lambda **kwargs: kwargs["state"],
            _run_moe_transport=lambda **kwargs: kwargs["state"],
            _assert_fp32_residual=lambda value: self.assertEqual(
                value.dtype, torch.float32
            ),
        )
        executor._prepare_attention_input = MethodType(
            methods["_prepare_attention_input"], executor
        )
        methods["forward"](
            executor,
            layer=layer,
            positions=None,
            hidden_states=hidden,
            forward_batch=batch,
            residual=residual,
        )
        return events, received, subgroup, policy_calls

    def test_executor_normalizes_fp32_before_selecting_bf16_and_passes_tp2_group(self):
        events, received, subgroup, policy_calls = self.run_executor()
        self.assertEqual(events, ["norm", "select", "attention"])
        self.assertEqual(received[0]["hidden_states"].shape[0], 1025)
        self.assertIs(received[0]["prefill_bf16_all_gather_group"], subgroup)
        self.assertIsNotNone(received[0]["prefill_ag_qkv_slices"])
        self.assertIsNone(received[0]["prefill_mxfp8_all_gather_group"])
        self.assertEqual(policy_calls[0]["rows"], 2050)

    def test_executor_rejects_post_norm_abi_or_short_input_before_attention(self):
        for kwargs in ({"output_dtype": torch.float32}, {"rows": 127}):
            with self.subTest(kwargs=kwargs):
                events, received, subgroup, policy_calls = self.run_executor(**kwargs)
                self.assertEqual(events, ["norm", "select", "full-ag", "attention"])
                self.assertIsNone(received[0]["prefill_ag_qkv_slices"])
                self.assertIsNone(received[0]["prefill_bf16_all_gather_group"])
                self.assertEqual(
                    received[0]["hidden_states"].shape[0], kwargs.get("rows", 1025) * 2
                )
                self.assertEqual(
                    policy_calls[0]["rows"], received[0]["hidden_states"].shape[0]
                )

    def test_executor_excludes_graph_consumer_idle_and_unscattered_input(self):
        for kwargs, expected in (
            ({"graph": True}, ["norm", "full-ag", "attention"]),
            ({"consumer": True}, ["norm", "full-ag", "attention"]),
            ({"scattered": False}, ["norm", "attention"]),
            ({"idle": True}, []),
            ({"graph": True, "idle": True}, []),
        ):
            with self.subTest(kwargs=kwargs):
                events, received, _, _ = self.run_executor(**kwargs)
                self.assertEqual(events, expected)
                if received:
                    self.assertIsNone(received[0]["prefill_ag_qkv_slices"])
                    self.assertIsNone(received[0]["prefill_bf16_all_gather_group"])

    def test_executor_keeps_mxfp8_transport_separate(self):
        events, received, subgroup, _ = self.run_executor(reuse_mxfp8=True)
        self.assertEqual(events, ["norm", "attention"])
        self.assertTrue(received[0]["reuse_prefill_mxfp8_input"])
        self.assertIs(received[0]["prefill_mxfp8_all_gather_group"], subgroup)
        self.assertIsNone(received[0]["prefill_ag_qkv_slices"])
        self.assertIsNone(received[0]["prefill_bf16_all_gather_group"])
