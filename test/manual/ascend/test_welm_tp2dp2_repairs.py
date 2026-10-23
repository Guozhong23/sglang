"""CPU behavior tests for the TP2DP2EP4 second-stage fixes.

Device kernels are validated separately by the single-NPU QKV/pack tests.
"""

import copy
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from test_welm_dp_prefill_graph_torch import capture_metadata, make_adapter, make_batch
from test_welm_tp2dp2_admission import SRT, contract, load_nodes


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


Policy = module("welm_oproj_test", SRT / "models/welm_oproj_policy.py").WelmOProjPolicy
Geometry = module(
    "welm_geometry_test", SRT / "layers/welm_fused_qkv_geometry.py"
).FusedQkvGeometry
ScratchPool = module(
    "welm_scratch_test", SRT / "models/welm_dp_scratch.py"
).WelmDpScratchPool
pack = module(
    "sglang.srt.layers.welm_dp_pack", SRT / "layers/welm_dp_pack.py"
).pack_dp_slot


class TestOProj(unittest.TestCase):
    def test_hccl_lookup_uses_global_rank_and_keeps_group_cache(self):
        calls = []

        class Group:
            def rank(self):
                return 1

            def _get_backend(self, device):
                return NS(get_hccl_comm_name=lambda rank: calls.append(rank) or "hccl")

        fn = load_nodes(
            SRT / "models/welmv4.py",
            ["_get_welm_npu_o_proj_hcom_name"],
            dict(
                torch=NS(
                    device=lambda name: name,
                    distributed=NS(get_rank=lambda: 3),
                    npu=NS(current_device=lambda: 7),
                ),
                logger=logging.getLogger(__name__),
            ),
            class_name="Qwen2MoeAttention",
        )["_get_welm_npu_o_proj_hcom_name"]
        owner = NS(_welm_npu_o_proj_hcom_names={})
        groups = [NS(device_group=Group(), ranks=[2, 3]) for _ in range(2)]
        for group in (groups[0], groups[0], groups[1]):
            self.assertEqual(fn(owner, group), "hccl")
        self.assertEqual(calls, [3, 3])

    def test_dispatch_thresholds_and_graph_exclusion(self):
        for fusion in (False, True):
            for max_chunks in (0, 2):
                policy = Policy(fusion, max_chunks, 1024)
                for rows in (2, 2046, 2048, 8192):
                    for graph in (False, True):
                        for eligible in (False, True):
                            result = policy.select(
                                rows=rows,
                                tp_size=2,
                                eligible=eligible,
                                is_dp_graph=graph,
                            )
                            chunks = 2 if max_chunks == 2 and rows >= 2048 else 0
                            expected = (chunks, fusion and not chunks)
                            self.assertEqual(
                                result,
                                expected if eligible and not graph else (0, False),
                            )
        with self.assertRaises(ValueError):
            Policy(min_chunk_tokens=0)

    def test_launcher_honors_external_mc2_switch(self):
        script = SRT.parents[2] / "examples/runtime/welm_vl/runbf16-vlm-tp2dp2ep4.sh"
        with tempfile.TemporaryDirectory() as tmp:
            local_script = Path(tmp) / "launch.sh"
            # External CANN setup is unavailable on CPU; keep all launcher assignments.
            local_script.write_text(
                "\n".join(
                    line
                    for line in script.read_text().splitlines()
                    if not line.startswith("source ")
                )
            )
            stub = Path(tmp) / "python"
            stub.write_text(
                '#!/bin/bash\nprintf "%s %s\\n" "$SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER" "$SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS" > "$CHECK_OUTPUT"\n'
            )
            stub.chmod(0o755)
            for requested in (None, "0", "1"):
                env = dict(
                    os.environ,
                    PATH=f"{tmp}:{os.environ['PATH']}",
                    PROFILE_ROOT=tmp,
                    LOG_ROOT=tmp,
                    CHECK_OUTPUT=f"{tmp}/result",
                )
                env.pop("SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER", None)
                if requested is not None:
                    env["SGLANG_NPU_PREFILL_OPROJ_MATMUL_REDUCE_SCATTER"] = requested
                subprocess.run(
                    ["bash", str(local_script)],
                    env=env,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                self.assertEqual(
                    Path(env["CHECK_OUTPUT"]).read_text().split(),
                    [requested or "0", "2"],
                )


class TestGeometry(unittest.TestCase):
    def test_model_fused_abi_guards_and_q12_outputs(self):
        calls = []
        backend = NS(
            token_to_kv_pool=NS(
                get_kv_buffer=lambda layer: (torch.empty(16, 256), torch.empty(16, 256))
            ),
            _is_swa_layer=lambda layer: False,
        )
        fn = load_nodes(
            SRT / "models/welmv4.py",
            ["_try_npu_fused_qkv"],
            dict(
                torch=torch,
                ForwardMode=NS(EXTEND=1, MIXED=3, DECODE=2, TARGET_VERIFY=5),
                get_attn_backend=lambda: backend,
                _get_welm_fused_qkv_program=lambda *args: lambda *runtime: calls.append(
                    runtime
                ),
            ),
            class_name="Qwen2MoeAttention",
        )["_try_npu_fused_qkv"]
        owner = NS(
            num_heads=12,
            num_kv_heads=1,
            head_dim=256,
            qk_rope_head_dim=64,
            q_norm=None,
            k_norm=NS(weight=torch.ones(256, dtype=torch.bfloat16), eps=1e-6),
            q_size=3072,
            kv_size=256,
            kv_mirror_layer_idx=0,
            kv_mirror_imitated_layers=[],
            attn=NS(layer_id=0),
            rotary_emb=NS(cos_sin_cache=torch.zeros(32, 64)),
            qkv_proj=NS(
                bias=None, weight=torch.zeros(3584, 2048, dtype=torch.bfloat16)
            ),
        )
        hidden = torch.zeros(2, 2048, dtype=torch.bfloat16)
        positions = torch.arange(2)
        batch = NS(welm_prefill_graph=None, forward_mode=2, out_cache_loc=positions)
        result = fn(owner, positions, hidden, batch)
        self.assertEqual(
            [t.shape if t is not None else None for t in result],
            [(2, 3072), (2, 256), (2, 256), None, None],
        )
        for name, value in (
            ("q_norm", object()),
            ("num_kv_heads", 2),
            ("head_dim", 128),
            ("num_heads", 8),
            ("qkv_proj_bias", torch.zeros(3584)),
        ):
            candidate = copy.copy(owner)
            setattr(candidate, name, value)
            self.assertIsNone(fn(candidate, positions, hidden, batch))
        batch.welm_prefill_graph = NS(is_dp=True)
        self.assertIsNone(fn(owner, positions, hidden, batch))
        self.assertEqual(len(calls), 1)

    def test_both_aot_entries_share_geometry_and_runtime_abi(self):
        ns = load_nodes(
            SRT / "layers/fused_qkv_proj_norm_rope_cache.py",
            ["compile_aot", "compile_aot_rank_chunk"],
            dict(
                FusedQkvGeometry=Geometry,
                HIDDEN=2048,
                HEAD_DIM=256,
                ROPE_DIM=64,
                BF16="bf16",
                F32="fp32",
                dtypes=NS(int64="i64", float32="fp32"),
                Dim=lambda name, **kw: name,
                TensorSpec=lambda shape, dtype: (shape, dtype),
                FusedQkvProjNormRopeCache=lambda **kw: NS(run=kw, run_rank_chunk=kw),
                cannbotdsl=NS(compile=lambda op, *specs: (op, specs)),
            ),
        )
        for width, heads, mirror in (
            (2048, 6, False),
            (2560, 6, True),
            (3584, 12, False),
            (4096, 12, True),
        ):
            geom = Geometry.from_width(width)
            self.assertEqual(
                geom.units,
                dict(k=heads, v=heads + 1, mirror_k=heads + 2, mirror_v=heads + 3),
            )
            for name, argc in (("compile_aot", 14), ("compile_aot_rank_chunk", 17)):
                op, specs = ns[name](width, 4096, 8192)
                self.assertEqual(len(specs), argc)
                self.assertEqual(op["num_q_heads"], heads)
                self.assertEqual(op["has_mirror"], mirror)
                self.assertEqual(specs[6][0][1], heads * 256)
                self.assertEqual(specs[9][0][0], specs[6][0][0] if mirror else 1)
        for kwargs in (dict(num_q_heads=8), dict(num_kv_heads=2), dict(head_dim=128)):
            with self.assertRaises(ValueError):
                Geometry(**kwargs)
        with self.assertRaises(ValueError):
            ns["compile_aot"](3072, 4096, 8192)


class TestIdleGraph(unittest.TestCase):
    def test_active_visual_range_ends_before_body_and_restores_on_failure(self):
        fn = load_nodes(
            SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
            ["_execute_body_capture"],
            dict(
                copy=copy,
                torch=torch,
                _slice_output_rows=lambda output, rows: output[:rows],
            ),
            class_name="PrefillCudaGraphRunner",
        )["_execute_body_capture"]
        for fail in (False, True):
            events = []

            class Range:
                def __enter__(self):
                    events.append("visual-enter")

                def __exit__(self, *args):
                    events.append("visual-exit")

            body = NS(forward=object())
            original = body.forward

            def outer(*args, **kw):
                events.append("compose")
                if fail:
                    raise ValueError("composition failed")
                body.forward()
                events.append("tail")
                return "result"

            adapter = NS(
                is_dp=True,
                dp_rank=0,
                pad_mirror=True,
                profile_ranges=True,
                request_real_rows_gpu=torch.tensor([1, 0]),
                profile_range=lambda name: Range(),
                replay=lambda *a, **kw: events.append("body") or torch.zeros(1, 4),
            )
            runner = NS(
                _is_full_backend=False,
                _input_embeds_arg_idx=3,
                layer_model=body,
                buffer_registry=NS(has_slot=lambda name: False),
                model_runner=NS(model=NS(forward=outer)),
                welm_adapter=adapter,
                _prefill_forward_context=lambda *a, **kw: nullcontext(),
            )
            batch = NS(
                batch_size=1,
                input_ids=torch.zeros(1),
                positions=torch.zeros(1),
                welm_prefill_graph_ticket=NS(request_rows=(1, 0)),
            )
            if fail:
                with self.assertRaisesRegex(ValueError, "composition failed"):
                    fn(runner, batch, batch, 128, 1, object())
            else:
                self.assertEqual(fn(runner, batch, batch, 128, 1, object()), "result")
            self.assertEqual(
                events,
                ["visual-enter", "compose", "visual-exit"]
                + ([] if fail else ["body", "tail"]),
            )
            self.assertIs(body.forward, original)

    def test_capture_roles_have_rank_uniform_order(self):
        fn = load_nodes(
            SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
            ["_capture_one_stream"],
            {},
            class_name="PrefillCudaGraphRunner",
        )["_capture_one_stream"]
        adapter = NS(is_dp=True, active_masks=(1, 2, 3), active_mask=3)
        calls = []
        fn(
            NS(
                welm_adapter=adapter,
                _capture_one_role=lambda: calls.append(adapter.active_mask),
            )
        )
        self.assertEqual(calls, [1, 2, 3])
        self.assertEqual(adapter.active_mask, 3)

    def test_roles_are_in_ticket_and_unanimous_vote(self):
        for tokens, requests, modes, mask in (
            ((123, 0), (1, 0), (1, 4), 1),
            ((0, 123), (0, 1), (4, 1), 2),
            ((123, 50), (2, 1), (1, 1), 3),
        ):
            ticket = contract.select_welm_dp_prefill_ticket(
                token_rows=tokens,
                request_rows=requests,
                phases=modes,
                all_extend=True,
                token_buckets=(64, 128),
                request_buckets=(1, 2, 4),
            )
            self.assertTrue(ticket.allowed)
            self.assertEqual(ticket.active_mask, mask)
            votes = [ticket.vote_record(True)] * 4
            self.assertTrue(
                contract.agree_welm_dp_prefill_ticket(ticket, votes).allowed
            )
            changed = list(votes[3])
            changed[3] ^= 3
            votes[3] = changed
            self.assertFalse(
                contract.agree_welm_dp_prefill_ticket(ticket, votes).allowed
            )
        for tokens, requests, modes in (
            ((0, 0), (0, 0), (4, 4)),
            ((123, 1), (1, 1), (1, 2)),
            ((123, 0), (1, 0), (3, 4)),
            ((0, 123), (1, 1), (4, 1)),
        ):
            self.assertFalse(
                contract.select_welm_dp_prefill_ticket(
                    token_rows=tokens,
                    request_rows=requests,
                    phases=modes,
                    all_extend=True,
                    token_buckets=(128,),
                    request_buckets=(1, 2),
                ).allowed
            )

    def test_idle_staging_handoff_and_role_switch_clear_stale_data(self):
        for dp in (0, 1):
            adapter = make_adapter(dp)
            adapter.active_masks = (1, 2, 3)
            for mask in adapter.active_masks:
                adapter.active_mask = mask
                capture_metadata(adapter)
            idle_mask = 1 << (1 - dp)
            keys = {adapter.key(128, active_mask=m) for m in (1, 2, 3)}
            self.assertEqual(len(keys), 3)
            ptrs = (
                adapter.full_write_locs.data_ptr(),
                adapter.prompt_real_rows_gpu.data_ptr(),
            )
            for _ in range(3):
                tokens, requests = [71, 71], [1, 1]
                tokens[dp] = requests[dp] = 0
                live = make_batch([])
                live.forward_mode, live.return_logprob, live.spec_info = 4, False, None
                live.extend_seq_lens_cpu = None
                live.welm_prefill_graph_ticket = NS(
                    allowed=True,
                    active_mask=idle_mask,
                    token_capacity=128,
                    request_capacity=1,
                    token_rows=tokens,
                    request_rows=requests,
                )
                self.assertTrue(
                    adapter.local_can_run(live, 128, 1, active_mask=idle_mask)
                )
                adapter.full_write_locs.fill_(777)
                adapter.mirror_q_used.fill_(1)
                static = copy.copy(live)
                adapter.prepare_replay(live, static, 128)
                self.assertEqual(live.batch_size, 0)
                self.assertTrue((adapter.full_write_locs == -1).all())
                self.assertFalse(adapter.valid_rows.any())
                self.assertFalse(adapter.mirror_q_used.any())
                # Any source attention/KV handoff would fail with this fixture.
                adapter.finish_prompt(
                    torch.full((64, 4), float("nan"), dtype=torch.bfloat16),
                    torch.full((64, 4), float("nan")),
                    static.positions,
                    static,
                )
                self.assertEqual(adapter.mirror_hidden.shape, (8, 4))
                self.assertEqual(adapter.mirror_residual.dtype, torch.float32)
                self.assertTrue((adapter.mirror_hidden == 0).all())
                self.assertTrue((adapter.mirror_residual == 0).all())
                self.assertEqual(
                    ptrs,
                    (
                        adapter.full_write_locs.data_ptr(),
                        adapter.prompt_real_rows_gpu.data_ptr(),
                    ),
                )

    def test_idle_replay_participates_in_lm_head_without_outer_model(self):
        fn = load_nodes(
            SRT / "model_executor/runner/prefill_cuda_graph_runner.py",
            ["_execute_body_capture"],
            dict(
                copy=copy,
                torch=torch,
                _slice_output_rows=lambda output, rows: output[:rows],
            ),
            class_name="PrefillCudaGraphRunner",
        )["_execute_body_capture"]
        events = []
        body = NS(forward=object())
        original = body.forward

        def logits(ids, hidden, head, batch):
            events.append("world-lm-head")
            self.assertEqual(hidden.shape, (0, 4))
            self.assertEqual(batch.batch_size, 0)
            self.assertEqual(batch.global_num_tokens_for_logprob_gpu.tolist(), [1, 0])
            return NS(next_token_logits=torch.empty(0, 8))

        model = NS(
            forward=Mock(side_effect=AssertionError("IDLE ran ViT/OE")),
            lm_head=object(),
            logits_processor=logits,
        )
        adapter = NS(
            is_dp=True,
            dp_rank=1,
            request_real_rows_gpu=torch.tensor([1, 0]),
            replay=lambda *a, **kw: events.append("body-replay") or torch.zeros(1, 4),
        )
        runner = NS(
            _is_full_backend=False,
            _input_embeds_arg_idx=3,
            layer_model=body,
            model_runner=NS(model=model),
            welm_adapter=adapter,
            _prefill_forward_context=lambda *a, **kw: nullcontext(),
        )
        batch = NS(
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            welm_prefill_graph_ticket=NS(request_rows=(1, 0)),
        )
        result = fn(runner, batch, copy.copy(batch), 128, 0, object())
        self.assertEqual(events, ["body-replay", "world-lm-head"])
        self.assertEqual(result.next_token_logits.shape[0], 0)
        self.assertIs(body.forward, original)


class TestScratchAndPack(unittest.TestCase):
    def test_startup_scratch_uses_real_allocator_and_all_role_shapes(self):
        from test_welm_dp_prefill_graph_torch import rows
        from enum import Enum, auto

        class Transport(Enum):
            DEEPEP_NORMAL_AG_SCATTERED = auto()
            EXPLICIT_AG_LOCAL_SORT_EP_AR = auto()
            DP_GATHER_TP_MOE = auto()

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
        prepare = load_nodes(
            SRT / "models/welmv4_dp_attention.py",
            ["prepare_forward_scratch"],
            dict(helpers, WelmMoeTransport=Transport, WelmRunnerRole=NS(DRAFT="draft")),
            class_name="WelmDpAttentionExecutor",
        )["prepare_forward_scratch"]
        pool = ScratchPool(
            lambda: NS(
                record=lambda: None, query=lambda: True, synchronize=lambda: None
            )
        )
        plan = NS(
            finalized=True,
            role="target",
            has_moe_ep=True,
            outer_target_dp_size=2,
            outer_target_dp_rank=0,
            attn_tp_size=2,
            attn_tp_rank=0,
            first_mirror_consumer=3,
            replica_dp_group=object(),
        )
        executor = NS(runner_plan=plan)
        executor.prepare_forward_scratch = lambda **kw: prepare(executor, **kw)
        model = NS(
            config=NS(hidden_size=8),
            layers=[NS(_welm_dp_executor=executor)],
            welm_dp_scratch_pool=pool,
        )
        imports = {
            "sglang.srt.layers.dp_attention": NS(
                DpPaddingMode=NS(MAX_LEN=NS(is_max_len=lambda: True))
            ),
            "sglang.srt.model_executor.forward_batch_info": NS(
                ForwardMode=NS(EXTEND=NS(is_target_verify=lambda: False)),
                build_welm_dp_moe_row_layout=rows["build_welm_dp_moe_row_layout"],
            ),
        }
        warmup = sys.modules["welm_scratch_test"].warmup_eager_scratch
        with patch.dict(sys.modules, imports):
            warmup(
                model, plan, capacities=(128, 256), dtype=torch.bfloat16, device="cpu"
            )
            pointers = {
                v.data_ptr()
                for entry in pool.entries
                for v in entry["tensors"].values()
                if isinstance(v, torch.Tensor)
            }
            warmup(
                model, plan, capacities=(128, 256), dtype=torch.bfloat16, device="cpu"
            )
        after = {
            v.data_ptr()
            for entry in pool.entries
            for v in entry["tensors"].values()
            if isinstance(v, torch.Tensor)
        }
        self.assertEqual(pointers, after)
        idle_shapes = {
            tuple(entry["tensors"]["_idle_hidden_buffer"].shape)
            for entry in pool.entries
            if "_idle_hidden_buffer" in entry["tensors"]
        }
        self.assertEqual(idle_shapes, {(64, 8), (128, 8), (1, 8), (16, 8)})
        for entry in pool.entries:
            if "_idle_residual_buffer" in entry["tensors"]:
                self.assertEqual(
                    entry["tensors"]["_idle_residual_buffer"].dtype, torch.float32
                )

    def test_pack_drops_nan_padding_and_nonleader_replicas(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [float("nan"), 9.0], [8.0, 8.0]])
        output = torch.empty(8, 2)
        pointer = output.data_ptr()
        for dp in (0, 1):
            for live in (0, 1, 2, 4, 1):
                counts = torch.tensor([live, live])
                for contribute in (False, True):
                    pack(output, source, counts, dp, dp * 4, contribute=contribute)
                    expected = torch.zeros_like(output)
                    if contribute:
                        expected[dp * 4 : dp * 4 + live] = source[:live]
                    torch.testing.assert_close(output, expected, equal_nan=True)
                    self.assertEqual(output.data_ptr(), pointer)

    def test_scratch_reuses_only_completed_leases(self):
        events = []

        def make_event():
            event = NS(done=False, record=lambda: None)
            event.query = lambda: event.done
            events.append(event)
            return event

        pool = ScratchPool(make_event)

        def acquire(capacity):
            batch, view = NS(), NS(real_rows_gpu=torch.tensor([1, 0]), _buffer=None)
            with pool.scope(batch):
                lease = batch.welm_dp_scratch_lease
                lease.bind(view, (capacity, "bf16", "prompt", "idle"))
                if view._buffer is None:
                    view._buffer = torch.empty(capacity, 4)
                lease.capture(view)
            self.assertIsNone(batch.welm_dp_scratch_lease)
            return view._buffer

        large = acquire(128)
        pending = acquire(128)
        self.assertNotEqual(large.data_ptr(), pending.data_ptr())
        for event in events:
            event.done = True
        acquire(32)
        self.assertEqual(large.data_ptr(), acquire(128).data_ptr())


class TestNpuPack(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch_npu"), "Ascend torch_npu required"
    )
    def test_fused_pack_matches_torch_with_nan_padding(self):
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            self.skipTest("No Ascend NPU available")
        device = os.environ.get("NPU_DEVICE", "npu:0")
        torch.npu.set_device(device)
        for dtype in (torch.bfloat16, torch.float32):
            source = torch.randn(257, 2048, dtype=dtype, device=device)
            output = torch.empty(514, 2048, dtype=dtype, device=device)
            reference = torch.empty_like(output)
            pointer = output.data_ptr()
            for live in (257, 1, 0, 256, 257):
                for dp in (0, 1):
                    counts = torch.tensor(
                        [live, live], dtype=torch.int32, device=device
                    )
                    payload = source.clone()
                    payload[live:] = float("nan")
                    for leader in (True, False):
                        pack(
                            reference, payload, counts, dp, dp * 257, contribute=leader
                        )
                        pack(
                            output,
                            payload,
                            counts,
                            dp,
                            dp * 257,
                            contribute=leader,
                            fused=True,
                        )
                        torch.npu.synchronize()
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)
                        self.assertEqual(output.data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
