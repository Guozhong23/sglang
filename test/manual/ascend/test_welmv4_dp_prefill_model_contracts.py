"""Execute DP graph model methods with host fakes; no torch/NPU required.

Covers graph handoff ownership, capacity/masking, and normalization/collective
ordering. These tests do not validate device kernels or HCCL capture.
"""

import ast
import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/models/welmv4_dp_attention.py"


def load_method(name, namespace, *, path=SOURCE, class_name="WelmDpAttentionExecutor"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if getattr(n, "name", "") == class_name)
    method = copy.deepcopy(next(n for n in cls.body if getattr(n, "name", "") == name))
    method.decorator_list = []
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Tensor:
    def __init__(self, data, dtype="float32"):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = "cpu"
        self.shape = (
            (len(data), len(data[0]))
            if data and isinstance(data[0], list)
            else (len(data),)
        )

    def numel(self):
        return math.prod(self.shape)

    def to(self, dtype):
        return Tensor(self.data, dtype)

    def contiguous(self):
        return self

    def index_select(self, dim, indices):
        assert dim == 0
        return Tensor([self.data[int(i)] for i in indices.data], self.dtype)

    def masked_fill_(self, mask, value):
        for i, invalid in enumerate(mask.data):
            if invalid:
                self.data[i] = (
                    [value] * self.shape[1] if len(self.shape) == 2 else value
                )
        return self

    def __getitem__(self, key):
        # The only indexing used by production handoff is mask[:, None].
        assert key == (slice(None), None)
        return self

    def __sub__(self, value):
        return Tensor([x - value for x in self.data], self.dtype)

    def __eq__(self, value):
        return Tensor([x == value for x in self.data], "bool")

    def __and__(self, other):
        return Tensor([a and b for a, b in zip(self.data, other.data)], "bool")

    def __invert__(self):
        return Tensor([not x for x in self.data], "bool")


def zeros_like(t):
    return Tensor(
        [[0] * t.shape[1] for _ in t.data] if len(t.shape) == 2 else [0] * len(t.data),
        t.dtype,
    )


def where(mask, a, b):
    return Tensor(
        [x if keep else y for keep, x, y in zip(mask.data, a.data, b.data)], a.dtype
    )


TORCH = NS(
    long="long",
    bool="bool",
    float32="float32",
    bfloat16="bf16",
    zeros_like=zeros_like,
    where=where,
    div=lambda t, n, rounding_mode: Tensor([v // n for v in t.data], "long"),
)
LAYOUT = NS(EP_SCATTERED="scattered", DP_LOCAL_TP_ATTN_FULL="full")
FINISH = NS(ATTN_TP_REDUCE_SCATTER="rs", ATTN_TP_ALL_REDUCE="ar")
TRANSPORT = NS(
    DEEPEP_NORMAL_AG_SCATTERED="normal",
    EXPLICIT_AG_LOCAL_SORT_EP_AR="full",
    DP_GATHER_TP_MOE="tp",
)
ROLE = NS(TARGET_DP="target", DRAFT="draft")


class ModelContracts(unittest.TestCase):
    def test_handoff_unique_owners_fixed_capacity_and_live_request_mask(self):
        fn = load_method("prepare_graph_mirror_handoff", {"torch": TORCH})
        hidden = Tensor([[float("nan")], [2], [3], [4], [5], [6], [7], [8]], "bf16")
        tail = Tensor([2, 6, 999, 999], "long")
        valid = Tensor([True, True, False, False], "bool")
        partials = []
        for rank in (0, 1):
            calls = []
            executor = NS(
                runner_plan=NS(
                    finalized=True,
                    first_mirror_consumer=33,
                    attn_tp_size=2,
                    attn_tp_rank=rank,
                ),
                _input_is_scattered=lambda **kw: True,
                _prepare_attention_input=lambda **kw: (
                    calls.append("norm+attn-group-AG") or hidden,
                    kw["residual"],
                ),
                _assert_fp32_residual=lambda t: self.assertEqual(t.dtype, "float32"),
            )
            batch = NS(
                welm_prefill_graph=NS(is_dp=True),
                welm_prefill_graph_phase="prompt",
                welm_dp_moe_row_layout=NS(
                    current=NS(local_slot_rows=8), request=NS(local_real_rows=99)
                ),
            )
            residual = Tensor(
                [[float("nan")], [2], [3], [4]] if rank == 0 else [[5], [6], [7], [8]]
            )
            out, partial, scattered = fn(
                executor,
                layer=NS(layer_id=33),
                hidden_states=hidden,
                residual=residual,
                forward_batch=batch,
                tail_indices=tail,
                request_valid_mask=valid,
            )
            self.assertEqual(calls, ["norm+attn-group-AG"])
            self.assertTrue(scattered)
            self.assertEqual(out.data, [[3], [7], [0], [0]])
            self.assertEqual(out.shape, (4, 1))
            partials.append(partial.data)
        self.assertEqual(partials, [[[3], [0], [0], [0]], [[0], [7], [0], [0]]])
        # Reuse the same capacity with a different live B. CPU real B stays 99.
        valid.data[:] = [True, False, False, False]
        tail.data[:] = [6, 999, 999, 999]
        out, partial, _ = fn(
            executor,
            layer=NS(layer_id=33),
            hidden_states=hidden,
            residual=residual,
            forward_batch=batch,
            tail_indices=tail,
            request_valid_mask=valid,
        )
        self.assertEqual(out.data, [[7], [0], [0], [0]])
        self.assertEqual(partial.data, [[7], [0], [0], [0]])

    def test_full_residual_handoff_has_no_shard_assumption(self):
        fn = load_method("prepare_graph_mirror_handoff", {"torch": TORCH})
        executor = NS(
            runner_plan=NS(finalized=True, first_mirror_consumer=33),
            _input_is_scattered=lambda **kw: False,
            _prepare_attention_input=lambda **kw: (kw["hidden_states"], kw["residual"]),
            _assert_fp32_residual=lambda t: self.assertEqual(t.dtype, "float32"),
        )
        batch = NS(
            welm_prefill_graph=NS(is_dp=True),
            welm_prefill_graph_phase="prompt",
            welm_dp_moe_row_layout=NS(current=NS(local_slot_rows=3), request=NS()),
        )
        out, residual, partial = fn(
            executor,
            layer=NS(layer_id=33),
            hidden_states=Tensor([[1], [2], [3]], "bf16"),
            residual=Tensor([[11], [12], [13]]),
            forward_batch=batch,
            tail_indices=Tensor([2, 999], "long"),
            request_valid_mask=Tensor([True, False], "bool"),
        )
        self.assertFalse(partial)
        self.assertEqual(out.data, [[3], [0]])
        self.assertEqual(residual.data, [[13], [0]])

    def test_mirror_scratch_never_prepares_t_shard_for_odd_b(self):
        calls = []
        fn = load_method(
            "prepare_forward_scratch",
            {
                "WelmRunnerRole": ROLE,
                "WelmMoeTransport": TRANSPORT,
                "welm_dp_prepare_full_transport_scratch": lambda view, hidden: calls.append(
                    ("full", view.local_slot_rows)
                ),
                "welm_dp_prepare_attn_scattered_transport_scratch": lambda *a, **kw: self.fail(
                    "Mirror must not prepare scattered T slots"
                ),
            },
        )
        request = NS(local_slot_rows=1)
        batch = NS(
            welm_prefill_graph=NS(is_dp=True),
            welm_prefill_graph_phase="mirror",
            welm_dp_moe_row_layout=NS(current=NS(local_slot_rows=1), request=request),
            is_extend_in_batch=True,
            enable_kv_mirror=True,
            forward_mode=NS(is_target_verify=lambda: False),
        )
        executor = NS(
            runner_plan=NS(
                finalized=True, role=ROLE.TARGET_DP, first_mirror_consumer=33
            )
        )
        fn(executor, forward_batch=batch, hidden_states=NS(dim=lambda: 2))
        self.assertEqual(calls, [("full", 1)])

    def test_mirror_first_and_later_layers_keep_bcap_and_kernel_plan(self):
        fn = load_method(
            "forward",
            {
                "torch": TORCH,
                "WelmRunnerRole": ROLE,
                "WelmDpLayout": LAYOUT,
                "WelmMoeTransport": TRANSPORT,
                "AttentionFinishKind": FINISH,
                "WelmDpLayerState": NS,
                "envs": NS(
                    SGLANG_NPU_PREFILL_OPROJ_RS_PIPELINE_MAX_CHUNKS=NS(get=lambda: 8)
                ),
            },
        )
        for layer_id, real_b in ((33, 1), (34, 0)):
            calls = []
            attn_kwargs = []
            view = NS(local_real_rows=real_b, local_slot_rows=4)
            rows = NS(current=NS(), request=view)
            hidden = Tensor([[1], [2], [0], [0]], "bf16")
            residual = Tensor([[1], [2], [0], [0]])

            class Attention:
                def __call__(self, **kwargs):
                    attn_kwargs.append(kwargs)
                    return kwargs["hidden_states"]

            layer = NS(
                layer_id=layer_id,
                kv_mirror_layers=[33, 34],
                self_attn=Attention(),
                is_final_layer=True,
            )

            def prepare(**kw):
                calls.append("norm")
                return kw["hidden_states"], kw["residual"]

            def finish(**kw):
                calls.append(("finish", kw["graph_mirror_residual_partial"]))
                return kw["state"]

            executor = NS(
                runner_plan=NS(
                    finalized=True,
                    first_mirror_consumer=33,
                    role=ROLE.TARGET_DP,
                    attn_tp_size=2,
                    o_proj_returns_partial=True,
                ),
                planner=NS(
                    resolve=lambda **kw: NS(attention_finish=FINISH.ATTN_TP_ALL_REDUCE)
                ),
                _active_row_view=lambda **kw: view,
                _input_is_scattered=lambda **kw: layer_id == 33,
                _is_first_target_mirror_transition=lambda **kw: layer_id == 33,
                _prepare_attention_input=prepare,
                _finish_attention_and_norm=finish,
                _run_moe_transport=lambda **kw: kw["state"],
                _assert_fp32_residual=lambda t: None,
                _prepare_first_mirror_query=lambda **kw: self.fail(
                    "Repeated T->B selection"
                ),
                _trim_mirror_prefill_rows=lambda *a: self.fail(
                    "Captured CPU real-B slice"
                ),
            )
            batch = NS(
                welm_dp_moe_row_layout=rows,
                is_extend_in_batch=True,
                enable_kv_mirror=True,
                welm_prefill_graph=NS(is_dp=True, mirror_residual_is_partial=True),
                welm_prefill_graph_phase="mirror",
                _original_forward_mode=None,
                forward_mode=NS(
                    is_target_verify=lambda: False,
                    is_extend_without_speculative=lambda: True,
                ),
            )
            out, res = fn(
                executor,
                layer=layer,
                positions=None,
                hidden_states=hidden,
                forward_batch=batch,
                residual=residual,
            )
            self.assertEqual(out.shape, (4, 1))
            self.assertEqual(res.shape, (4, 1))
            self.assertEqual(
                calls,
                [("finish", True)] if layer_id == 33 else ["norm", ("finish", False)],
            )
            self.assertFalse(attn_kwargs[0]["use_o_proj_matmul_reduce_scatter"])
            self.assertEqual(attn_kwargs[0]["o_proj_rs_pipeline_chunks"], 0)
            self.assertFalse(attn_kwargs[0]["reuse_prefill_mxfp8_input"])

    def test_graph_uses_normal_moe_when_megamoe_is_configured(self):
        valid = Tensor([True, True, False, False], "bool")
        fn = load_method(
            "_run_moe_transport",
            {
                "WelmMoeTransport": TRANSPORT,
                "WelmDpLayout": LAYOUT,
                "welm_dp_attn_scattered_valid_mask": lambda *a, **kw: valid,
                "welm_dp_attn_scattered_invalid_mask": lambda *a: ~valid,
            },
        )
        calls = []

        class MLP:
            welm_prefill_megamoe = NS(
                meets_prefill_threshold=lambda n: True,
                can_run=lambda n: self.fail("Graph must not select MegaMoE"),
            )

            def __call__(self, hidden, *args, **kwargs):
                calls.append(kwargs)
                return hidden

        layer = NS(mlp=MLP(), is_final_layer=False)
        executor = NS(
            runner_plan=NS(outer_target_dp_rank=0, attn_tp_rank=0, attn_tp_size=2),
            _store_final_components=lambda layer, x: x,
        )
        state = NS(
            hidden_layout=LAYOUT.EP_SCATTERED,
            hidden_states=Tensor([[1], [2], [float("nan")], [float("nan")]], "bf16"),
            residual=Tensor([[1], [2], [float("nan")], [float("nan")]]),
        )
        batch = NS(
            welm_prefill_graph=NS(is_dp=True),
            welm_dp_all_active_ordinary_prefill=True,
            _original_forward_mode="extend",
            forward_mode="extend",
        )
        module = NS(ForwardMode=NS(EXTEND="extend"))
        with patch.dict(
            sys.modules, {"sglang.srt.model_executor.forward_batch_info": module}
        ):
            result = fn(
                executor,
                layer=layer,
                state=state,
                forward_batch=batch,
                batch_plan=NS(moe_transport=TRANSPORT.DEEPEP_NORMAL_AG_SCATTERED),
                active_view=NS(local_slot_rows=8, local_real_rows=2),
            )
        self.assertFalse(calls[0]["use_welm_prefill_megamoe"])
        self.assertTrue(calls[0]["use_welm_prefill_normal_stream_policy"])
        self.assertEqual(result.hidden_states.data, [[1], [2], [0], [0]])
        self.assertEqual(result.residual.data, [[1], [2], [0], [0]])

    def test_target_plan_uses_text_geometry_for_vlm_and_pure_text(self):
        path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
        parallel = NS(moe_ep_group="EP4", moe_tp_group="MoE-TP1")
        fn = load_method(
            "_maybe_build_welm_target_runner_plan",
            {
                "get_parallel": lambda: parallel,
                "envs": NS(
                    SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=NS(get=lambda: True),
                    SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=NS(get=lambda: False),
                ),
            },
            path=path,
            class_name="ModelRunner",
        )
        modules = {
            "sglang.srt.models.welmv4_dp_attention": NS(
                build_welm_target_runner_plan=lambda **kw: NS(**kw)
            ),
            "sglang.srt.distributed": NS(
                get_moe_ep_normal_comm_group=lambda: "NORMAL-EP4"
            ),
        }
        for arch in (
            "WeLMV4VLMForConditionalGeneration",
            "WeLMMoonViTVLForConditionalGeneration",
            "WeLMV4MoeForCausalLM",
        ):
            with self.subTest(architecture=arch), patch.dict(sys.modules, modules):
                # In particular the outer VLM config has no layer geometry.
                outer = NS(architectures=[arch])
                text = NS(
                    num_hidden_layers=48,
                    num_target_hidden_layers=48,
                    kv_mirror_layers=["33", "34", "47", "48"],
                )
                runner = NS(
                    is_draft_worker=False,
                    device="npu",
                    server_args=NS(enable_dp_attention=True),
                    model_config=NS(hf_config=outer, hf_text_config=text),
                    ps=NS(moe_ep_size=4, attn_tp_size=2),
                    attention_tp_group="TP2",
                )
                plan = fn(runner)
                self.assertEqual(plan.first_mirror_consumer, 33)
                self.assertEqual(plan.attn_tp_group, "TP2")
                self.assertEqual(plan.moe_ep_group, "EP4")
                self.assertEqual(plan.moe_ep_normal_group, "NORMAL-EP4")
                self.assertTrue(plan.o_proj_returns_partial)
                self.assertTrue(plan.normal_allgather_enabled)
                # NextN/draft workers must not receive the target plan.
                runner.is_draft_worker = True
                self.assertIsNone(fn(runner))

    def test_partial_b_residual_reduction_follows_oproj_before_norm(self):
        calls = []
        fn = load_method(
            "_finish_attention_and_norm",
            {
                "WelmMoeTransport": TRANSPORT,
                "AttentionFinishKind": FINISH,
                "WelmDpLayout": LAYOUT,
                "finish_welm_attn_partial_full": lambda x, **kw: calls.append(
                    "oproj-AR"
                )
                or x,
            },
        )
        group = NS(all_reduce=lambda x: calls.append("residual-AR") or x)
        executor = NS(
            runner_plan=NS(attn_tp_group=group), _assert_fp32_residual=lambda x: None
        )
        hidden = Tensor([[1], [0]], "bf16")
        residual = Tensor([[2], [0]])
        state = NS(
            hidden_states=hidden,
            residual=residual,
            residual_layout=LAYOUT.DP_LOCAL_TP_ATTN_FULL,
        )
        layer = NS(
            self_attn=NS(use_o_norm=False),
            post_attention_layernorm=lambda a, b, **kw: (
                calls.append("norm") or a,
                b,
                None,
            ),
        )
        result = fn(
            executor,
            layer=layer,
            state=state,
            attn_output=hidden,
            forward_batch=NS(),
            batch_plan=NS(
                moe_transport=TRANSPORT.EXPLICIT_AG_LOCAL_SORT_EP_AR,
                attention_finish=FINISH.ATTN_TP_ALL_REDUCE,
            ),
            active_view=NS(local_slot_rows=2),
            mirror_transition=True,
            mirror_row_indices=None,
            graph_mirror_residual_partial=True,
        )
        self.assertEqual(calls, ["oproj-AR", "residual-AR", "norm"])
        self.assertEqual(result.residual.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
