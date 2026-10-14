"""CPU behavior checks for WeLM MegaMoE model integration.

Execute the complete production SparseMoE forward with real torch tensors,
FFN math and TopK. The external MegaMoE backend and stream submissions are
stand-ins; these tests do not validate Ascend numerics, HCCL or graph capture.
Run: python test/manual/ascend/test_welmv4_megamoe_model.py
"""

import ast
import sys
import types
import unittest
from enum import IntEnum, auto
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_welmv4_prefill_graph_contracts import SRT, load_method, load_nodes


NS = types.SimpleNamespace
MODEL = SRT / "models/welmv4.py"
RUNTIME = SRT / "hardware_backend/npu/moe/welmv4_megamoe.py"
ForwardMode = load_nodes(
    SRT / "model_executor/forward_batch_info.py",
    ["ForwardMode"],
    {"IntEnum": IntEnum, "auto": auto},
)["ForwardMode"]


def method_node(path, class_name, method_name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def assignment_expression(path, class_name, method_name, target):
    """Read a selector from the production method, without reproducing it."""
    method = method_node(path, class_name, method_name)
    assign = next(
        n for n in ast.walk(method)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)
    )
    return compile(ast.Expression(assign.value), str(path), "eval")


def call_keyword_expression(path, class_name, method_name, keyword):
    method = method_node(path, class_name, method_name)
    argument = next(n for n in ast.walk(method) if isinstance(n, ast.keyword) and n.arg == keyword)
    return compile(ast.Expression(argument.value), str(path), "eval")


NON_DP_SELECTOR = assignment_expression(
    MODEL, "Qwen2MoeDecoderLayer", "_forward_non_dp", "use_megamoe_prefill"
)
NON_DP_CALL = call_keyword_expression(
    MODEL, "Qwen2MoeDecoderLayer", "_forward_non_dp", "use_welm_prefill_megamoe"
)
THRESHOLD = load_method(RUNTIME, "WelmPrefillMegaMoE", "meets_prefill_threshold", {})
CAPACITY = load_method(RUNTIME, "WelmPrefillMegaMoE", "can_run", {})


def select_non_dp(runtime, hidden, batch, *, scattered=True, tp_size=4):
    scope = dict(
        megamoe=runtime, forward_batch=batch, output_hidden_is_scattered=scattered,
        hidden_states=hidden, tp_size=tp_size,
    )
    scope["use_megamoe_prefill"] = eval(NON_DP_SELECTOR, {"ForwardMode": ForwardMode}, scope)
    return eval(NON_DP_CALL, {}, scope)


class TorchCPU:
    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def mm(left, right, *, out_dtype):
        # CPU torch.mm has no NPU out_dtype argument. Keep actual matmul math.
        return torch.mm(left.to(out_dtype), right.to(out_dtype))

class TopKResult:
    def __init__(self, ids, weights):
        self.topk_ids, self.topk_weights = ids, weights


class SharedFFN:
    def __init__(self, events):
        self.events = events
        self.w1 = torch.tensor(
            [[0.5, -0.25], [0.25, 0.75], [1.0, 0.5], [-0.5, 0.25]],
            dtype=torch.bfloat16,
        )
        self.w2 = torch.tensor([[0.5, 0.25], [-0.25, 1.0]], dtype=torch.bfloat16)

    def reference(self, x):
        gate, up = (x @ self.w1.t()).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.w2.t()

    def gate_up_proj(self, x):
        self.events.append("shared_gate_up")
        return x @ self.w1.t(), None

    def forward_from_gate_up(self, gate_up):
        self.events.append("shared_ffn")
        gate, up = gate_up.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.w2.t()

    def __call__(self, x):
        gate_up, _ = self.gate_up_proj(x)
        return self.forward_from_gate_up(gate_up)


class RoutedFFN:
    def __init__(self, events):
        self.events = events

    @staticmethod
    def reference(x, topk):
        scale = ((topk.topk_ids + 1) * topk.topk_weights).sum(-1, keepdim=True)
        return (x.float() * scale).to(x.dtype)

    def __call__(self, x, topk):
        self.events.append("routed_backend")
        return self.reference(x, topk)


class MegaMoE:
    def __init__(self, events, shared):
        self.events, self.shared = events, shared
        self.calls = []
        self.threshold_checks = []
        self.prefill_token_threshold = 512
        self.max_local_rows = 16384

    def meets_prefill_threshold(self, rows):
        self.threshold_checks.append(rows)
        return THRESHOLD(self, rows)

    can_run = CAPACITY

    @staticmethod
    def local_valid_rows(shard_rows, real_rows, attn_tp_rank):
        return min(max(real_rows - attn_tp_rank * shard_rows, 0), shard_rows)

    def forward_layer(self, experts, x, topk, num_valid_rows, **kwargs):
        self.events.append("mega")
        mask = kwargs["valid_row_mask"]
        self.calls.append((x.clone(), num_valid_rows, dict(kwargs, valid_row_mask=(
            mask.clone() if mask is not None else None
        ))))
        output = experts.reference(x, topk)
        if kwargs["shared_weights"] is not None:
            self.events.append("fused_shared")
            output = output + self.shared.reference(x)
        if kwargs["zero_output_padding"]:
            if mask is not None:
                output.masked_fill_(~mask[:, None], 0)
            else:
                output[num_valid_rows:].zero_()
        return output


class TestModelForward(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.flags = {"mega": True, "streams": True}
        self.parallel = NS(moe_ep_size=4, attn_tp_rank=0)
        self.envs = NS(
            WELM_NPU_USE_MEGAMOE=NS(get=lambda: self.flags["mega"]),
            SGLANG_NPU_USE_MULTI_STREAM=NS(get=lambda: self.flags["streams"]),
            SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=NS(get=lambda: True),
        )

        def on_shared_stream(x, fn):
            self.events.append("shared_stream")
            return fn(x)

        namespace = {
            "torch": TorchCPU(), "F": F, "_is_npu": True, "envs": self.envs,
            "get_moe_a2a_backend": lambda: NS(is_deepep=lambda: True, is_none=lambda: False),
            "get_parallel": lambda: self.parallel,
            "process_shared_expert": on_shared_stream,
            "wait_share_stream": lambda: self.events.append("wait_shared"),
            "DeepEPMode": NS(NORMAL="normal", LOW_LATENCY="low_latency"),
            "StandardTopKOutput": TopKResult,
        }
        methods = {
            name: load_method(MODEL, "Qwen2MoeSparseMoeBlock", name, namespace)
            for name in ("forward", "_forward_shared_expert", "_mask_npu_padded_topk")
        }
        methods["_mask_npu_padded_topk"] = staticmethod(methods["_mask_npu_padded_topk"])
        methods["__call__"] = methods["forward"]
        self.block = type("TestSparseMoE", (), methods)()
        self.block.shared_expert = SharedFFN(self.events)
        self.block.shared_expert_gate = None
        self.block.experts = RoutedFFN(self.events)
        self.block.welm_prefill_megamoe = MegaMoE(self.events, self.block.shared_expert)
        self.block.welm_megamoe_shared_weights = object()
        self.block.is_nextn = self.block.is_kv_mirror_consumer = False
        self.block.layer_id, self.block.num_hidden_layers = 0, 1
        self.block.tp_size = 1
        self.block.welm_runner_plan = None
        self.block.custom_routing_function = None
        self.block.last_final_experts_output = self.block.last_final_shared_output = object()
        self.router_weight = torch.tensor([[0.5, 1.0], [0.25, -0.5]], dtype=torch.bfloat16)
        self.block.get_npu_router_compute_weight_t = lambda: self.router_weight
        self.block._resolve_deepep_mode_for_topk = lambda _: "normal"

        def topk(hidden, logits, **kwargs):
            self.events.append("topk")
            weights, ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
            return TopKResult(ids, weights)

        self.block.topk = topk
        self.x = torch.tensor([[0.5, -1.0], [1.0, 0.25], [-0.5, 0.75], [1.5, -0.5]],
                              dtype=torch.bfloat16)

    @staticmethod
    def batch(mode=ForwardMode.EXTEND, **kwargs):
        values = dict(
            forward_mode=mode, enable_kv_mirror=False,
            num_token_non_padded=None, num_token_non_padded_cpu=4,
            welmv4_npu_deepep_scattered=True, welmv4_npu_deepep_full_mirror=False,
            welm_prefill_graph=None, welm_prefill_graph_phase="prompt",
        )
        values.update(kwargs)
        return NS(**values)

    def reference(self, x):
        logits = torch.mm(x.float(), self.router_weight.float())
        weights, ids = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
        return (self.block.experts.reference(x, TopKResult(ids, weights))
                + self.block.shared_expert.reference(x))

    def test_fusion_skips_external_shared_and_publishes_only_combined_output(self):
        for skip_components in (False, True):
            with self.subTest(skip_components=skip_components):
                self.events.clear()
                output, routed, shared = self.block(
                    self.x.clone(), None, self.batch(), use_welm_prefill_megamoe=True,
                    return_components=True, skip_component_output=skip_components,
                )
                torch.testing.assert_close(output, self.reference(self.x), rtol=0, atol=0)
                self.assertIsNone(routed)
                self.assertIsNone(shared)
                self.assertIsNone(self.block.last_final_experts_output)
                self.assertIsNone(self.block.last_final_shared_output)
                self.assertEqual(self.events, ["topk", "mega", "fused_shared"])
                self.assertIs(self.block.welm_prefill_megamoe.calls[-1][2]["shared_weights"],
                              self.block.welm_megamoe_shared_weights)

    def test_prefill_fallback_retains_shared_once_and_overlaps_it(self):
        for mode, mirror in ((ForwardMode.EXTEND, False), (ForwardMode.MIXED, False),
                             (ForwardMode.EXTEND, True)):
            with self.subTest(mode=mode, mirror=mirror):
                self.events.clear()
                self.block.is_kv_mirror_consumer = mirror
                output = self.block(self.x.clone(), None, self.batch(mode, enable_kv_mirror=mirror),
                                    use_welm_prefill_normal_stream_policy=True)
                torch.testing.assert_close(output, self.reference(self.x), rtol=0, atol=0)
                expected = ["shared_stream", "shared_gate_up", "shared_ffn"]
                if mirror:
                    expected += ["topk"]
                else:
                    expected.insert(0, "topk")
                expected += ["routed_backend", "wait_shared"]
                self.assertEqual(self.events, expected)

    def test_threshold_and_capacity_select_fusion_or_full_shared_overlap(self):
        runtime = self.block.welm_prefill_megamoe
        for enabled, padded_rows, capacity, fused in (
            (False, 516, 16384, False),
            (True, 508, 16384, False),
            (True, 512, 16384, False),
            (True, 516, 16384, True),
            (True, 4 * 16385, 16384, False),
        ):
            with self.subTest(enabled=enabled, rows=padded_rows, capacity=capacity):
                self.flags["mega"] = enabled
                self.block.welm_prefill_megamoe = runtime if enabled else None
                runtime.max_local_rows = capacity
                runtime.threshold_checks.clear()
                runtime.calls.clear()
                self.events.clear()
                local_rows = padded_rows // 4
                hidden = self.x.repeat((local_rows + 3) // 4, 1)[:local_rows]
                batch = self.batch(num_token_non_padded_cpu=padded_rows)
                selected = select_non_dp(self.block.welm_prefill_megamoe, hidden, batch)
                self.assertEqual(selected, fused)
                output = self.block(hidden.clone(), None, batch,
                                    use_welm_prefill_megamoe=selected)
                torch.testing.assert_close(output, self.reference(hidden), rtol=0, atol=0)
                self.assertEqual(runtime.threshold_checks, [padded_rows] if enabled else [])
                self.assertEqual(self.events, ["topk", "mega", "fused_shared"] if fused else [
                    "topk", "shared_stream", "shared_gate_up", "shared_ffn",
                    "routed_backend", "wait_shared",
                ])

    def test_decode_verify_and_nextn_keep_existing_shared_stream_policy(self):
        for mode, nextn in ((ForwardMode.DECODE, False), (ForwardMode.TARGET_VERIFY, False),
                            (ForwardMode.DRAFT_EXTEND_V2, True)):
            with self.subTest(mode=mode, nextn=nextn):
                traces = []
                runtime = self.block.welm_prefill_megamoe
                for enabled in (False, True):
                    self.events.clear()
                    self.flags["mega"] = enabled
                    self.block.welm_prefill_megamoe = runtime if enabled else None
                    self.block.is_nextn = nextn
                    output = self.block(
                        self.x.clone(), None, self.batch(mode, welmv4_npu_deepep_scattered=False),
                        use_welm_decode_like_stream_policy=nextn,
                    )
                    torch.testing.assert_close(output, self.reference(self.x), rtol=0, atol=0)
                    traces.append(list(self.events))
                self.assertEqual(traces[0], traces[1])
                self.assertEqual(traces[1].count("shared_stream"), 1)
                self.assertEqual(traces[1].count("wait_shared"), 1)
                self.assertNotIn("mega", traces[1])

    def test_graph_uses_live_device_valid_rows_instead_of_capture_cpu_count(self):
        count = torch.tensor([4], dtype=torch.int32)
        batch = self.batch(num_token_non_padded=count, num_token_non_padded_cpu=4,
                           welm_prefill_graph=NS(pad_mirror=False))
        for real_rows in (4, 1, 0, 3, 4):
            with self.subTest(real_rows=real_rows):
                count.fill_(real_rows)
                hidden = self.x.clone()
                hidden[real_rows:] = float("nan")
                output = self.block(hidden, None, batch, use_welm_prefill_megamoe=True)
                captured_input, physical_rows, kwargs = self.block.welm_prefill_megamoe.calls[-1]
                mask = torch.arange(4) < real_rows
                torch.testing.assert_close(kwargs["valid_row_mask"], mask)
                self.assertEqual(physical_rows, 4)
                self.assertTrue(torch.isfinite(captured_input).all())
                torch.testing.assert_close(captured_input[real_rows:], torch.zeros_like(captured_input[real_rows:]))
                expected = self.reference(self.x)
                expected[real_rows:] = 0
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_dp_idle_keeps_the_same_padded_collective_call(self):
        path = SRT / "models/welmv4_dp_attention.py"
        selector = assignment_expression(path, "WelmDpAttentionExecutor", "_run_moe_transport", "use_megamoe")
        runtime = self.block.welm_prefill_megamoe
        runtime.can_run = lambda rows: 0 < rows <= 16384
        for mode, real_rows in ((ForwardMode.EXTEND, 3), (ForwardMode.IDLE, 0)):
            batch = self.batch(mode, welm_dp_all_active_ordinary_prefill=True)
            selected = eval(selector, {}, dict(
                megamoe_length_matched=True, forward_batch=batch, megamoe=runtime,
                active_view=NS(local_slot_rows=4), plan=NS(attn_tp_size=1),
            ))
            self.assertTrue(selected)
            hidden = self.x.clone()
            hidden[real_rows:] = 0  # DP transport sanitizes before entering the MLP.
            self.events.clear()
            self.block(hidden, None, batch, use_welm_prefill_megamoe=selected,
                       megamoe_num_valid_rows=real_rows, valid_row_mask=torch.arange(4) < real_rows)
            captured_input, count, kwargs = runtime.calls[-1]
            self.assertEqual((captured_input.shape[0], count), (4, real_rows))
            self.assertFalse(kwargs["zero_output_padding"])  # DP clears at its transport boundary.
            self.assertEqual(self.events, ["topk", "mega", "fused_shared"])
        batch.welm_dp_all_active_ordinary_prefill = False
        self.assertFalse(eval(selector, {}, dict(
            megamoe_length_matched=True, forward_batch=batch, megamoe=runtime,
            active_view=NS(local_slot_rows=4), plan=NS(attn_tp_size=1),
        )))


class TestSelectorsAndStartup(unittest.TestCase):
    def test_non_dp_selector_keeps_fusion_on_pure_extend(self):
        for enabled in (False, True):
            for mode in ForwardMode:
                for scattered in (False, True):
                    with self.subTest(enabled=enabled, mode=mode, scattered=scattered):
                        runtime = MegaMoE([], None)
                        selected = select_non_dp(
                            runtime if enabled else None,
                            torch.empty(129, 2), NS(forward_mode=mode), scattered=scattered,
                        )
                        expected = enabled and scattered and mode == ForwardMode.EXTEND
                        self.assertEqual(selected, expected)
                        self.assertEqual(runtime.threshold_checks, [516] if expected else [])

    def test_startup_off_skips_threshold_and_backend_then_checks_enabled_target(self):
        flags = {"mega": True}
        parallel = NS(moe_tp_size=1)

        def unexpected_threshold_read():
            self.fail("Disabled MegaMoE must not read or validate its threshold")

        envs = NS(
            WELM_NPU_USE_MEGAMOE=NS(get=lambda: flags["mega"]),
            WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=NS(get=unexpected_threshold_read),
        )
        initialize = load_method(SRT / "model_executor/model_runner.py", "ModelRunner",
                                 "maybe_init_welm_prefill_megamoe",
                                 {"envs": envs, "torch": torch, "get_parallel": lambda: parallel})
        runner = NS(is_draft_worker=False, device="npu", dtype=torch.bfloat16,
                    ps=NS(moe_ep_size=4), model_config=NS(hf_config=NS(architectures=["WeLMV4MoeForCausalLM"])))
        module_name = "sglang.srt.hardware_backend.npu.moe.welmv4_megamoe"
        module = types.ModuleType(module_name)
        calls, runtime = [], object()
        module.init_welm_prefill_megamoe = lambda value: calls.append(value) or runtime
        with patch.dict(sys.modules, {module_name: module}):
            flags["mega"] = False
            runner.device = "cpu"
            initialize(runner)
            self.assertIsNone(runner.welm_prefill_megamoe)
            self.assertEqual(calls, [])
            flags["mega"] = True
            with self.assertRaisesRegex(ValueError, "requires an NPU"):
                initialize(runner)
            self.assertEqual(calls, [])
            runner.is_draft_worker = True
            initialize(runner)
            self.assertIsNone(runner.welm_prefill_megamoe)
            self.assertEqual(calls, [])
            runner.is_draft_worker = False
            runner.device = "npu"
            initialize(runner)
            self.assertIs(runner.welm_prefill_megamoe, runtime)
            self.assertEqual(calls, [runner])


if __name__ == "__main__":
    unittest.main(verbosity=2)
