"""CPU control-flow checks for shared-expert overlap after the WeLM merge.

Execute the production methods with CPU tensors and explicit stream/kernel
stand-ins. These tests cover ordering and fallback selection, not NPU timing.
"""

import ast
import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/models/welmv4.py"
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _method(class_name, method_name, namespace):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], method],
        type_ignores=[],
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[method_name]


@pytest.mark.parametrize("clamp", [None, 0.5])
@pytest.mark.parametrize("reduce_scatter", [False, True])
def test_split_shared_gate_up_preserves_mlp_math_and_reduce_contract(
    clamp, reduce_scatter
):
    forward = _method("Qwen2MoeMLP", "forward", {"torch": torch, "F": F})
    finish = _method(
        "Qwen2MoeMLP", "forward_from_gate_up", {"torch": torch, "F": F}
    )
    values = torch.tensor([[1.0, -2.0], [-3.0, 4.0]])
    gate_up = torch.cat((values, values + 1), dim=-1)
    reductions = []

    def down_proj(x, *, skip_all_reduce):
        reductions.append(skip_all_reduce)
        return x * 2, None

    model = SimpleNamespace(
        gate_up_proj=lambda x: (gate_up.clone(), None),
        swiglu_clamp_limit=clamp,
        act_fn=lambda x: F.silu(x[..., :2]) * x[..., 2:],
        down_proj=down_proj,
    )
    model.forward_from_gate_up = MethodType(finish, model)
    ordinary = forward(model, values, use_reduce_scatter=reduce_scatter)
    split = finish(model, gate_up.clone(), use_reduce_scatter=reduce_scatter)
    gate = F.silu(values)
    up = values + 1
    if clamp is not None:
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    expected = gate * up * 2
    torch.testing.assert_close(ordinary, expected)
    torch.testing.assert_close(split, expected)
    assert reductions == [reduce_scatter, reduce_scatter]


def _run_sparse_moe(*, sidecar, multi_stream):
    events = []
    device_stream = object()

    class Backend:
        def is_none(self):
            return False

        def is_deepep(self):
            return True

        def is_megamoe(self):
            return False

    class Mode:
        def is_extend_or_draft_extend_or_mixed(self, **kwargs):
            return True

        def is_extend_without_speculative(self):
            return True

        def is_decode(self):
            return False

        def is_target_verify(self):
            return False

    class TopK:
        def __call__(self, hidden, logits, **kwargs):
            events.append("topk")
            return SimpleNamespace(
                topk_ids=torch.zeros((hidden.shape[0], 1), dtype=torch.int32),
                topk_weights=torch.ones((hidden.shape[0], 1)),
            )

    class Shared:
        def __call__(self, hidden):
            events.append("shared_whole")
            return hidden * 5

        def gate_up_proj(self, hidden):
            events.append("shared_gate_up")
            return hidden * 5, None

        def forward_from_gate_up(self, gate_up):
            events.append("shared_finish")
            return gate_up

    class MegaMoE:
        def local_valid_rows(self, num_tokens, *args):
            return num_tokens

        def forward_layer(self, experts, hidden, topk, rows, **kwargs):
            events.append("megamoe")
            return hidden * 2

    def experts(hidden, topk):
        events.append("fallback_experts")
        return hidden * 2

    def router_mm(hidden, weight, **kwargs):
        events.append("router")
        return torch.mm(hidden, weight)

    torch_proxy = SimpleNamespace(
        **{
            key: value for key, value in vars(torch).items()
            if key not in ("mm", "get_device_module")
        },
        mm=router_mm,
        get_device_module=lambda: SimpleNamespace(current_stream=lambda: device_stream),
    )
    envs = SimpleNamespace(
        SGLANG_NPU_WELMV4_MOE_STAGE_DUMP=SimpleNamespace(get=lambda: False),
        SGLANG_NPU_USE_MULTI_STREAM=SimpleNamespace(get=lambda: multi_stream),
        SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=SimpleNamespace(get=lambda: True),
    )
    namespace = {
        "torch": torch_proxy,
        "F": F,
        "_is_npu": True,
        "envs": envs,
        "get_moe_a2a_backend": Backend,
        "get_parallel": lambda: SimpleNamespace(moe_ep_size=4, attn_tp_rank=0),
        "process_shared_expert": lambda hidden, fn: fn(hidden),
        "wait_share_stream": lambda: events.append("wait"),
        "DeepEPMode": SimpleNamespace(NORMAL="normal", LOW_LATENCY="low_latency"),
    }
    forward = _method("Qwen2MoeSparseMoeBlock", "forward", namespace)
    shared = _method("Qwen2MoeSparseMoeBlock", "_forward_shared_expert", namespace)
    model = SimpleNamespace(
        shared_expert=Shared(),
        shared_expert_gate=None,
        is_kv_mirror_consumer=False,
        custom_routing_function=None,
        is_nextn=False,
        layer_id=0,
        num_hidden_layers=48,
        tp_size=1,
        get_npu_router_compute_weight_t=lambda: torch.ones((2, 1)),
        _resolve_deepep_mode_for_topk=lambda _: "normal",
        welm_prefill_megamoe=MegaMoE(),
        topk=TopK(),
        experts=experts,
    )
    model._forward_shared_expert = MethodType(shared, model)
    batch = SimpleNamespace(
        forward_mode=Mode(),
        enable_kv_mirror=False,
        welm_prefill_graph=None,
        welmv4_npu_deepep_scattered=True,
        welmv4_npu_deepep_full_mirror=False,
        num_token_non_padded=None,
        num_token_non_padded_cpu=2,
    )
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def record_stream(tensor, stream):
        assert stream is device_stream
        events.append("record_stream")

    with patch.object(torch.Tensor, "record_stream", record_stream):
        result = forward(
            model,
            hidden,
            None,
            batch,
            use_welm_prefill_megamoe=sidecar,
            force_serial_shared_expert=sidecar,
        )
    torch.testing.assert_close(result, hidden * 7)
    return events


def test_megamoe_overlaps_only_gate_up_and_waits_before_consumption():
    assert _run_sparse_moe(sidecar=True, multi_stream=True) == [
        "router", "shared_gate_up", "topk", "wait", "record_stream",
        "shared_finish", "megamoe",
    ]


def test_megamoe_serial_path_still_computes_shared_expert():
    assert _run_sparse_moe(sidecar=True, multi_stream=False) == [
        "shared_whole", "router", "topk", "megamoe",
    ]


def test_small_m_or_capacity_fallback_keeps_deepep_shared_overlap():
    # The layer calls with sidecar=False after either threshold/capacity rejects
    # MegaMoE. The shared expert must retain the normal DeepEP stream policy.
    assert _run_sparse_moe(sidecar=False, multi_stream=True) == [
        "router", "topk", "shared_whole", "fallback_experts", "wait",
    ]
