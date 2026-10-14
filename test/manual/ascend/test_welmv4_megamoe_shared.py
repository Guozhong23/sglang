"""CPU contracts for WeLM shared MegaMoE; real tensors, no Ascend extension.

Run with ``python -m unittest discover -s test/manual/ascend
-p test_welmv4_megamoe_shared.py``. Device numerics and graph capture require
the separate NPU smoke test and the matching operator packages.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


NS = types.SimpleNamespace
SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def env(value):
    return NS(get=lambda: value)


ENV = NS(
    WELM_NPU_USE_MEGAMOE=env(True),
    WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD=env(0),
    SGLANG_DEEPEP_NORMAL_USE_ALLGATHER=env(False),
    SGLANG_DEEPEP_NORMAL_USE_ALLTOALL=env(False),
)


class SiluAndMul(torch.nn.Module):
    """Avoid importing device kernels just to identify the activation type."""

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up


def load_runtime():
    name = "_test_welmv4_megamoe_shared_runtime"
    spec = importlib.util.spec_from_file_location(
        name, SRT / "hardware_backend/npu/moe/welmv4_megamoe.py"
    )
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve postponed annotations through the module registry.
    sys.modules[name] = module
    with patch.dict(sys.modules, {"sglang.srt.environ": NS(envs=ENV)}):
        spec.loader.exec_module(module)
    return module


RUNTIME = load_runtime()


def make_mlp():
    generator = torch.Generator().manual_seed(42)

    def weight(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.bfloat16) / 8

    config = NS(
        num_experts=4, num_local_experts=2, hidden_size=16,
        intermediate_size_per_partition=16, top_k=2, activation="silu",
        is_gated=True, gemm1_clamp_limit=None, apply_router_weight_on_input=False,
        gemm1_alpha=None, swiglu_limit=None, no_combine=False,
        routed_scaling_factor=None,
    )
    experts = NS(
        w13_weight=weight(2, 32, 16), w2_weight=weight(2, 16, 16),
        moe_runner_config=config, moe_ep_size=2, moe_ep_rank=0, moe_tp_size=1,
        _expert_storage_rank=0, _num_local_routed=2, num_local_experts=2,
        _num_global_routed=4, num_fused_shared_experts=0, with_bias=False,
        quant_method=NS(),
    )
    shared = NS(
        gate_up_proj=NS(weight=weight(32, 16), bias=None, tp_rank=0, tp_size=1),
        down_proj=NS(weight=weight(16, 16), bias=None, tp_rank=0, tp_size=1,
                     reduce_results=False),
        swiglu_clamp_limit=None, act_fn=SiluAndMul(),
    )
    return NS(
        layer_id=3, experts=experts, shared_expert=shared, shared_expert_gate=None,
        is_nextn=False, is_kv_mirror_consumer=False,
        welm_local_ep_kernel_available=True,
    )


def make_runtime(op):
    runtime = RUNTIME.WelmPrefillMegaMoE.__new__(RUNTIME.WelmPrefillMegaMoE)
    runtime.symm_buffer = object()
    runtime._mega_moe = op
    return runtime


class TestSharedWeights(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.layers.activation": NS(SiluAndMul=SiluAndMul),
        }))

    def prepare(self, mlp):
        return RUNTIME._prepare_shared_weights(mlp, ep_size=2, ep_rank=0)

    def test_bf16_uses_full_gate_then_up_storage(self):
        mlp = make_mlp()
        shared = self.prepare(mlp)
        self.assertEqual(shared.w1.shape, (1, 32, 16))
        self.assertEqual(shared.w2.shape, (1, 16, 16))
        self.assertEqual(
            shared.w1.data_ptr(), mlp.shared_expert.gate_up_proj.weight.data_ptr()
        )
        self.assertEqual(
            shared.w2.data_ptr(), mlp.shared_expert.down_proj.weight.data_ptr()
        )
        self.assertTrue(shared.w1.is_contiguous())
        x = torch.randn((3, 16), dtype=torch.bfloat16)
        gate_up = F.linear(x, mlp.shared_expert.gate_up_proj.weight)
        expected = F.linear(
            mlp.shared_expert.act_fn(gate_up),
            mlp.shared_expert.down_proj.weight,
        )
        gate, up = F.linear(x, shared.w1[0]).chunk(2, dim=-1)
        self.assertTrue(
            torch.equal(F.linear(F.silu(gate) * up, shared.w2[0]), expected)
        )

    def test_rejects_unsupported_semantics_and_layout(self):
        cases = (
            ("shared_expert_gate", lambda m: torch.nn.Linear(16, 1), "gate"),
            ("shared_expert", lambda m: None, "shared expert"),
            ("shared_expert.swiglu_clamp_limit", lambda m: 7, "clamp"),
            ("shared_expert.act_fn", lambda m: torch.nn.ReLU(), "SiluAndMul"),
            ("shared_expert.gate_up_proj.bias", lambda m: torch.zeros(32), "bias"),
            ("shared_expert.down_proj.tp_size", lambda m: 2, "complete replicas"),
            ("shared_expert.down_proj.reduce_results", lambda m: True, "reduce"),
            ("experts.moe_runner_config.gemm1_clamp_limit", lambda m: 2, "clamp"),
            ("experts._expert_storage_rank", lambda m: 1, "continuous EP"),
            ("experts.num_fused_shared_experts", lambda m: 1, "continuous EP"),
            (
                "experts.quant_method.load_up_proj_weight_first",
                lambda m: True, "gate-then-up",
            ),
            ("shared_expert.down_proj.weight", lambda m: torch.zeros(16, 16), "BF16"),
            (
                "shared_expert.down_proj.weight",
                lambda m: m.shared_expert.down_proj.weight.t(), "contiguous",
            ),
            (
                "shared_expert.gate_up_proj.weight",
                lambda m: torch.zeros(30, 16, dtype=torch.bfloat16), "shape",
            ),
            (
                "experts.w13_weight",
                lambda m: m.experts.w13_weight.to(torch.float8_e4m3fn), "BF16",
            ),
        )
        for path, value, message in cases:
            with self.subTest(message=message):
                mlp = make_mlp()
                target = mlp
                parts = path.split(".")
                for part in parts[:-1]:
                    target = getattr(target, part)
                setattr(target, parts[-1], value(mlp))
                with self.assertRaisesRegex(ValueError, f"layer 3:.*{message}"):
                    self.prepare(mlp)

    def test_fusion_passes_weights_and_adds_shared_once_without_route_scaling(self):
        mlp = make_mlp()
        shared = self.prepare(mlp)
        captured = {}

        def cpu_op(x, ids, weights, w1, w2, buffer, **kwargs):
            captured.update(kwargs)
            result = torch.zeros_like(x)
            for k in range(ids.shape[1]):
                first = torch.bmm(w1[0][ids[:, k].long()], x.unsqueeze(-1)).squeeze(-1)
                gate, up = first.chunk(2, dim=-1)
                expert = torch.bmm(
                    w2[0][ids[:, k].long()], (F.silu(gate) * up).unsqueeze(-1)
                ).squeeze(-1)
                result += expert * weights[:, k, None]
            for index in range(kwargs["l1_weights_shared"][0].shape[0]):
                gate, up = F.linear(x, kwargs["l1_weights_shared"][0][index]).chunk(
                    2, dim=-1
                )
                result += F.linear(
                    F.silu(gate) * up, kwargs["l2_weights_shared"][0][index]
                )
            return result, torch.zeros(2, dtype=torch.int32)

        runtime = make_runtime(cpu_op)
        x = torch.randn((3, 16), dtype=torch.bfloat16)
        topk = NS(topk_ids=torch.tensor([[0, 1]] * 3), topk_weights=torch.zeros(3, 2))
        output = runtime.forward_layer(mlp.experts, x, topk, 3, shared_weights=shared)
        expected = F.linear(
            mlp.shared_expert.act_fn(F.linear(x, shared.w1[0])), shared.w2[0]
        )
        self.assertTrue(torch.equal(output, expected))
        self.assertIs(captured["l1_weights_shared"][0], shared.w1)
        for name in (
            "l1_weights_sf", "l2_weights_sf", "l1_weights_shared_sf",
            "l2_weights_shared_sf",
        ):
            self.assertIsNone(captured[name])

    def test_padding_masks_complete_output_and_preserves_legal_ids(self):
        mlp = make_mlp()
        shared = self.prepare(mlp)
        x = torch.zeros(4, 16, dtype=torch.bfloat16)
        topk = NS(topk_ids=torch.tensor([[0, 1]] * 4), topk_weights=torch.ones(4, 2))
        for mask in (None, torch.tensor([True, False, True, False])):
            with self.subTest(mask=mask):
                op = Mock(
                    side_effect=lambda x, *_args, **_kwargs: (torch.ones_like(x), None)
                )
                output = make_runtime(op).forward_layer(
                    mlp.experts, x, topk, 2 if mask is None else 4,
                    valid_row_mask=mask, shared_weights=shared,
                )
                valid = (
                    torch.tensor([True, True, False, False]) if mask is None else mask
                )
                self.assertTrue(torch.equal(output[:, 0], valid.to(torch.bfloat16)))
                ids, weights = op.call_args.args[1:3]
                self.assertEqual(ids.dtype, torch.int32)
                self.assertTrue(torch.equal(ids, topk.topk_ids))
                self.assertTrue(torch.equal(weights[:, 0], valid.to(torch.bfloat16)))
                self.assertTrue(torch.all(topk.topk_weights == 1))
                # DP owns the final output mask when it opts out here.
                unmasked = make_runtime(op).forward_layer(
                    mlp.experts, x, topk, 2, valid_row_mask=mask,
                    shared_weights=shared, zero_output_padding=False,
                )
                self.assertTrue(torch.all(unmasked == 1))

    def test_missing_shared_weights_rejected_before_operator(self):
        mlp = make_mlp()
        x = torch.zeros(1, 16, dtype=torch.bfloat16)
        topk = NS(topk_ids=torch.tensor([[0, 1]]), topk_weights=torch.ones(1, 2))
        op = Mock(return_value=(x, None))
        with self.assertRaisesRegex(RuntimeError, "prepared shared weights"):
            make_runtime(op).forward_layer(mlp.experts, x, topk, 1)
        op.assert_not_called()

    def test_disabled_does_not_read_threshold_or_import_operator(self):
        self.enterContext(patch.object(ENV, "WELM_NPU_USE_MEGAMOE", env(False)))
        threshold = Mock(side_effect=ValueError("invalid threshold text"))
        self.enterContext(patch.object(
            ENV, "WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD", NS(get=threshold)
        ))
        with patch("builtins.__import__") as import_module:
            # No model or runtime prerequisites are needed while disabled.
            self.assertIsNone(RUNTIME.init_welm_prefill_megamoe(object()))
        threshold.assert_not_called()
        import_module.assert_not_called()

    def test_threshold_is_strict_and_nonpositive_disables_filter(self):
        runtime = make_runtime(Mock())
        runtime.prefill_token_threshold = 128
        self.assertFalse(runtime.meets_prefill_threshold(127))
        self.assertFalse(runtime.meets_prefill_threshold(128))
        self.assertTrue(runtime.meets_prefill_threshold(129))
        for threshold in (0, -1):
            runtime.prefill_token_threshold = threshold
            self.assertTrue(runtime.meets_prefill_threshold(1))

    def test_old_wrapper_rejected_before_call(self):
        with self.assertRaisesRegex(RuntimeError, "missing parameters"):
            RUNTIME._check_shared_abi(lambda x, **kwargs: None)

        def new_wrapper(*args, l1_weights_shared=None, l2_weights_shared=None,
                        l1_weights_shared_sf=None, l2_weights_shared_sf=None):
            pass

        RUNTIME._check_shared_abi(new_wrapper)

    def test_identity_mapping_allowed_but_remapping_and_eplb_rejected(self):
        mlp = make_mlp()
        metadata = NS(
            physical_to_logical_map_cpu=torch.arange(4).repeat(4, 1),
            ep_size=2, num_physical_experts=4, num_logical_experts=4,
        )
        RUNTIME._validate_expert_mapping([mlp], metadata, ep_size=2, enable_eplb=False)
        RUNTIME._validate_expert_mapping([mlp], None, ep_size=2, enable_eplb=False)
        metadata.physical_to_logical_map_cpu[3] = torch.tensor([1, 0, 2, 3])
        with self.assertRaisesRegex(ValueError, "layer 3:.*remapped"):
            RUNTIME._validate_expert_mapping(
                [mlp], metadata, ep_size=2, enable_eplb=False
            )
        with self.assertRaisesRegex(ValueError, "dynamic EPLB"):
            RUNTIME._validate_expert_mapping([mlp], None, ep_size=2, enable_eplb=True)
        metadata.physical_to_logical_map_cpu[3] = torch.arange(4)
        metadata.num_physical_experts = 8
        with self.assertRaisesRegex(ValueError, "redundant"):
            RUNTIME._validate_expert_mapping(
                [mlp], metadata, ep_size=2, enable_eplb=False
            )

    def test_init_prepares_one_window_and_shared_views_before_forward(self):
        events = []
        threshold = Mock(return_value=128)
        self.enterContext(patch.object(
            ENV, "WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD", NS(get=threshold)
        ))

        def wrapper(*args, l1_weights_shared=None, l2_weights_shared=None,
                    l1_weights_shared_sf=None, l2_weights_shared_sf=None):
            raise AssertionError("init must not execute the operator")

        def create_buffer(*args, **kwargs):
            events.append("buffer")
            self.assertEqual(kwargs["dispatch_quant_mode"], 0)
            self.assertEqual(kwargs["dispatch_quant_out_dtype"], 28)
            return NS(destroy=Mock())

        group = Mock()
        group._get_backend.return_value.get_hccl_comm_name.side_effect = (
            lambda *_args, **_kwargs: events.append("hccl")
        )
        coordinator = NS(device_group=group)
        mlp = make_mlp()
        runner = NS(
            model=NS(model=NS(layers=[NS(mlp=mlp)])),
            runner_parallel_plan=NS(moe_ep_normal_group=coordinator),
            server_args=NS(enable_eplb=False),
        )
        identity = NS(
            physical_to_logical_map_cpu=torch.arange(4).repeat(4, 1),
            ep_size=2, num_physical_experts=4, num_logical_experts=4,
        )
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.distributed": NS(get_moe_ep_group=lambda: coordinator,
                get_moe_ep_normal_comm_group=lambda: coordinator),
            "sglang.srt.layers.moe": NS(
                get_moe_a2a_backend=lambda: NS(is_deepep=lambda: True)
            ),
            "sglang.srt.eplb.expert_location": NS(
                get_global_expert_location_metadata=lambda: identity
            ),
            "npu_ops_transformer.ops.mega_moe": NS(mega_moe=wrapper,
                get_symm_buffer_for_mega_moe=create_buffer),
            "npu_ops_transformer.ops.comm_context": NS(comm_context_op_builder=NS(
                load=lambda: events.append("load"))),
        }))
        self.enterContext(patch.dict(RUNTIME._RUNTIMES, clear=True))
        self.enterContext(patch.object(torch.distributed, "get_rank", return_value=0))
        self.enterContext(patch.object(
            torch.distributed, "get_world_size", return_value=2
        ))
        self.enterContext(patch.object(
            torch.distributed, "barrier",
            side_effect=lambda **_: events.append("barrier"),
        ))
        self.enterContext(patch.object(
            torch, "npu", NS(synchronize=lambda _: events.append("sync")), create=True
        ))
        runtime = RUNTIME.init_welm_prefill_megamoe(runner)
        self.assertEqual(events, ["load", "hccl", "barrier", "sync", "buffer"])
        self.assertEqual(runtime.prefill_token_threshold, 128)
        threshold.assert_called_once_with()
        self.assertIs(mlp.welm_prefill_megamoe, runtime)
        self.assertEqual(mlp.welm_megamoe_shared_weights.w1.data_ptr(),
                         mlp.shared_expert.gate_up_proj.weight.data_ptr())
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            RUNTIME.init_welm_prefill_megamoe(runner)
        runtime.close()
        runtime.close()
        runtime.symm_buffer.destroy.assert_called_once()

    def test_no_eligible_layers_and_quantized_route_fail_before_runtime(self):
        self.enterContext(patch.dict(sys.modules, {
            "sglang.srt.distributed": NS(get_moe_ep_group=lambda: None,
                get_moe_ep_normal_comm_group=lambda: None),
            "sglang.srt.layers.moe": NS(
                get_moe_a2a_backend=lambda: NS(is_deepep=lambda: True)
            ),
        }))
        runner = NS(model=NS(model=NS(layers=[])))
        with self.assertRaisesRegex(ValueError, "no eligible"):
            RUNTIME.init_welm_prefill_megamoe(runner)
        mlp = make_mlp()
        mlp.experts.w13_weight = mlp.experts.w13_weight.to(torch.float8_e4m3fn)
        runner.model.model.layers = [NS(mlp=mlp)]
        with self.assertRaisesRegex(ValueError, "layer 3:.*BF16"):
            RUNTIME.init_welm_prefill_megamoe(runner)


if __name__ == "__main__":
    unittest.main()
