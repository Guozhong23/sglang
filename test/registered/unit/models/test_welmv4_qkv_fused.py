import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor import forward_context
from sglang.srt.models import welmv4
from sglang.srt.models.welmv4 import Qwen2MoeAttention
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakePool:
    def __init__(self, rows=128):
        self.k_cache = torch.empty(rows, 1, 256, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)

    def get_key_buffer(self, _layer_id):
        return self.k_cache

    def get_value_buffer(self, _layer_id):
        return self.v_cache


class TestWeLMv4FusedQKV(unittest.TestCase):
    def setUp(self):
        welmv4._WELMV4_FUSED_QKV_PROGRAMS.clear()

    @staticmethod
    def _attention():
        unquantized_method = type("UnquantizedLinearMethod", (), {})()
        return SimpleNamespace(
            q_norm=None,
            k_norm=SimpleNamespace(
                weight=torch.empty(256, dtype=torch.bfloat16),
                eps=1e-6,
            ),
            head_dim=256,
            qk_rope_head_dim=64,
            num_heads=6,
            num_kv_heads=1,
            q_size=1536,
            kv_size=256,
            rotary_emb=SimpleNamespace(
                is_neox_style=True,
                head_size=256,
                rotary_dim=64,
                cos_sin_cache=torch.empty(32, 64, dtype=torch.float32),
            ),
            qkv_proj=SimpleNamespace(
                quant_method=unquantized_method,
                weight=torch.empty(2048, 2048, dtype=torch.bfloat16),
                bias=None,
            ),
            attn=SimpleNamespace(layer_id=3),
        )

    @staticmethod
    def _batch(slot_mapping, batch_size=1):
        return SimpleNamespace(
            batch_size=batch_size,
            forward_mode=SimpleNamespace(
                is_extend_without_speculative=lambda: True
            ),
            out_cache_loc=slot_mapping,
        )

    def test_aot_artifact_is_reused_for_dynamic_m_and_swa_slots(self):
        attention = self._attention()
        pool = _FakePool()
        backend = SimpleNamespace(
            _is_swa_layer=lambda _layer: True,
            forward_metadata=SimpleNamespace(swa_out_cache_loc=None),
        )
        compile_calls = []
        launch_slot_mappings = []

        def compile_aot(*args, **kwargs):
            compile_calls.append((args, kwargs))

            def launch(*launch_args):
                launch_slot_mappings.append(launch_args[5])

            return launch

        operator_module = types.ModuleType("fused_qkv_proj_norm_rope_cache")
        operator_module.compile_aot = compile_aot

        with (
            envs.SGLANG_NPU_WELMV4_FUSED_QKV.override(True),
            patch.object(welmv4, "_is_npu", True),
            patch.object(forward_context, "get_attn_backend", return_value=backend),
            patch.object(
                forward_context, "get_token_to_kv_pool", return_value=pool
            ),
            patch.dict(
                sys.modules,
                {"fused_qkv_proj_norm_rope_cache": operator_module},
            ),
        ):
            for num_tokens in (2, 3):
                positions = torch.arange(num_tokens, dtype=torch.int64)
                full_slots = torch.arange(20, 20 + num_tokens, dtype=torch.int64)
                swa_slots = torch.arange(4, 4 + num_tokens, dtype=torch.int64)
                backend.forward_metadata.swa_out_cache_loc = swa_slots
                output = Qwen2MoeAttention._try_npu_fused_qkv_prefill(
                    attention,
                    torch.empty(
                        num_tokens, 2048, dtype=torch.bfloat16
                    ).contiguous(),
                    positions,
                    self._batch(full_slots),
                    need_mirror=False,
                )
                self.assertEqual(
                    [tuple(t.shape) for t in output],
                    [
                        (num_tokens, 1536),
                        (num_tokens, 256),
                        (num_tokens, 256),
                    ],
                )

        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(
            compile_calls[0],
            (
                (2048, 128, 32),
                {"return_v": True, "positions_contiguous": True},
            ),
        )
        self.assertEqual(len(launch_slot_mappings), 2)
        torch.testing.assert_close(
            launch_slot_mappings[0], torch.tensor([4, 5], dtype=torch.int64)
        )
        torch.testing.assert_close(
            launch_slot_mappings[1], torch.tensor([4, 5, 6], dtype=torch.int64)
        )

    def test_unsupported_batch_and_quantization_fall_back(self):
        attention = self._attention()
        hidden = torch.empty(2, 2048, dtype=torch.bfloat16)
        positions = torch.arange(2, dtype=torch.int64)
        slots = torch.arange(2, dtype=torch.int64)
        with (
            envs.SGLANG_NPU_WELMV4_FUSED_QKV.override(True),
            patch.object(welmv4, "_is_npu", True),
        ):
            result = Qwen2MoeAttention._try_npu_fused_qkv_prefill(
                attention,
                hidden,
                positions,
                self._batch(slots, batch_size=2),
                need_mirror=False,
            )
            self.assertIsNone(result)

            attention.qkv_proj.quant_method = SimpleNamespace()
            result = Qwen2MoeAttention._try_npu_fused_qkv_prefill(
                attention,
                hidden,
                positions,
                self._batch(slots),
                need_mirror=False,
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
