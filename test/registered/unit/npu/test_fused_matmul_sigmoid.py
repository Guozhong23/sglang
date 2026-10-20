"""Unit tests for sglang.srt.hardware_backend.npu.fused_matmul."""

import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import fused_matmul
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=5, suite="stage-a-unit-test-npu")


class _FakeProj(nn.Module):
    """Minimal stand-in for a bias-free ColumnParallelLinear gate projection."""

    def __init__(self, in_features=16, out_features=4, bias=False, quant_method=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)
        self.quant_method = (
            UnquantizedLinearMethod() if quant_method is None else quant_method
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias), None


class _ConstProj(_FakeProj):
    """Projection that ignores its input, so mock tensors can hit the fallback."""

    def forward(self, x):
        return torch.zeros(1, 4, dtype=torch.bfloat16), None


def _fake_npu():
    npu = MagicMock()
    npu.npu_fused_matmul.return_value = torch.zeros(8, 4, dtype=torch.bfloat16)
    return npu


def _npu_tensor(dim=2, dtype=torch.bfloat16):
    x = MagicMock()
    x.dim.return_value = dim
    x.dtype = dtype
    x.is_contiguous.return_value = True
    x.device.type = "npu"
    return x


@contextmanager
def _fused_on(npu):
    with (
        envs.SGLANG_NPU_USE_FUSED_MM_SIGMOID.override(True),
        patch.object(fused_matmul, "torch_npu", npu),
        patch.object(fused_matmul, "is_npu", lambda: True),
    ):
        yield


class TestMatmulSigmoid(unittest.TestCase):
    def setUp(self):
        fused_matmul._disabled_reason = None
        self.x = torch.randn(8, 16, dtype=torch.bfloat16)
        self.proj = _FakeProj()

    def test_flag_off_matches_reference(self):
        with envs.SGLANG_NPU_USE_FUSED_MM_SIGMOID.override(False):
            out = fused_matmul.matmul_sigmoid(self.proj, self.x)
        ref = torch.sigmoid(F.linear(self.x, self.proj.weight))
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_flag_off_matches_old_sigmoid_mul_sequence(self):
        # Old: g = proj(x); attn = sigmoid(g) * attn
        # New: g = matmul_sigmoid(proj, x); attn = attn * g
        # With the flag off this must be bit-identical, which is what the
        # WeLM-v4 patch relies on for a safe fallback.
        attn = torch.randn(8, 4, 3, dtype=torch.bfloat16)
        old = attn.clone()
        old.mul_(torch.sigmoid(F.linear(self.x, self.proj.weight)).unsqueeze(-1))

        new = attn.clone()
        with envs.SGLANG_NPU_USE_FUSED_MM_SIGMOID.override(False):
            gate = fused_matmul.matmul_sigmoid(self.proj, self.x)
        new.mul_(gate.unsqueeze(-1))

        torch.testing.assert_close(new, old, rtol=0, atol=0)

    def test_flag_on_calls_fused_kernel(self):
        npu = _fake_npu()
        x = _npu_tensor()
        with _fused_on(npu):
            out = fused_matmul.matmul_sigmoid(self.proj, x)

        self.assertIs(out, npu.npu_fused_matmul.return_value)
        npu.npu_fused_matmul.assert_called_once()
        args, kwargs = npu.npu_fused_matmul.call_args
        self.assertIs(args[0], x)
        torch.testing.assert_close(args[1], self.proj.weight.t(), rtol=0, atol=0)
        self.assertEqual(kwargs, {"fused_op_type": "sigmoid"})

    def test_failure_disables_fast_path(self):
        npu = _fake_npu()
        npu.npu_fused_matmul.side_effect = RuntimeError("aclnnFusedMatmul missing")
        proj = _ConstProj()
        x = _npu_tensor()

        with _fused_on(npu):
            out = fused_matmul.matmul_sigmoid(proj, x)
            self.assertIsNotNone(fused_matmul._disabled_reason)
            self.assertFalse(fused_matmul.is_enabled())
            # The failed launch is not retried on the next token.
            out_next = fused_matmul.matmul_sigmoid(proj, x)

        self.assertEqual(npu.npu_fused_matmul.call_count, 1)
        torch.testing.assert_close(out, torch.full((1, 4), 0.5, dtype=torch.bfloat16))
        torch.testing.assert_close(out_next, out)

    def test_can_fuse_rejects_unsupported_cases(self):
        npu = _fake_npu()
        non_contiguous = _npu_tensor()
        non_contiguous.is_contiguous.return_value = False
        cpu_tensor = _npu_tensor()
        cpu_tensor.device.type = "cpu"

        with _fused_on(npu):
            self.assertFalse(fused_matmul._can_fuse(self.proj, _npu_tensor(dim=3)))
            self.assertFalse(
                fused_matmul._can_fuse(self.proj, _npu_tensor(dtype=torch.float32))
            )
            self.assertFalse(fused_matmul._can_fuse(self.proj, non_contiguous))
            self.assertFalse(fused_matmul._can_fuse(self.proj, cpu_tensor))
            self.assertFalse(
                fused_matmul._can_fuse(_FakeProj(bias=True), _npu_tensor())
            )
            self.assertFalse(
                fused_matmul._can_fuse(_FakeProj(quant_method=object()), _npu_tensor())
            )
            self.assertTrue(fused_matmul._can_fuse(self.proj, _npu_tensor()))


if __name__ == "__main__":
    unittest.main()
