"""Fused matmul + sigmoid for the attention output gate on Ascend NPU.

``torch_npu.npu_fused_matmul(x, w.t(), fused_op_type="sigmoid")`` maps to CANN's
FusedMatMul with the sigmoid epilogue: it computes ``sigmoid(x @ w.t())`` in a
single kernel, accumulating and evaluating the sigmoid in fp32 before casting
back to bf16/fp16.

The epilogue only exists on Ascend 950 (arch35 / DAV_3510) and the CANN
operator is still WIP, so the path is opt-in
(``SGLANG_NPU_USE_FUSED_MM_SIGMOID``), guarded by shape/dtype/quantization
checks, and silently falls back to ``sigmoid(linear(x))`` on any failure.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_npu

try:  # torch_npu only exists in the Ascend PyTorch build
    import torch_npu
except Exception:  # pragma: no cover - depends on the host installation
    torch_npu = None

logger = logging.getLogger(__name__)

# Set once the fused kernel is known to be unusable so that following tokens do
# not pay for a failing aclnn launch again.
_disabled_reason: Optional[str] = None


def _npu_available() -> bool:
    try:
        return is_npu()
    except RuntimeError:
        # torch_npu is installed but no visible device (e.g. CPU-only CI).
        return False


def _has_fused_matmul() -> bool:
    return torch_npu is not None and hasattr(torch_npu, "npu_fused_matmul")


def _flag_on() -> bool:
    """Read the opt-in flag; treat a missing field as OFF, never crash."""
    flag = getattr(envs, "SGLANG_NPU_USE_FUSED_MM_SIGMOID", None)
    return flag is not None and bool(flag.get())


def is_enabled() -> bool:
    """Whether the fused path may be used, without touching a device tensor."""
    return bool(
        _disabled_reason is None
        and _flag_on()
        and _npu_available()
        and _has_fused_matmul()
    )


def _is_unquantized(proj: torch.nn.Module) -> bool:
    quant_method = getattr(proj, "quant_method", None)
    if quant_method is None:
        return True
    # Imported lazily so this module stays cheap to import on non-NPU hosts.
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    return isinstance(quant_method, UnquantizedLinearMethod)


def _can_fuse(proj: torch.nn.Module, x: torch.Tensor) -> bool:
    if not is_enabled():
        return False
    # FusedMatMul(sigmoid) is 2D-only and BF16/FP16-only.
    if x.dim() != 2 or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not x.is_contiguous() or x.device.type != "npu":
        return False
    weight = getattr(proj, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype != x.dtype:
        return False
    # The epilogue accepts a bias, but every gate projection here is bias-free;
    # keep the fast path limited to that case.
    if getattr(proj, "bias", None) is not None:
        return False
    return _is_unquantized(proj)


def matmul_sigmoid(proj: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return ``sigmoid(proj(x))``, fused into one kernel when possible."""
    global _disabled_reason

    if _can_fuse(proj, x):
        try:
            return torch_npu.npu_fused_matmul(
                x, proj.weight.t(), fused_op_type="sigmoid"
            )
        except Exception as exc:  # aclnn failure must not break the model
            _disabled_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "FusedMatMul(sigmoid) failed; falling back to sigmoid(linear(x)) "
                "for the rest of the process. Error: %s",
                _disabled_reason,
            )

    out = proj(x)
    if isinstance(out, tuple):
        out = out[0]
    return torch.sigmoid(out)


__all__ = ["is_enabled", "matmul_sigmoid"]
