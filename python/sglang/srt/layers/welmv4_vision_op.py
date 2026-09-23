# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Device-independent reference operations for the WeLM vision encoder.

Keep these separate from the decoder's fused kernels: vision uses 72-wide
heads and fp32 rotary coefficients, unlike the text model's partial RoPE.
All operations dispatch through PyTorch and work on CPU and Ascend NPU.
"""

from __future__ import annotations

import torch


def welmv4_vision_quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Compute QuickGELU in fp32, rounding only the final result to x.dtype."""
    x_fp32 = x.float()
    return (x_fp32 * torch.sigmoid(1.702 * x_fp32)).to(x.dtype)


def welmv4_vision_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply half-rotate RoPE with fp32 coefficients and bf16 Q/K outputs.

    The final dimension may be any positive even size (including 72).
    Coefficients may be half-width or the duplicated full-width form used
    by VisionAttention. No intermediate product is rounded to bf16.
    """
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("WeLM vision RoPE requires bfloat16 query and key tensors")
    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError(
            "query and key must have equal (tokens, heads, head_dim) shapes"
        )
    head_dim = q.shape[-1]
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
    half = head_dim // 2
    if cos.shape != sin.shape or cos.shape not in (
        (q.shape[0], half),
        (q.shape[0], head_dim),
    ):
        raise ValueError("cos and sin must have shape (tokens, head_dim/2 or head_dim)")
    if not (q.device == k.device == cos.device == sin.device):
        raise ValueError("query, key, cos and sin must be on the same device")
    cos = cos[:, :half].float().unsqueeze(1)
    sin = sin[:, :half].float().unsqueeze(1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        lo, hi = x.float().chunk(2, dim=-1)
        return torch.cat((lo * cos - hi * sin, lo * sin + hi * cos), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)
