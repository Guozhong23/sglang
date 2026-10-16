from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch


def _env_bool(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be 0, 1, false or true; got {value!r}")
    return value in {"1", "true"}


@dataclass(frozen=True)
class VisionNPUFusionConfig:
    rope: bool = False
    gelu: bool = False
    add_layer_norm: bool = False

    @classmethod
    def from_env(cls) -> "VisionNPUFusionConfig":
        if _env_bool("SGLANG_VLM_NPU_USE_MC2"):
            raise ValueError(
                "ViT MC2 has been removed; set SGLANG_VLM_NPU_USE_MC2=0. "
                "Vision projections use the ordinary RowParallelLinear path."
            )
        return cls(
            rope=_env_bool("SGLANG_VLM_NPU_FUSE_ROPE"),
            gelu=_env_bool("SGLANG_VLM_NPU_FUSE_GELU"),
            add_layer_norm=_env_bool("SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM"),
        )

    def validate_runtime(self, device: torch.device) -> None:
        if device.type != "npu":
            return
        if self.gelu or self.add_layer_norm or self.rope:
            from sglang.srt.layers import welmv4_vision_npu_kernels

            if self.gelu:
                welmv4_vision_npu_kernels.validate_gelu_runtime()


def vision_column_parallel_linear_gelu(
    layer, hidden: torch.Tensor, approximate: str, config: VisionNPUFusionConfig
) -> Optional[torch.Tensor]:
    if approximate not in {"none", "tanh", "quick"}:
        raise ValueError(f"Unsupported vision GELU mode: {approximate}")
    if not (
        config.gelu
        and approximate != "quick"
        and hidden.device.type == "npu"
        and hidden.dtype == torch.bfloat16
        and hidden.ndim >= 2
        and hidden.numel() > 0
        and hidden.is_contiguous()
        and not layer.gather_output
        and not layer.skip_bias_add
        and layer.quant_method.__class__.__name__ == "UnquantizedLinearMethod"
        and layer.weight.ndim == 2
        and layer.weight.shape[0] > 0
        and layer.weight.dtype == hidden.dtype
        and layer.weight.device == hidden.device
        and layer.weight.is_contiguous()
        and hidden.shape[-1] == layer.weight.shape[-1]
        and (
            layer.bias is None
            or (
                layer.bias.shape == (layer.weight.shape[0],)
                and layer.bias.dtype == hidden.dtype
                and layer.bias.device == hidden.device
                and layer.bias.is_contiguous()
            )
        )
    ):
        return None
    from sglang.srt.layers.welmv4_vision_npu_kernels import linear_gelu

    return linear_gelu(hidden, layer.weight, layer.bias, approximate)


def vision_add_layer_norm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    norm,
    config: VisionNPUFusionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the dtype-rounded residual sum, matching eager BF16 semantics.

    AdaLayerNormV2 scales/shifts *after* normalization, so it cannot fuse this
    residual addition. npu_add_layer_norm is a closer operation, but its API
    does not guarantee the BF16 sum boundary required by this model. The
    Triton implementation explicitly rounds before computing mean/variance.
    """
    if (
        config.add_layer_norm
        and hidden.device.type == "npu"
        and hidden.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and hidden.ndim >= 2
        and hidden.numel() > 0
        and hidden.shape == residual.shape
        and hidden.dtype == residual.dtype
        and hidden.device == residual.device
        and hidden.is_contiguous()
        and residual.is_contiguous()
        and tuple(norm.normalized_shape) == (hidden.shape[-1],)
        and norm.weight is not None
        and norm.bias is not None
        and norm.weight.device == hidden.device
        and norm.bias.device == hidden.device
        and norm.weight.dtype == hidden.dtype
        and norm.bias.dtype == hidden.dtype
        and norm.weight.is_contiguous()
        and norm.bias.is_contiguous()
        and 0 < hidden.shape[-1] <= 8192
    ):
        from sglang.srt.layers.welmv4_vision_npu_kernels import add_layer_norm

        return add_layer_norm(hidden, residual, norm.weight, norm.bias, norm.eps)
    summed = hidden + residual
    return summed, norm(summed)
