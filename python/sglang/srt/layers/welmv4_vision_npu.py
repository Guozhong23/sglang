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
    mc2: bool = False
    gelu: bool = False
    add_layer_norm: bool = False
    mc2_min_m: int = 128
    mc2_comm_mode: str = "ccu"

    def __post_init__(self):
        if self.mc2_min_m < 1:
            raise ValueError("SGLANG_VLM_NPU_MC2_MIN_M must be positive")
        if self.mc2_comm_mode not in {"ccu", "ai_cpu"}:
            raise ValueError("SGLANG_VLM_NPU_MC2_COMM_MODE must be ccu or ai_cpu")

    @classmethod
    def from_env(cls) -> "VisionNPUFusionConfig":
        return cls(
            mc2=_env_bool("SGLANG_VLM_NPU_USE_MC2"),
            gelu=_env_bool("SGLANG_VLM_NPU_FUSE_GELU"),
            add_layer_norm=_env_bool("SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM"),
            mc2_min_m=int(os.environ.get("SGLANG_VLM_NPU_MC2_MIN_M", "128")),
            mc2_comm_mode=os.environ.get("SGLANG_VLM_NPU_MC2_COMM_MODE", "ccu"),
        )

    def validate_runtime(self, device: torch.device) -> None:
        if device.type != "npu":
            return
        if self.mc2:
            import torch_npu

            if not callable(getattr(torch_npu, "npu_mm_all_reduce_base", None)):
                raise RuntimeError(
                    "Vision MC2 requires torch_npu.npu_mm_all_reduce_base"
                )
        if self.gelu or self.add_layer_norm:
            from sglang.srt.layers import welmv4_vision_npu_kernels

            if self.gelu:
                welmv4_vision_npu_kernels.validate_gelu_runtime()


def _mc2_eligible(layer, hidden: torch.Tensor, config: VisionNPUFusionConfig) -> bool:
    return (
        config.mc2
        and hidden.device.type == "npu"
        and hidden.dtype == torch.bfloat16
        and hidden.ndim >= 2
        and hidden.shape[-1] > 0
        and hidden.numel() // hidden.shape[-1] >= config.mc2_min_m
        and hidden.is_contiguous()
        and layer.tp_size > 1
        and layer.input_is_parallel
        and layer.reduce_results
        and not layer.use_dp_attention_reduce
        and not getattr(layer, "use_decode_attn_tp", False)
        and layer.quant_method.__class__.__name__ == "UnquantizedLinearMethod"
        and layer.weight.ndim == 2
        and layer.weight.dtype == hidden.dtype
        and layer.weight.device == hidden.device
        and layer.weight.is_contiguous()
    )


def vision_row_parallel_linear(
    layer, hidden: torch.Tensor, config: VisionNPUFusionConfig
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Return None before any collective when the ordinary row-linear path is needed."""
    if not _mc2_eligible(layer, hidden, config):
        return None
    config.validate_runtime(hidden.device)
    import torch_npu

    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    if group.world_size != layer.tp_size or group.rank_in_group != layer.tp_rank:
        raise RuntimeError("Vision MC2 row-linear does not match the current TP group")
    weight = layer.weight
    if hidden.shape[-1] != weight.shape[1]:
        raise ValueError("Vision MC2 input width does not match the weight")
    process_group = group.device_group
    cached = getattr(layer, "_vision_mc2_hcom", None)
    if cached is None or cached[0] is not process_group:
        backend = process_group._get_backend(hidden.device)
        name = backend.get_hccl_comm_name(process_group.rank())
        cached = (process_group, name)
        layer._vision_mc2_hcom = cached
    bias = layer.bias if layer.tp_rank == 0 and not layer.skip_bias_add else None
    output = torch_npu.npu_mm_all_reduce_base(
        hidden.reshape(-1, hidden.shape[-1]),
        weight.transpose(0, 1),
        cached[1],
        reduce_op="sum",
        bias=bias,
        comm_turn=0,
        comm_mode=config.mc2_comm_mode,
    )
    output = output.reshape(*hidden.shape[:-1], weight.shape[0])
    return output, layer.bias if layer.skip_bias_add else None


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
        and layer.weight.dtype == hidden.dtype
        and layer.weight.device == hidden.device
        and layer.weight.is_contiguous()
        and hidden.shape[-1] == layer.weight.shape[-1]
        and (
            layer.bias is None
            or (
                layer.bias.dtype == hidden.dtype
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
    """Normalize the dtype-rounded residual sum, matching eager BF16 semantics."""
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
        and norm.weight.device == hidden.device
        and norm.bias.device == hidden.device
        and norm.weight.is_contiguous()
        and norm.bias.is_contiguous()
        and hidden.numel() > 0
        and hidden.shape[-1] <= 8192
    ):
        from sglang.srt.layers.welmv4_vision_npu_kernels import add_layer_norm

        return add_layer_norm(hidden, residual, norm.weight, norm.bias, norm.eps)
    summed = hidden + residual
    return summed, norm(summed)
