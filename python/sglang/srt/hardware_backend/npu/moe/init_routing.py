"""
NPU MoE init routing components.

Prepare token routing before expert computation. Two API versions are provided:
- v1: legacy routing using ``npu_moe_init_routing``.
- v2: improved routing using ``npu_moe_init_routing_v2``.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch

# ``npu_moe_init_routing_v2`` quant_mode selecting MXFP8: the op emits an
# float8_e4m3fn payload plus an e8m0 block scale, fusing the activation quant
# that would otherwise need a separate ``npu_dynamic_mx_quant`` pass.
MXFP8_QUANT_MODE = 3


def _normalize_mxfp_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape a flat 2D e8m0 block scale ``[N, M]`` into pair-split ``[N, M//2, 2]``.

    ``npu_moe_init_routing_v2(quant_mode=3)`` emits the scale flat, while the
    grouped matmul wants the pair-split view. Already-3D scales (what
    ``npu_dynamic_mx_quant`` returns) pass through untouched. Mirrors
    vllm-ascend's ``maybe_normalize_mxfp_scale_layout``.
    """
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


class BaseInitRouting(ABC):
    """Abstract base for NPU MoE init routing."""

    @abstractmethod
    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: ...


class NPUMoEInitRouting_v1(BaseInitRouting):
    """
    NPU MoE init routing (v1 API).

    Uses ``npu_moe_init_routing`` with a manually constructed ``row_idx`` tensor.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, None


class NPUMoEInitRouting_v2(BaseInitRouting):
    """
    NPU MoE init routing (v2 API).

    Uses ``npu_moe_init_routing_v2``, which integrates expert token counting.
    """

    def __init__(
        self,
        quant_mode: int = -1,
        expert_tokens_num_type: int = 1,
        active_expert_range: Optional[Tuple[int, int]] = None,
        mxfp8_quant_before_routing: bool = False,
    ):
        self.quant_mode = quant_mode
        self.expert_tokens_num_type = expert_tokens_num_type
        self.active_expert_range = active_expert_range
        self.mxfp8_quant_before_routing = mxfp8_quant_before_routing

    def _route_prequantized_mxfp8(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        active_expert_range,
        pre_quant_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize [T,H] before Top-K expansion and route payload/scale.

        MoeInitRouting's native MX mode expands BF16 rows first and then runs
        DynamicMxQuant on [T*top_k,H].  Instead, quantize [T,H], ask the routing
        op to sort a one-column source-token id, and use that permutation for
        both the FP8 payload and its E8M0 scale.  The one-column metadata route
        preserves the operator's exact expert ordering and expanded_row_idx
        contract without repeating the H-wide BF16 traffic.
        """
        num_tokens = hidden_states.shape[0]
        if pre_quant_input is None:
            quantized_states, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
                hidden_states.contiguous(), dst_type=torch.float8_e4m3fn
            )
        else:
            quantized_states, input_scale = pre_quant_input
            if (
                quantized_states.ndim != 2
                or quantized_states.shape != hidden_states.shape
            ):
                raise ValueError(
                    "Pre-quantized MXFP8 payload must match the MoE input shape, "
                    f"got payload={tuple(quantized_states.shape)} and "
                    f"input={tuple(hidden_states.shape)}."
                )
            if input_scale.shape[0] != num_tokens:
                raise ValueError(
                    "Pre-quantized MXFP8 scale must have one row per MoE token, "
                    f"got scale={tuple(input_scale.shape)} and tokens={num_tokens}."
                )
            if quantized_states.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "Pre-quantized MoE payload must use float8_e4m3fn, got "
                    f"{quantized_states.dtype}."
                )

        source_token_rows = torch.arange(
            num_tokens,
            dtype=torch.float32,
            device=hidden_states.device,
        ).view(num_tokens, 1)
        (
            expanded_token_rows,
            expanded_row_idx,
            expert_tokens,
            _,
        ) = torch.ops.npu.npu_moe_init_routing_v2(
            source_token_rows,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=self.expert_tokens_num_type,
            expert_tokens_num_flag=True,
            active_expert_range=active_expert_range,
            quant_mode=-1,
            row_idx_type=0,
        )

        # active_expert_range can leave an invalid static-shape suffix. Make
        # every gather index safe without a data-dependent slice; GMM consumes
        # only the valid prefix described by expert_tokens.
        source_token_idx = torch.nan_to_num(
            expanded_token_rows.squeeze(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_(0.0, float(num_tokens - 1))
        source_token_idx = source_token_idx.to(torch.int64)
        expanded_states = quantized_states.index_select(0, source_token_idx)
        expanded_scale = input_scale.index_select(0, source_token_idx)
        return (
            expanded_states,
            expanded_row_idx,
            expert_tokens.to(torch.int64),
            expanded_scale,
        )

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        top_k: int,
        mxfp8_pre_quant_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]
        if self.active_expert_range is None:
            # Preserve the original full-range path exactly for existing
            # dispatchers. Only the opt-in local-EP dispatcher supplies a
            # narrower range.
            active_expert_range = [0, num_experts]
        else:
            active_expert_range = list(self.active_expert_range)
            if not (
                0
                <= active_expert_range[0]
                < active_expert_range[1]
                <= num_experts
            ):
                raise ValueError(
                    "active_expert_range must be a non-empty sub-range of "
                    f"[0, {num_experts}], got {active_expert_range}"
                )

        if (
            self.quant_mode == MXFP8_QUANT_MODE
            and (self.mxfp8_quant_before_routing or mxfp8_pre_quant_input is not None)
            and num_tokens > 0
        ):
            return self._route_prequantized_mxfp8(
                hidden_states,
                topk_ids,
                num_experts,
                top_k,
                active_expert_range,
                pre_quant_input=mxfp8_pre_quant_input,
            )

        hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * top_k,
                expert_num=num_experts,
                expert_tokens_num_type=self.expert_tokens_num_type,
                expert_tokens_num_flag=True,
                active_expert_range=active_expert_range,
                quant_mode=self.quant_mode,
            )
        )
        if self.quant_mode == -1:
            pertoken_scale = None
        elif self.quant_mode == MXFP8_QUANT_MODE:
            pertoken_scale = _normalize_mxfp_scale(pertoken_scale)
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale


class NPUMoEInitRouting_Quant(BaseInitRouting):
    """
    NPU MoE init routing (Quant API).

    Uses ``npu_moe_init_routing_quant``, which integrates expert token counting.
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _, pertoken_scale = (
            torch.ops.npu.npu_moe_init_routing_quant(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_mode=1,
                expert_tokens_before_capacity_flag=False,
                quant_mode=1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens, pertoken_scale
