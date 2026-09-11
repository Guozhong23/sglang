"""Ascend 950 MegaMoE integration for WeLM MXFP8 routed experts.

The vendor operator owns the whole EP routed-expert chain:

    Dispatch(MXFP8) -> GMM1 -> SwiGLU+MXFP8 -> GMM2 -> Combine

Shared experts intentionally stay outside this module so the model can run
them on its existing independent stream. The registered communication buffer
is process-wide because the vendor runtime permits only one MegaMoE context
per HCCL group and all WeLM layers have the same expert geometry.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_resources, get_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


logger = logging.getLogger(__name__)

_STATE_KEY = "npu_megamoe_state"
_MAX_ACTUAL_BS = 16384
_MXFP8_E4M3_TYPE = 24


def _load_vendor_api():
    # Eager package import JIT-builds every extension unless minimal mode is
    # selected before npu_ops_transformer is imported for the first time.
    os.environ.setdefault("NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE", "minimal")
    try:
        from npu_ops_transformer.ops.mega_moe import (
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )
    except Exception as exc:
        raise RuntimeError(
            "Ascend MegaMoE requires npu_ops_transformer and the matching "
            "custom_transformer CANN operator package. Install the wheel and "
            ".run package, then export custom_transformer/op_api/lib through "
            "LD_LIBRARY_PATH before launching SGLang."
        ) from exc
    return get_symm_buffer_for_mega_moe, mega_moe


def _derive_max_tokens_per_rank(ep_size: int) -> int:
    configured = envs.SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK.get()
    if configured > 0:
        return configured

    args = get_server_args()
    max_global_tokens = 0
    max_prefill_buffer_tokens = getattr(args, "max_prefill_buffer_tokens", None)
    if callable(max_prefill_buffer_tokens):
        max_global_tokens = int(max_prefill_buffer_tokens() or 0)
    max_global_tokens = max(
        max_global_tokens,
        int(getattr(args, "max_prefill_tokens", 0) or 0),
        int(getattr(args, "chunked_prefill_size", 0) or 0),
    )
    if max_global_tokens <= 0:
        raise RuntimeError(
            "Cannot derive MegaMoE token capacity. Set "
            "SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK to the maximum LOCAL "
            "prefill rows on one EP rank."
        )
    return max(1, math.ceil(max_global_tokens / ep_size))


def _get_state():
    buffers = get_resources().buffers
    state = buffers.get(_STATE_KEY)
    if state is None:
        state = SimpleNamespace(
            sym_buffer=None,
            geometry=None,
            max_tokens_per_rank=None,
            destroy_registered=False,
        )
        buffers[_STATE_KEY] = state
    return state


def _destroy_state(state) -> None:
    sym_buffer = getattr(state, "sym_buffer", None)
    if sym_buffer is None:
        return
    state.sym_buffer = None
    try:
        sym_buffer.destroy()
    except Exception:
        # Interpreter shutdown may already have torn down torch.distributed.
        logger.debug("Ignoring MegaMoE buffer cleanup failure", exc_info=True)


def _get_symm_buffer(layer: "FusedMoE"):
    state = _get_state()
    ep_group = get_moe_ep_group().device_group
    ep_size = torch.distributed.get_world_size(ep_group)
    geometry = (
        id(ep_group),
        int(layer.num_experts),
        int(layer.top_k),
        int(layer.hidden_size),
        int(layer.intermediate_size_per_partition),
        int(ep_size),
    )
    if state.sym_buffer is not None:
        if state.geometry != geometry:
            raise RuntimeError(
                "One HCCL group can own only one MegaMoE context, but a second "
                f"expert geometry was requested: {state.geometry} vs {geometry}."
            )
        return state.sym_buffer

    if ep_size < 2 or layer.num_experts % ep_size != 0:
        raise RuntimeError(
            f"MegaMoE requires EP >= 2 and num_experts divisible by EP, got "
            f"experts={layer.num_experts}, ep={ep_size}."
        )
    max_tokens_per_rank = _derive_max_tokens_per_rank(ep_size)

    # SymmBuffer obtains an already initialized HCCL communicator name.
    rank = torch.distributed.get_rank(ep_group)
    ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(
        rank, init_comm=True
    )
    get_buffer, _ = _load_vendor_api()
    state.sym_buffer = get_buffer(
        ep_group,
        num_experts=layer.num_experts,
        num_max_tokens_per_rank=max_tokens_per_rank,
        num_topk=layer.top_k,
        hidden=layer.hidden_size,
        intermediate_hidden=0,
        max_recv_token_num=0,
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=_MXFP8_E4M3_TYPE,
        combine_quant_mode=0,
        comm_alg="",
    )
    state.geometry = geometry
    state.max_tokens_per_rank = max_tokens_per_rank
    if not state.destroy_registered:
        atexit.register(_destroy_state, state)
        state.destroy_registered = True
    logger.info(
        "Initialized Ascend MegaMoE: experts=%d local_experts=%d topk=%d "
        "hidden=%d intermediate=%d max_local_tokens=%d",
        layer.num_experts,
        layer.num_experts // ep_size,
        layer.top_k,
        layer.hidden_size,
        layer.intermediate_size_per_partition,
        max_tokens_per_rank,
    )
    return state.sym_buffer


def _validate_inputs(
    layer: "FusedMoE",
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != layer.hidden_size:
        raise RuntimeError(
            f"MegaMoE x must be [M, {layer.hidden_size}], got "
            f"{tuple(hidden_states.shape)}."
        )
    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(
            f"MegaMoE currently requires BF16 x, got {hidden_states.dtype}."
        )
    if hidden_states.shape[0] == 0 or hidden_states.shape[0] > _MAX_ACTUAL_BS:
        raise RuntimeError(
            f"MegaMoE local BS must be in [1, {_MAX_ACTUAL_BS}], got "
            f"{hidden_states.shape[0]}. Ensure WeLM scattered prefill is active."
        )
    expected_topk_shape = (hidden_states.shape[0], layer.top_k)
    if tuple(topk_ids.shape) != expected_topk_shape:
        raise RuntimeError(
            f"MegaMoE topk_ids must be {expected_topk_shape}, got "
            f"{tuple(topk_ids.shape)}."
        )
    if tuple(topk_weights.shape) != expected_topk_shape:
        raise RuntimeError(
            f"MegaMoE topk_weights must be {expected_topk_shape}, got "
            f"{tuple(topk_weights.shape)}."
        )

    state = _get_state()
    if (
        state.max_tokens_per_rank is not None
        and hidden_states.shape[0] > state.max_tokens_per_rank
    ):
        raise RuntimeError(
            "MegaMoE local rows exceed the registered SymmBuffer capacity: "
            f"{hidden_states.shape[0]} > {state.max_tokens_per_rank}. Increase "
            "SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK and restart the server."
        )

    if envs.SGLANG_NPU_MEGAMOE_VALIDATE_INPUTS.get():
        # Debug only: these checks synchronize the device.
        ids64 = topk_ids.to(torch.int64)
        if bool(((ids64 < 0) | (ids64 >= layer.num_experts)).any().item()):
            raise RuntimeError("MegaMoE topk_ids contain an out-of-range expert id.")
        sorted_ids = torch.sort(ids64, dim=-1).values
        if layer.top_k > 1 and bool(
            (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item()
        ):
            raise RuntimeError(
                "MegaMoE requires distinct expert ids for every token."
            )


def forward_megamoe(
    layer: "FusedMoE",
    hidden_states: torch.Tensor,
    topk_output: "TopKOutput",
) -> torch.Tensor:
    """Run the fused routed-expert path and return LOCAL combined rows."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not TopKOutputChecker.format_is_standard(topk_output):
        raise RuntimeError(
            "Ascend MegaMoE requires StandardTopKOutput; grouped/bypassed "
            "routing is not supported."
        )
    if not getattr(layer, "_npu_megamoe_weights_processed", False):
        raise RuntimeError(
            "MegaMoE weights were not prepared in canonical MXFP8 layout. "
            "Use a ModelSlim W8A8_MXFP8 or online MXFP8 MoE checkpoint."
        )
    if get_parallel().moe_tp_size != 1:
        raise RuntimeError(
            "Ascend MegaMoE currently requires moe_tp_size=1 (pure EP experts)."
        )

    x = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
    # Routing selection stays FP32. Only Combine's routing weights are cast to
    # BF16 because that is the vendor operator's public contract.
    topk_weights = topk_output.topk_weights.to(torch.bfloat16).contiguous()
    sym_buffer = _get_symm_buffer(layer)
    _validate_inputs(layer, x, topk_ids, topk_weights)

    _, mega_moe = _load_vendor_api()
    output, expert_token_nums = mega_moe(
        x,
        topk_ids,
        topk_weights,
        [layer.w13_weight],
        [layer.w2_weight],
        sym_buffer,
        scales=None,
        l1_weights_sf=[layer.w13_weight_scale],
        l2_weights_sf=[layer.w2_weight_scale],
        x_active_mask=None,
    )
    # No host readback on the hot path.
    del expert_token_nums
    return output


def expected_welm_weight_shapes(
    layer: "FusedMoE",
) -> Tuple[tuple, tuple, tuple, tuple]:
    """Return canonical MegaMoE shapes for diagnostics and unit tests."""
    e = layer.num_local_experts
    h = layer.hidden_size
    i = layer.intermediate_size_per_partition
    return (
        (e, 2 * i, h),
        (e, h, i),
        (e, 2 * i, math.ceil(h / 64), 2),
        (e, h, math.ceil(i / 64), 2),
    )
