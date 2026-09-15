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
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

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
            debug_calls={},
            debug_seen_calls={},
            debug_weight_layers=set(),
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


def _debug_layer_selected(layer_id: int) -> bool:
    raw = (envs.SGLANG_NPU_MEGAMOE_DEBUG_LAYERS.get() or "").strip().lower()
    if raw in ("*", "all"):
        return True
    try:
        return layer_id in {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError:
        logger.error(
            "[MegaMoE-Debug] Invalid SGLANG_NPU_MEGAMOE_DEBUG_LAYERS=%r; "
            "expected comma-separated integers, 'all', or '*'.",
            raw,
        )
        return False


def _begin_debug_call(layer: "FusedMoE") -> Optional[int]:
    if not (
        envs.SGLANG_NPU_MEGAMOE_DEBUG.get()
        or envs.SGLANG_NPU_MEGAMOE_SHADOW_COMPARE.get()
        or envs.SGLANG_NPU_MEGAMOE_DUMP.get()
    ):
        return None
    layer_id = int(layer.layer_id)
    if not _debug_layer_selected(layer_id):
        return None
    state = _get_state()
    seen_index = int(state.debug_seen_calls.get(layer_id, 0))
    state.debug_seen_calls[layer_id] = seen_index + 1
    if seen_index < max(0, envs.SGLANG_NPU_MEGAMOE_DEBUG_SKIP_CALLS.get()):
        return None
    call_index = int(state.debug_calls.get(layer_id, 0))
    if call_index >= max(0, envs.SGLANG_NPU_MEGAMOE_DEBUG_MAX_CALLS.get()):
        return None
    state.debug_calls[layer_id] = call_index + 1
    return call_index


@torch.no_grad()
def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Return compact host-side statistics for a debug tensor."""
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    finite_count = finite.sum()
    safe = torch.where(finite, value, torch.zeros_like(value))
    count = finite_count.clamp_min(1).to(value.dtype)
    minimum = torch.where(finite, value, torch.full_like(value, float("inf"))).min()
    maximum = torch.where(finite, value, torch.full_like(value, float("-inf"))).max()
    packed = torch.stack(
        (
            finite_count.to(value.dtype),
            minimum,
            maximum,
            safe.sum() / count,
            torch.sqrt(torch.square(safe).sum() / count),
        )
    ).cpu()
    finite_n, minimum_v, maximum_v, mean_v, rms_v = packed.tolist()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": f"{int(finite_n)}/{tensor.numel()}",
        "min": minimum_v,
        "max": maximum_v,
        "mean": mean_v,
        "rms": rms_v,
    }


def _all_gather_variable_rows(
    tensor: torch.Tensor, ep_group: Any, row_sizes: Sequence[int]
) -> torch.Tensor:
    """All-gather dim 0 even when EP ranks own different row counts."""
    max_rows = max(row_sizes)
    if tensor.shape[0] < max_rows:
        padding = torch.zeros(
            (max_rows - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding), dim=0)
    gathered = ep_group.all_gather(tensor.contiguous(), dim=0)
    chunks = gathered.reshape(len(row_sizes), max_rows, *tensor.shape[1:])
    return torch.cat(
        tuple(chunks[rank, :rows] for rank, rows in enumerate(row_sizes)), dim=0
    )


@torch.no_grad()
def _collect_debug_inputs(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[Any, list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    ep_group = get_moe_ep_group()
    local_rows = torch.tensor([x.shape[0]], dtype=torch.int32, device=x.device)
    row_sizes = [
        int(value)
        for value in ep_group.all_gather(local_rows, dim=0).cpu().tolist()
    ]
    return (
        ep_group,
        row_sizes,
        _all_gather_variable_rows(x, ep_group, row_sizes),
        _all_gather_variable_rows(topk_ids, ep_group, row_sizes),
        _all_gather_variable_rows(topk_weights, ep_group, row_sizes),
    )


@torch.no_grad()
def _select_raw_byte_sample(
    tensor: torch.Tensor,
    selections: Sequence[tuple[int, Sequence[int]]],
) -> torch.Tensor:
    """Index a one-byte FP8 tensor through its supported UINT8 view."""
    if tensor.element_size() != 1:
        raise RuntimeError(
            f"MegaMoE raw-byte sampling requires one-byte tensors, got {tensor.dtype}."
        )
    selected = tensor.detach().view(torch.uint8)
    for dim, indices in selections:
        index = torch.tensor(indices, dtype=torch.int64, device=tensor.device)
        selected = torch.index_select(selected, dim, index)
    return selected.contiguous().cpu()


@torch.no_grad()
def _log_weight_samples(layer: "FusedMoE", ep_rank: int) -> None:
    """Log checkpoint bytes around gate/up and MX block boundaries."""
    state = _get_state()
    layer_id = int(layer.layer_id)
    if layer_id in state.debug_weight_layers:
        return
    state.debug_weight_layers.add(layer_id)

    for prefix in ("w13", "w2"):
        weight = getattr(layer, f"{prefix}_weight").detach()
        scale = getattr(layer, f"{prefix}_weight_scale").detach()
        expert_indices = sorted({0, weight.shape[0] - 1})
        if prefix == "w13":
            split = layer.intermediate_size_per_partition
            row_indices = sorted({0, split - 1, split, weight.shape[1] - 1})
        else:
            row_indices = sorted({0, weight.shape[1] - 1})
        k_indices = sorted(
            {
                0,
                min(31, weight.shape[2] - 1),
                min(32, weight.shape[2] - 1),
                min(63, weight.shape[2] - 1),
                weight.shape[2] - 1,
            }
        )
        block_indices = sorted({0, scale.shape[2] - 1})
        weight_sample = _select_raw_byte_sample(
            weight,
            ((0, expert_indices), (1, row_indices), (2, k_indices)),
        )
        scale_bytes = _select_raw_byte_sample(
            scale,
            ((0, expert_indices), (1, row_indices), (2, block_indices)),
        )
        try:
            scale_values = scale_bytes.view(scale.dtype).float().tolist()
        except Exception:
            scale_values = "E8M0 float conversion unavailable"
        logger.info(
            "[MegaMoE-Debug] layer=%d ep_rank=%d %s real-weight sample "
            "weight(shape=%s,dtype=%s,stride=%s,contiguous=%s) "
            "scale(shape=%s,dtype=%s,stride=%s,contiguous=%s) "
            "experts=%s rows=%s k=%s weight_raw_u8=%s scale_blocks=%s "
            "scale_raw_u8=%s scale_values=%s",
            layer_id,
            ep_rank,
            prefix,
            tuple(weight.shape),
            weight.dtype,
            tuple(weight.stride()),
            weight.is_contiguous(),
            tuple(scale.shape),
            scale.dtype,
            tuple(scale.stride()),
            scale.is_contiguous(),
            expert_indices,
            row_indices,
            k_indices,
            weight_sample.tolist(),
            block_indices,
            scale_bytes.tolist(),
            scale_values,
        )


@torch.no_grad()
def _log_expert_counts(
    layer: "FusedMoE",
    ep_rank: int,
    global_topk_ids: torch.Tensor,
    expert_token_nums: torch.Tensor,
) -> None:
    global_counts = torch.bincount(
        global_topk_ids.detach().to(torch.int64).cpu().reshape(-1),
        minlength=layer.num_experts,
    )
    local_experts = layer.num_local_experts
    first_expert = ep_rank * local_experts
    expected = global_counts[first_expert : first_expert + local_experts]
    actual = expert_token_nums.detach().to(torch.int64).cpu().reshape(-1)
    if actual.numel() != expected.numel():
        logger.error(
            "[MegaMoE-Debug] layer=%d ep_rank=%d expert-count shape mismatch: "
            "operator=%s expected=%s",
            layer.layer_id,
            ep_rank,
            tuple(actual.shape),
            tuple(expected.shape),
        )
        return
    delta = actual - expected
    mismatch = torch.nonzero(delta, as_tuple=False).reshape(-1)
    busiest = torch.argsort(expected, descending=True)[:8]
    logger.info(
        "[MegaMoE-Debug] layer=%d ep_rank=%d expert-counts "
        "global_routes=%d expected_local=%d operator_local=%d mismatches=%d "
        "max_abs_delta=%d busiest(global_id,expected,actual)=%s "
        "first_mismatches(local_id,expected,actual,delta)=%s",
        layer.layer_id,
        ep_rank,
        int(global_counts.sum()),
        int(expected.sum()),
        int(actual.sum()),
        mismatch.numel(),
        int(delta.abs().max()) if delta.numel() else 0,
        [
            (
                first_expert + int(index),
                int(expected[index]),
                int(actual[index]),
            )
            for index in busiest
        ],
        [
            (
                int(index),
                int(expected[index]),
                int(actual[index]),
                int(delta[index]),
            )
            for index in mismatch[:16]
        ],
    )


@torch.no_grad()
def _log_shadow_comparison(
    layer: "FusedMoE",
    output: torch.Tensor,
    ep_group: Any,
    row_sizes: Sequence[int],
    global_x: torch.Tensor,
    global_topk_ids: torch.Tensor,
    global_topk_weights: torch.Tensor,
    ep_rank: int,
    log_comparison: bool = True,
    allow_large_reference: bool = False,
) -> Optional[torch.Tensor]:
    global_rows = sum(row_sizes)
    limit = max(0, envs.SGLANG_NPU_MEGAMOE_SHADOW_MAX_GLOBAL_ROWS.get())
    if global_rows > limit and not allow_large_reference:
        logger.warning(
            "[MegaMoE-Debug] layer=%d shadow comparison skipped: global_rows=%d "
            "> SGLANG_NPU_MEGAMOE_SHADOW_MAX_GLOBAL_ROWS=%d",
            layer.layer_id,
            global_rows,
            limit,
        )
        return None

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    # AscendLocalEPDispatcher consumes only ids and weights; avoid gathering
    # the unnecessary and much larger [M, num_experts] router-logit tensor.
    reference_topk = StandardTopKOutput(
        topk_weights=global_topk_weights,
        topk_ids=global_topk_ids,
        router_logits=torch.empty(0, dtype=torch.float32, device=global_x.device),
    )
    reference_partial = layer.forward_local_ep_partial(global_x, reference_topk)
    reference_global = ep_group.all_reduce(reference_partial)
    offset = sum(row_sizes[:ep_rank])
    reference = reference_global[offset : offset + row_sizes[ep_rank]]

    if not log_comparison:
        return reference

    actual_f = output.detach().float()
    reference_f = reference.detach().float()
    diff = actual_f - reference_f
    eps = torch.finfo(torch.float32).eps
    actual_norm = torch.square(actual_f).sum().sqrt()
    reference_norm = torch.square(reference_f).sum().sqrt()
    packed = torch.stack(
        (
            torch.isfinite(actual_f).sum().float(),
            torch.isfinite(reference_f).sum().float(),
            torch.isfinite(diff).sum().float(),
            torch.square(actual_f).mean().sqrt(),
            torch.square(reference_f).mean().sqrt(),
            diff.abs().max(),
            diff.abs().mean(),
            torch.square(diff).mean().sqrt(),
            torch.square(diff).mean().sqrt()
            / (torch.square(reference_f).mean().sqrt() + eps),
            (actual_f.reshape(-1) * reference_f.reshape(-1)).sum()
            / (actual_norm * reference_norm + eps),
        )
    ).cpu()
    (
        actual_finite,
        reference_finite,
        diff_finite,
        actual_rms,
        reference_rms,
        max_abs,
        mean_abs,
        rmse,
        nrmse,
        cosine,
    ) = packed.tolist()
    sample_width = min(16, output.shape[-1])
    logger.info(
        "[MegaMoE-Shadow] layer=%d ep_rank=%d local_rows=%d global_rows=%d "
        "actual_finite=%d/%d reference_finite=%d/%d diff_finite=%d/%d "
        "actual_rms=%.8g reference_rms=%.8g rms_ratio=%.8g "
        "max_abs=%.8g mean_abs=%.8g rmse=%.8g nrmse=%.8g cosine=%.10f "
        "actual_row0=%s reference_row0=%s",
        layer.layer_id,
        ep_rank,
        output.shape[0],
        global_rows,
        int(actual_finite),
        output.numel(),
        int(reference_finite),
        reference.numel(),
        int(diff_finite),
        diff.numel(),
        actual_rms,
        reference_rms,
        actual_rms / (reference_rms + eps),
        max_abs,
        mean_abs,
        rmse,
        nrmse,
        cosine,
        actual_f[0, :sample_width].cpu().tolist() if output.shape[0] else [],
        reference_f[0, :sample_width].cpu().tolist() if reference.shape[0] else [],
    )
    return reference


def forward_megamoe(
    layer: "FusedMoE",
    hidden_states: torch.Tensor,
    topk_output: "TopKOutput",
) -> torch.Tensor:
    """Run the fused routed-expert path and return LOCAL combined rows."""
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if not getattr(layer, "_npu_megamoe_prefill_enabled", True):
        end_layer = getattr(layer, "_npu_megamoe_prefill_end_layer", "unknown")
        raise RuntimeError(
            "Ascend MegaMoE was invoked outside the WeLM token-sharded "
            f"prefill range: layer={layer.layer_id}, end_layer={end_layer}."
        )
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
    reference_topk_weights = topk_output.topk_weights.contiguous()
    # Routing selection stays FP32. Only Combine's routing weights are cast to
    # BF16 because that is the vendor operator's public contract.
    topk_weights = reference_topk_weights.to(torch.bfloat16).contiguous()
    sym_buffer = _get_symm_buffer(layer)
    _validate_inputs(layer, x, topk_ids, topk_weights)

    debug_call = _begin_debug_call(layer)
    use_shadow_reference = (
        envs.SGLANG_NPU_MEGAMOE_SHADOW_USE_REFERENCE.get()
    )
    debug_inputs = None
    ep_rank = -1
    if use_shadow_reference and debug_call is None:
        debug_inputs = _collect_debug_inputs(x, topk_ids, reference_topk_weights)
        ep_group, _, _, _, _ = debug_inputs
        ep_rank = torch.distributed.get_rank(ep_group.device_group)
    if debug_call is not None:
        debug_inputs = _collect_debug_inputs(x, topk_ids, reference_topk_weights)
        ep_group, row_sizes, _, global_topk_ids, _ = debug_inputs
        ep_rank = torch.distributed.get_rank(ep_group.device_group)
        route_cast_error = (
            reference_topk_weights.float() - topk_weights.float()
        ).abs()
        logger.info(
            "[MegaMoE-Debug] BEGIN layer=%d call=%d ep_rank=%d row_sizes=%s "
            "x=%s route_weights_fp32=%s route_weights_bf16=%s "
            "route_cast_max_abs=%.8g local_ids_row0=%s local_weights_row0=%s "
            "global_id_range=[%d,%d]",
            layer.layer_id,
            debug_call,
            ep_rank,
            row_sizes,
            _tensor_stats(x),
            _tensor_stats(reference_topk_weights),
            _tensor_stats(topk_weights),
            float(route_cast_error.max().cpu()),
            topk_ids[0].cpu().tolist() if topk_ids.shape[0] else [],
            reference_topk_weights[0].float().cpu().tolist()
            if reference_topk_weights.shape[0]
            else [],
            int(global_topk_ids.min().cpu()),
            int(global_topk_ids.max().cpu()),
        )
        try:
            _log_weight_samples(layer, ep_rank)
        except Exception:
            logger.exception(
                "[MegaMoE-Debug] layer=%d ep_rank=%d failed to log weight "
                "samples; continuing the numerical comparison.",
                layer.layer_id,
                ep_rank,
            )

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
    if debug_inputs is not None:
        (
            ep_group,
            row_sizes,
            global_x,
            global_topk_ids,
            global_topk_weights,
        ) = debug_inputs
        should_log = debug_call is not None and (
            envs.SGLANG_NPU_MEGAMOE_DEBUG.get()
            or envs.SGLANG_NPU_MEGAMOE_SHADOW_COMPARE.get()
        )
        if should_log:
            logger.info(
                "[MegaMoE-Debug] layer=%d call=%d ep_rank=%d output=%s "
                "output_row0=%s expert_token_nums_shape=%s",
                layer.layer_id,
                debug_call,
                ep_rank,
                _tensor_stats(output),
                output[0, :16].float().cpu().tolist() if output.shape[0] else [],
                tuple(expert_token_nums.shape),
            )
            _log_expert_counts(layer, ep_rank, global_topk_ids, expert_token_nums)
        needs_selected_reference = debug_call is not None and (
            envs.SGLANG_NPU_MEGAMOE_SHADOW_COMPARE.get()
            or envs.SGLANG_NPU_MEGAMOE_DUMP.get()
        )
        reference_output: Optional[torch.Tensor] = None
        if use_shadow_reference or needs_selected_reference:
            reference_output = _log_shadow_comparison(
                layer,
                output,
                ep_group,
                row_sizes,
                global_x,
                global_topk_ids,
                global_topk_weights,
                ep_rank,
                log_comparison=(
                    debug_call is not None
                    and envs.SGLANG_NPU_MEGAMOE_SHADOW_COMPARE.get()
                ),
                allow_large_reference=use_shadow_reference,
            )
        if use_shadow_reference and reference_output is None:
            raise RuntimeError(
                "MegaMoE reference-output mode could not produce a reference "
                f"for layer {layer.layer_id}."
            )
        if (
            debug_call is not None
            and reference_output is not None
            and envs.SGLANG_NPU_MEGAMOE_DUMP.get()
        ):
            from sglang.srt.hardware_backend.npu.moe.mega_moe_dump import (
                dump_megamoe_comparison,
            )

            dump_megamoe_comparison(
                layer=layer,
                call_index=debug_call,
                ep_rank=ep_rank,
                row_sizes=row_sizes,
                x_local=x,
                router_logits_local=getattr(topk_output, "router_logits", None),
                topk_ids_local=topk_ids,
                topk_weights_fp32_local=reference_topk_weights,
                topk_weights_bf16_local=topk_weights,
                x_global=global_x,
                topk_ids_global=global_topk_ids,
                topk_weights_global=global_topk_weights,
                actual_output=output,
                reference_output=reference_output,
                expert_token_nums=expert_token_nums,
            )
        if use_shadow_reference:
            assert reference_output is not None
            output = reference_output
        if should_log:
            logger.info(
                "[MegaMoE-Debug] END layer=%d call=%d ep_rank=%d",
                layer.layer_id,
                debug_call,
                ep_rank,
            )
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
