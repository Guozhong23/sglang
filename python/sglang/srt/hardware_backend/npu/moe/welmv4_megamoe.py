"""BF16 routed+shared MegaMoE for WeLM ordinary-prefill EP shards.

The target runner creates one registered buffer before KV-cache sizing. All
eligible layers share it; decode, mirror consumers and NextN keep their backend.
The custom npu_ops_transformer wheel is imported only during initialization.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_LOCAL_ROWS = 16384
_SHARED_ARGUMENTS = (
    "l1_weights_shared",
    "l2_weights_shared",
    "l1_weights_shared_sf",
    "l2_weights_shared_sf",
)
# Keep buffers alive until explicit graceful/group teardown, not Python GC.
_RUNTIMES: dict[object, WelmPrefillMegaMoE] = {}


@dataclass(frozen=True)
class WelmMegaMoESharedWeights:
    """Full replicated BF16 FFN weights; views retain the original storage."""

    w1: torch.Tensor
    w2: torch.Tensor


def _check_shared_abi(mega_moe):
    try:
        parameters = inspect.signature(mega_moe).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "WeLM MegaMoE shared fusion cannot inspect the installed wrapper; "
            "install the matching shared-expert wheel and CANN operator package"
        ) from exc
    missing = [name for name in _SHARED_ARGUMENTS if name not in parameters]
    if missing:
        raise RuntimeError(
            "WeLM MegaMoE shared fusion requires the shared-expert wrapper; "
            f"missing parameters: {', '.join(missing)}. Install matching "
            "npu_ops_transformer and CANN operator packages."
        )


def _validate_expert_mapping(mlps, metadata, *, ep_size: int, enable_eplb: bool):
    if enable_eplb:
        raise ValueError("WeLM MegaMoE shared fusion does not support dynamic EPLB")
    if metadata is None:
        return
    # Normal WeLM startup also creates this metadata. An identity map is the
    # ordinary contiguous EP layout, not evidence of expert rebalancing.
    mapping = metadata.physical_to_logical_map_cpu
    for mlp in mlps:
        num_experts = mlp.experts.moe_runner_config.num_experts
        if (
            metadata.ep_size != ep_size
            or metadata.num_physical_experts != num_experts
            or metadata.num_logical_experts != num_experts
            or mapping.device.type != "cpu"
            or mapping.ndim != 2
            or not 0 <= mlp.layer_id < mapping.shape[0]
            or not torch.equal(
                mapping[mlp.layer_id], torch.arange(num_experts, dtype=mapping.dtype)
            )
        ):
            raise ValueError(
                f"WeLM MegaMoE shared fusion layer {mlp.layer_id}: "
                "remapped or redundant expert layouts are unsupported"
            )


def _prepare_shared_weights(mlp, *, ep_size: int, ep_rank: int):
    """Validate semantics/layout before allocating a communication context."""
    from sglang.srt.layers.activation import SiluAndMul

    def require(condition, message):
        if not condition:
            raise ValueError(
                f"WeLM MegaMoE shared fusion layer {mlp.layer_id}: {message}"
            )

    experts = mlp.experts
    config = experts.moe_runner_config
    shared = mlp.shared_expert
    require(
        _has_bf16_routed_weights(experts),
        "only BF16 routed weights are supported",
    )
    require(shared is not None, "a shared expert is required")
    require(mlp.shared_expert_gate is None, "shared_expert_gate is unsupported")
    require(
        shared.swiglu_clamp_limit is None or shared.swiglu_clamp_limit <= 0,
        "shared SwiGLU clamp is unsupported",
    )
    require(
        isinstance(shared.act_fn, SiluAndMul),
        "shared activation must be SiluAndMul",
    )
    require(
        config.gemm1_clamp_limit is None or config.gemm1_clamp_limit <= 0,
        "routed SwiGLU clamp is unsupported",
    )
    require(
        config.activation == "silu" and config.is_gated,
        "routed activation must be SwiGLU",
    )
    require(not getattr(experts, "with_bias", False), "routed bias is unsupported")
    require(
        not getattr(experts.quant_method, "load_up_proj_weight_first", False),
        "routed W1 must use gate-then-up order",
    )
    require(
        not config.apply_router_weight_on_input
        and not config.no_combine
        and config.routed_scaling_factor in (None, 1.0)
        and config.gemm1_alpha in (None, 1.0)
        and config.swiglu_limit in (None, 0.0),
        "routed FFN requires ordinary SwiGLU and output routing weights",
    )

    h = config.hidden_size
    ir = config.intermediate_size_per_partition
    require(ep_size >= 2, "EP size must be at least 2")
    require(
        0 < config.num_experts <= 1024 and config.num_experts % ep_size == 0,
        "routed expert count must be divisible by EP size and at most 1024",
    )
    e = config.num_experts // ep_size
    require(0 < config.top_k <= config.num_experts, "invalid routed topK")
    require(
        experts.moe_tp_size == 1
        and experts.moe_ep_size == ep_size
        and experts.moe_ep_rank == ep_rank
        and experts._expert_storage_rank == ep_rank
        and experts._num_local_routed == e
        and experts.num_local_experts == e
        and experts._num_global_routed == config.num_experts
        and not experts.num_fused_shared_experts,
        "routed weights must be continuous EP shards without fused/redundant slots",
    )
    gate_up = shared.gate_up_proj
    down = shared.down_proj
    require(
        gate_up.tp_size == down.tp_size == 1
        and gate_up.tp_rank == down.tp_rank == 0,
        "shared weights must be complete replicas (TP rank 0, size 1)",
    )
    require(gate_up.bias is None and down.bias is None, "shared bias is unsupported")
    require(not down.reduce_results, "shared down projection must not reduce across TP")
    require(down.weight.ndim == 2, "shared down weight must be [H, Is]")
    is_ = down.weight.shape[1]
    require(
        h > 0 and h % 16 == 0 and is_ > 0 and is_ % 16 == 0,
        "BF16 H and shared intermediate size must be positive multiples of 16",
    )
    require(ir > 0, "routed intermediate size must be positive")

    device = experts.w13_weight.device
    for name, weight, shape in (
        ("routed W1", experts.w13_weight, (e, 2 * ir, h)),
        ("routed W2", experts.w2_weight, (e, h, ir)),
        ("shared W1", gate_up.weight, (2 * is_, h)),
        ("shared W2", down.weight, (h, is_)),
    ):
        require(weight.dtype == torch.bfloat16, f"{name} must be BF16")
        require(tuple(weight.shape) == shape, f"{name} must have shape {shape}")
        require(weight.device == device, f"{name} is on a different device")
        require(weight.is_contiguous(), f"{name} must be contiguous ND")
        if weight.device.type in ("npu", "privateuseone"):
            import torch_npu

            require(torch_npu.get_npu_format(weight) == 2, f"{name} must use ND format")
    return WelmMegaMoESharedWeights(
        w1=gate_up.weight.detach().unsqueeze(0),
        w2=down.weight.detach().unsqueeze(0),
    )


def _has_bf16_routed_weights(experts):
    return all(
        getattr(experts, f"{prefix}_weight").dtype == torch.bfloat16
        for prefix in ("w13", "w2")
    )


class WelmPrefillMegaMoE:
    def __init__(
        self,
        group,
        *,
        config,
        device,
        prefill_token_threshold: int,
    ):
        from npu_ops_transformer.ops.mega_moe import (
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )

        self.group = group
        _check_shared_abi(mega_moe)
        from npu_ops_transformer.ops.comm_context import comm_context_op_builder

        # Build/load before warmup or graph capture, on every worker.
        comm_context_op_builder.load()
        self.max_local_rows = _MAX_LOCAL_ROWS
        self.prefill_token_threshold = prefill_token_threshold
        self._mega_moe = mega_moe
        self._closed = False
        # SymmBuffer queries the name with init_comm=False. Initialize on every
        # participating rank now, not lazily on the first non-empty request.
        group._get_backend(device).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=True
        )
        torch.distributed.barrier(group=group)
        torch.npu.synchronize(device)
        self.symm_buffer = get_symm_buffer_for_mega_moe(
            group,
            num_experts=config.num_experts,
            num_max_tokens_per_rank=_MAX_LOCAL_ROWS,
            num_topk=config.top_k,
            hidden=config.hidden_size,
            intermediate_hidden=config.intermediate_size_per_partition,
            max_recv_token_num=0,
            dispatch_quant_mode=0,
            dispatch_quant_out_dtype=28,
            combine_quant_mode=0,
            comm_alg="",
        )

    def meets_prefill_threshold(self, num_tokens: int) -> bool:
        # Match OProj's padded MM rows before attention-TP ReduceScatter, not
        # per-rank MoE rows or the sequence length including cached prefixes.
        return (
            self.prefill_token_threshold <= 0
            or num_tokens > self.prefill_token_threshold
        )

    def can_run(self, num_rows: int) -> bool:
        # DP callers pass the common physical slot, not local real-token count.
        return 0 < num_rows <= self.max_local_rows

    @staticmethod
    def local_valid_rows(shard_rows: int, real_rows: int, attn_tp_rank: int) -> int:
        # Ordinary prefill has a valid prefix followed by padding in each DP
        # slot. ReduceScatter preserves that order inside each attn-TP shard.
        # Use host metadata only; never read a device scalar to skip padding.
        return min(max(real_rows - attn_tp_rank * shard_rows, 0), shard_rows)

    def forward_layer(
        self,
        experts,
        hidden_states,
        topk_output,
        num_valid_rows: int,
        *,
        zero_output_padding: bool = True,
        valid_row_mask: torch.Tensor | None = None,
        shared_weights: WelmMegaMoESharedWeights | None = None,
    ):
        if shared_weights is None:
            raise RuntimeError(
                "WeLM MegaMoE shared fusion requires prepared shared weights"
            )
        # The caller skips DeepEP's -1 masking. TopK IDs are already legal and
        # distinct, including padding rows; only their mixture weights need 0.
        ids = topk_output.topk_ids.to(torch.int32)
        weights = topk_output.topk_weights.to(torch.bfloat16)
        has_padding = num_valid_rows < hidden_states.shape[0]
        if valid_row_mask is not None:
            weights.masked_fill_(~valid_row_mask[:, None], 0)
        elif has_padding:
            # TopK returns FP32: the cast above owns a new BF16 tensor. Clear
            # only its suffix, without another full tensor copy or mask kernel.
            weights[num_valid_rows:].zero_()
        output, _ = self._mega_moe(
            hidden_states,
            ids,
            weights,
            # Opted-in experts retain ND during weight postprocessing,
            # in [E, 2I, H]/[E, H, I] order. No per-forward format conversion
            # or extra weight copy; fallback GMMs share these ND weights.
            [experts.w13_weight],
            [experts.w2_weight],
            self.symm_buffer,
            l1_weights_sf=None,
            l2_weights_sf=None,
            l1_weights_shared=[shared_weights.w1],
            l2_weights_shared=[shared_weights.w2],
            l1_weights_shared_sf=None,
            l2_weights_shared_sf=None,
        )
        if valid_row_mask is not None and zero_output_padding:
            output.masked_fill_(~valid_row_mask[:, None], 0)
        elif has_padding and zero_output_padding:
            output[num_valid_rows:].zero_()
        return output

    def close(self):
        if not self._closed:
            # The wheel drains device work before freeing registered memory.
            # Never call this on a scheduler exception/wedged-device path.
            self.symm_buffer.destroy()
            self._closed = True


def init_welm_prefill_megamoe(model_runner):
    """Bind BF16 routed+shared layers only when the main switch is enabled."""
    if not envs.WELM_NPU_USE_MEGAMOE.get():
        return None
    prefill_token_threshold = envs.WELM_NPU_MEGAMOE_PREFILL_TOKEN_THRESHOLD.get()

    from sglang.srt.distributed import (
        get_moe_ep_group,
        get_moe_ep_normal_comm_group,
    )
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_deepep():
        raise ValueError("WeLM MegaMoE shared fusion requires the DeepEP backend")

    eligible = []
    for layer in model_runner.model.model.layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if (
            experts is None
            or mlp.is_nextn
            or mlp.is_kv_mirror_consumer
            or not mlp.welm_local_ep_kernel_available
        ):
            continue
        if not _has_bf16_routed_weights(experts):
            raise ValueError(
                f"WeLM MegaMoE shared fusion layer {mlp.layer_id}: "
                "only BF16 routed weights are supported"
            )
        eligible.append(mlp)
    if not eligible:
        raise ValueError(
            "WeLM MegaMoE shared fusion found no eligible target prefill layers"
        )

    plan = model_runner.runner_parallel_plan
    if plan is not None:
        coordinator = plan.moe_ep_normal_group or plan.moe_ep_group
    elif (
        envs.SGLANG_DEEPEP_NORMAL_USE_ALLGATHER.get()
        or envs.SGLANG_DEEPEP_NORMAL_USE_ALLTOALL.get()
    ):
        coordinator = get_moe_ep_normal_comm_group()
    else:
        coordinator = get_moe_ep_group()
    group = coordinator.device_group
    if group in _RUNTIMES:
        raise RuntimeError("WeLM MegaMoE context already initialized for this EP group")
    shared_weights = {}
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

    ep_size = torch.distributed.get_world_size(group)
    ep_rank = torch.distributed.get_rank(group)
    _validate_expert_mapping(
        eligible,
        get_global_expert_location_metadata(),
        ep_size=ep_size,
        enable_eplb=model_runner.server_args.enable_eplb,
    )
    config = eligible[0].experts.moe_runner_config
    signature = (
        config.num_experts,
        config.hidden_size,
        config.top_k,
        config.intermediate_size_per_partition,
    )
    for mlp in eligible:
        shared_weights[mlp.layer_id] = _prepare_shared_weights(
            mlp, ep_size=ep_size, ep_rank=ep_rank
        )
        layer_config = mlp.experts.moe_runner_config
        if (
            layer_config.num_experts,
            layer_config.hidden_size,
            layer_config.top_k,
            layer_config.intermediate_size_per_partition,
        ) != signature:
            raise ValueError(
                "WeLM MegaMoE shared fusion requires matching routed "
                "dimensions across layers"
            )
    runtime = WelmPrefillMegaMoE(
        group,
        config=eligible[0].experts.moe_runner_config,
        device=eligible[0].experts.w13_weight.device,
        prefill_token_threshold=prefill_token_threshold,
    )
    _RUNTIMES[group] = runtime
    for mlp in eligible:
        mlp.welm_megamoe_shared_weights = shared_weights[mlp.layer_id]
        mlp.welm_prefill_megamoe = runtime
    logger.info(
        "WeLM MegaMoE BF16 shared fusion initialized: layers=%s, EP=%s, "
        "max_local_rows=%s, prefill_token_threshold=%s, wrapper=%s",
        [mlp.layer_id for mlp in eligible],
        ep_size,
        runtime.max_local_rows,
        runtime.prefill_token_threshold,
        inspect.getsourcefile(runtime._mega_moe),
    )
    return runtime


def close_welm_megamoe_runtimes():
    for runtime in _RUNTIMES.values():
        runtime.close()
