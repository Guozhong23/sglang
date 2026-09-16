"""BF16/MXFP8 MegaMoE for WeLM ordinary-prefill EP shards.

The target runner creates one registered buffer before KV-cache sizing. All
eligible layers share it; decode, mirror consumers and NextN keep their backend.
The custom npu_ops_transformer wheel is imported only during initialization.
Expert weights stay in physical ND format for both modes. MXFP8 additionally
passes the checkpoint's E8M0 block scales to the fused operator.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

_MAX_LOCAL_ROWS = 16384
_MXFP8_E4M3_TYPE = 24
_NPU_FORMAT_ND = 2
_MODE_BF16 = "bf16"
_MODE_MXFP8 = "mxfp8"
logger = logging.getLogger(__name__)
# Keep buffers alive until explicit graceful/group teardown, not Python GC.
_RUNTIMES: dict[object, WelmPrefillMegaMoE] = {}


def _parse_layer_selection(raw: str) -> Optional[frozenset[int]]:
    normalized = (raw or "").strip().lower()
    if normalized in ("", "*", "all"):
        return None
    if normalized in ("none", "off"):
        return frozenset()

    selected: set[int] = set()
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            layer_id = int(item)
            if layer_id < 0:
                raise ValueError("layer ids must be non-negative")
            selected.add(layer_id)
            continue
        bounds = item.split("-")
        if len(bounds) != 2 or not bounds[0] or not bounds[1]:
            raise ValueError(f"invalid inclusive layer range {item!r}")
        start, end = (int(value) for value in bounds)
        if start < 0 or end < start:
            raise ValueError(f"invalid inclusive layer range {item!r}")
        selected.update(range(start, end + 1))
    return frozenset(selected)


def _weight_mode(experts) -> Optional[str]:
    if (
        experts.w13_weight.dtype == torch.bfloat16
        and experts.w2_weight.dtype == torch.bfloat16
    ):
        return _MODE_BF16
    if (
        experts.w13_weight.dtype == torch.float8_e4m3fn
        and experts.w2_weight.dtype == torch.float8_e4m3fn
        and getattr(experts, "_npu_megamoe_weights_processed", False)
        and getattr(experts, "w13_weight_scale", None) is not None
        and getattr(experts, "w2_weight_scale", None) is not None
    ):
        return _MODE_MXFP8
    return None


def _validate_nd_tensors(experts, mode: str) -> None:
    """Fail before warmup if the ND-only custom operator would receive NZ."""

    import torch_npu

    get_npu_format = getattr(torch_npu, "get_npu_format", None)
    if get_npu_format is None:
        raise RuntimeError(
            "WeLM MegaMoE requires torch_npu.get_npu_format to verify its "
            "ND-only weight contract. Upgrade to the matching torch_npu build."
        )
    names = ["w13_weight", "w2_weight"]
    if mode == _MODE_MXFP8:
        names.extend(("w13_weight_scale", "w2_weight_scale"))
    for name in names:
        tensor = getattr(experts, name)
        tensor_format = int(get_npu_format(tensor))
        if tensor_format != _NPU_FORMAT_ND:
            raise RuntimeError(
                "WeLM MegaMoE supports only physical ND tensors, but "
                f"{name} has NPU format {tensor_format}; expected "
                f"ACL_FORMAT_ND={_NPU_FORMAT_ND}."
            )


class WelmPrefillMegaMoE:
    def __init__(self, group, *, config, device, weight_mode: str):
        from npu_ops_transformer.ops.mega_moe import (
            get_symm_buffer_for_mega_moe,
            mega_moe,
        )

        self.group = group
        configured_rows = envs.SGLANG_NPU_MEGAMOE_MAX_TOKENS_PER_RANK.get()
        self.max_local_rows = (
            configured_rows if configured_rows > 0 else _MAX_LOCAL_ROWS
        )
        self.weight_mode = weight_mode
        self._mega_moe = mega_moe
        self._closed = False
        # SymmBuffer queries the name with init_comm=False. Initialize on every
        # participating rank now, not lazily on the first non-empty request.
        group._get_backend(device).get_hccl_comm_name(
            torch.distributed.get_rank(group), init_comm=True
        )
        buffer_args = dict(
            num_experts=config.num_experts,
            num_max_tokens_per_rank=self.max_local_rows,
            num_topk=config.top_k,
            hidden=config.hidden_size,
            max_recv_token_num=0,
            combine_quant_mode=0,
            comm_alg="",
        )
        if weight_mode == _MODE_MXFP8:
            buffer_args.update(
                intermediate_hidden=0,
                dispatch_quant_mode=4,
                dispatch_quant_out_dtype=_MXFP8_E4M3_TYPE,
            )
        else:
            buffer_args.update(
                intermediate_hidden=config.intermediate_size_per_partition,
                dispatch_quant_mode=0,
            )
        self.symm_buffer = get_symm_buffer_for_mega_moe(
            group,
            **buffer_args,
        )
        logger.info(
            "Initialized WeLM prefill MegaMoE sidecar: mode=%s max_local_rows=%d",
            weight_mode,
            self.max_local_rows,
        )

    def can_run(self, num_rows: int, layer_id: Optional[int] = None) -> bool:
        # DP callers pass the common physical slot, not local real-token count.
        if not (0 < num_rows <= self.max_local_rows):
            return False
        if layer_id is None:
            return True
        raw = envs.SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS.get()
        try:
            selected = _parse_layer_selection(raw)
        except ValueError as exc:
            raise RuntimeError(
                "Invalid SGLANG_NPU_MEGAMOE_ACTUAL_LAYERS="
                f"{raw!r}; expected 'all', 'none', comma-separated ids, or "
                "inclusive ranges such as '0-7,16'."
            ) from exc
        return selected is None or int(layer_id) in selected

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
    ):
        # The caller skips DeepEP's -1 masking. TopK IDs are already legal and
        # distinct, including padding rows; only their mixture weights need 0.
        ids = topk_output.topk_ids.to(torch.int32)
        weights = topk_output.topk_weights.to(torch.bfloat16)
        has_padding = num_valid_rows < hidden_states.shape[0]
        if has_padding:
            # TopK returns FP32: the cast above owns a new BF16 tensor. Clear
            # only its suffix, without another full tensor copy or mask kernel.
            weights[num_valid_rows:].zero_()
        operator_args = {}
        if self.weight_mode == _MODE_MXFP8:
            if not getattr(experts, "_npu_megamoe_weights_processed", False):
                raise RuntimeError(
                    "WeLM MXFP8 MegaMoE weights were not prepared in ND layout."
                )
            operator_args.update(
                scales=None,
                l1_weights_sf=[experts.w13_weight_scale],
                l2_weights_sf=[experts.w2_weight_scale],
                x_active_mask=None,
            )
        output, _ = self._mega_moe(
            hidden_states,
            ids,
            weights,
            # Opted-in BF16/MXFP8 experts retain ND during postprocessing,
            # in [E, 2I, H]/[E, H, I] order. No per-forward format conversion
            # or extra weight copy; fallback GMMs share these ND weights.
            [experts.w13_weight],
            [experts.w2_weight],
            self.symm_buffer,
            **operator_args,
        )
        if envs.SGLANG_NPU_MEGAMOE_SYNC_AFTER_OP.get():
            # Debug only: distinguish a missing vendor-stream completion event
            # from numerical drift. Never enable for performance measurement.
            torch.npu.synchronize()
        if has_padding and zero_output_padding:
            output[num_valid_rows:].zero_()
        return output

    def close(self):
        if not self._closed:
            # The wheel drains device work before freeing registered memory.
            # Never call this on a scheduler exception/wedged-device path.
            self.symm_buffer.destroy()
            self._closed = True


def init_welm_prefill_megamoe(model_runner):
    """Bind loaded BF16/MXFP8 target layers without modifying their weights."""
    from sglang.srt.distributed import (
        get_moe_ep_group,
        get_moe_ep_normal_comm_group,
    )
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if not get_moe_a2a_backend().is_deepep():
        return None

    eligible = []
    eligible_modes = set()
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
        config = experts.moe_runner_config
        if config.gemm1_clamp_limit is not None and config.gemm1_clamp_limit > 0:
            continue  # This custom ABI has no routed SwiGLU clamp argument.
        mode = _weight_mode(experts)
        if mode is None:
            continue
        _validate_nd_tensors(experts, mode)
        eligible.append(mlp)
        eligible_modes.add(mode)
    if not eligible:
        return None
    if len(eligible_modes) != 1:
        raise RuntimeError(
            "One WeLM MegaMoE HCCL group cannot mix BF16 and MXFP8 expert "
            f"layers, got modes={sorted(eligible_modes)}."
        )
    weight_mode = next(iter(eligible_modes))

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
    runtime = WelmPrefillMegaMoE(
        group,
        config=eligible[0].experts.moe_runner_config,
        device=eligible[0].experts.w13_weight.device,
        weight_mode=weight_mode,
    )
    _RUNTIMES[group] = runtime
    for mlp in eligible:
        mlp.welm_prefill_megamoe = runtime
    return runtime


def close_welm_megamoe_runtimes():
    for runtime in _RUNTIMES.values():
        runtime.close()
