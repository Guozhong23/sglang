"""Debug-only tensor dumps for the Ascend MegaMoE integration."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE


logger = logging.getLogger(__name__)


@torch.no_grad()
def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu()


@torch.no_grad()
def _select_raw_bytes(
    tensor: torch.Tensor,
    selections: Sequence[tuple[int, Sequence[int]]],
) -> torch.Tensor:
    """Index FP8/E8M0 data through a UINT8 view supported by torch_npu."""
    if tensor.element_size() != 1:
        raise RuntimeError(
            f"MegaMoE raw-byte sampling requires one-byte tensors, got {tensor.dtype}."
        )
    selected = tensor.detach().view(torch.uint8)
    for dim, indices in selections:
        index = torch.tensor(indices, dtype=torch.int64, device=tensor.device)
        selected = torch.index_select(selected, dim, index)
    return _cpu(selected)


@torch.no_grad()
def _collect_weight_samples(layer: "FusedMoE") -> dict[str, Any]:
    samples: dict[str, Any] = {}
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
        samples[prefix] = {
            "weight_shape": tuple(weight.shape),
            "weight_dtype": str(weight.dtype),
            "weight_stride": tuple(weight.stride()),
            "weight_contiguous": weight.is_contiguous(),
            "scale_shape": tuple(scale.shape),
            "scale_dtype": str(scale.dtype),
            "scale_stride": tuple(scale.stride()),
            "scale_contiguous": scale.is_contiguous(),
            "expert_indices": expert_indices,
            "row_indices": row_indices,
            "k_indices": k_indices,
            "scale_block_indices": block_indices,
            "weight_raw_u8": _select_raw_bytes(
                weight,
                ((0, expert_indices), (1, row_indices), (2, k_indices)),
            ),
            "scale_raw_u8": _select_raw_bytes(
                scale,
                ((0, expert_indices), (1, row_indices), (2, block_indices)),
            ),
        }
    return samples


@torch.no_grad()
def dump_megamoe_comparison(
    *,
    layer: "FusedMoE",
    call_index: int,
    ep_rank: int,
    row_sizes: Sequence[int],
    x_local: torch.Tensor,
    router_logits_local: Optional[torch.Tensor],
    topk_ids_local: torch.Tensor,
    topk_weights_fp32_local: torch.Tensor,
    topk_weights_bf16_local: torch.Tensor,
    x_global: torch.Tensor,
    topk_ids_global: torch.Tensor,
    topk_weights_global: torch.Tensor,
    actual_output: torch.Tensor,
    reference_output: torch.Tensor,
    expert_token_nums: torch.Tensor,
) -> str:
    """Persist one self-contained, rank-local fused/reference comparison."""
    dump_root = os.path.abspath(
        os.path.expanduser(envs.SGLANG_NPU_MEGAMOE_DUMP_DIR.get())
    )
    rank_dir = os.path.join(dump_root, f"rank_{ep_rank:02d}")
    os.makedirs(rank_dir, exist_ok=True)
    layer_id = int(layer.layer_id)
    final_path = os.path.join(
        rank_dir, f"layer_{layer_id:03d}_call_{call_index:03d}.pt"
    )
    temporary_path = f"{final_path}.tmp.{os.getpid()}"

    payload: dict[str, Any] = {
        "format_version": 1,
        "layer_id": layer_id,
        "call_index": int(call_index),
        "ep_rank": int(ep_rank),
        "row_sizes": [int(value) for value in row_sizes],
        "local_row_offset": int(sum(row_sizes[:ep_rank])),
        "num_experts": int(layer.num_experts),
        "num_local_experts": int(layer.num_local_experts),
        "top_k": int(layer.top_k),
        "hidden_size": int(layer.hidden_size),
        "intermediate_size": int(layer.intermediate_size_per_partition),
        "x_local": _cpu(x_local),
        "router_logits_local": (
            _cpu(router_logits_local)
            if isinstance(router_logits_local, torch.Tensor)
            else None
        ),
        "topk_ids_local": _cpu(topk_ids_local),
        "topk_weights_fp32_local": _cpu(topk_weights_fp32_local),
        "topk_weights_bf16_local": _cpu(topk_weights_bf16_local),
        "x_global": _cpu(x_global),
        "topk_ids_global": _cpu(topk_ids_global),
        "topk_weights_global": _cpu(topk_weights_global),
        "actual_output": _cpu(actual_output),
        "reference_output": _cpu(reference_output),
        "difference_fp32": _cpu(actual_output.float() - reference_output.float()),
        "expert_token_nums": _cpu(expert_token_nums),
    }
    if envs.SGLANG_NPU_MEGAMOE_DUMP_WEIGHT_SAMPLES.get():
        payload["weight_samples"] = _collect_weight_samples(layer)

    torch.save(payload, temporary_path)
    os.replace(temporary_path, final_path)
    logger.info(
        "[MegaMoE-Dump] layer=%d call=%d ep_rank=%d rows=%d path=%s",
        layer_id,
        call_index,
        ep_rank,
        x_local.shape[0],
        final_path,
    )
    return final_path
