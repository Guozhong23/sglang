"""Debug-only paired stage dumps for the WeLM NPU MoE path.

Unlike ``mega_moe_dump``, this module is backend-neutral: the same hooks run
for DeepEP/LocalEP and MegaMoE so two deterministic server runs can be aligned
at the MoE input, router, routed/shared outputs, and final MoE output.

Copying tensors to the host synchronizes the NPU. These dumps must never be
used to diagnose asynchronous stream ordering or to measure performance.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SEEN_CALLS: dict[int, int] = {}
_CAPTURED_CALLS: dict[int, int] = {}
_SAFE_TAG = re.compile(r"[^A-Za-z0-9_.-]+")


def _parse_layers(raw: str) -> Optional[frozenset[int]]:
    normalized = (raw or "").strip().lower()
    if normalized in ("", "*", "all"):
        return None

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


def begin_welm_moe_stage_dump(layer_id: int, is_prefill: bool) -> Optional[int]:
    """Reserve a per-layer capture index, or return ``None`` when filtered."""

    if not envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP.get():
        return None
    if envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_PREFILL_ONLY.get() and not is_prefill:
        return None

    raw_layers = envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_LAYERS.get()
    try:
        selected = _parse_layers(raw_layers)
    except ValueError as exc:
        raise RuntimeError(
            "Invalid SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_LAYERS="
            f"{raw_layers!r}; expected comma-separated ids, inclusive ranges, "
            "'all', or '*'."
        ) from exc
    layer_id = int(layer_id)
    if selected is not None and layer_id not in selected:
        return None

    with _LOCK:
        seen_index = _SEEN_CALLS.get(layer_id, 0)
        _SEEN_CALLS[layer_id] = seen_index + 1
        if seen_index < max(
            0, envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_SKIP_CALLS.get()
        ):
            return None
        call_index = _CAPTURED_CALLS.get(layer_id, 0)
        if call_index >= max(
            0, envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_MAX_CALLS.get()
        ):
            return None
        _CAPTURED_CALLS[layer_id] = call_index + 1
        return call_index


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _safe_tag() -> str:
    raw = (envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_TAG.get() or "run").strip()
    return _SAFE_TAG.sub("_", raw) or "run"


def dump_welm_moe_stages(
    *,
    layer_id: int,
    call_index: int,
    rank: int,
    backend: str,
    is_prefill: bool,
    moe_input: torch.Tensor,
    router_logits: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    routed_output: torch.Tensor,
    shared_output: Optional[torch.Tensor],
    moe_output: torch.Tensor,
    valid_row_mask: Optional[torch.Tensor],
    invalid_row_mask: Optional[torch.Tensor],
    num_token_non_padded: Any,
) -> str:
    """Persist one rank-local, backend-neutral MoE stage payload."""

    dump_root = os.path.abspath(
        os.path.expanduser(envs.SGLANG_NPU_WELMV4_MOE_STAGE_DUMP_DIR.get())
    )
    tag = _safe_tag()
    rank_dir = os.path.join(dump_root, tag, f"rank_{int(rank):02d}")
    os.makedirs(rank_dir, exist_ok=True)
    final_path = os.path.join(
        rank_dir,
        f"layer_{int(layer_id):03d}_call_{int(call_index):03d}.pt",
    )
    temporary_path = f"{final_path}.tmp.{os.getpid()}"

    payload: dict[str, Any] = {
        "format_version": 1,
        "tag": tag,
        "backend": str(backend),
        "layer_id": int(layer_id),
        "call_index": int(call_index),
        "rank": int(rank),
        "is_prefill": bool(is_prefill),
        "moe_input": _cpu(moe_input),
        "router_logits": _cpu(router_logits),
        "topk_ids": _cpu(topk_ids),
        "topk_weights": _cpu(topk_weights),
        "routed_output": _cpu(routed_output),
        "shared_output": _cpu(shared_output),
        "moe_output": _cpu(moe_output),
        "valid_row_mask": _cpu(valid_row_mask),
        "invalid_row_mask": _cpu(invalid_row_mask),
        "num_token_non_padded": _cpu(num_token_non_padded),
    }
    torch.save(payload, temporary_path)
    os.replace(temporary_path, final_path)
    logger.info(
        "[WeLM-MoE-StageDump] tag=%s backend=%s layer=%d call=%d rank=%d "
        "rows=%d path=%s",
        tag,
        backend,
        layer_id,
        call_index,
        rank,
        moe_input.shape[0],
        final_path,
    )
    return final_path

