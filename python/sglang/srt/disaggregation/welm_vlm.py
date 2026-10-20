# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""WeLM VL PD contracts. No vision embeddings or OE tables cross this boundary."""

import hashlib
import json
import logging
import struct
import sys
from array import array

WELM_VLM_ARCHITECTURES = (
    "WeLMV4VLMForConditionalGeneration",
    "WeLMMoonViTVLForConditionalGeneration",
)

# Reuse the existing 16-int32 metadata row: 0:7 are cache/usage counts.
# 7:16 = version marker, expanded prompt length, seven SHA256 words.
PROMPT_SIGNATURE_SLICE = slice(7, 16)
PROMPT_SIGNATURE_VERSION = 0x574C0001


def is_welm_vlm(config):
    return bool(
        set(getattr(config, "architectures", None) or []) & set(WELM_VLM_ARCHITECTURES)
    )


def _semantic_config(value):
    """Ignore local checkpoint locations and HF bookkeeping, not model semantics."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {
            key: _semantic_config(item)
            for key, item in value.items()
            if not key.startswith("_")
            and key not in ("transformers_version", "torch_dtype", "dtype")
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_config(item) for item in value]
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_welm_vlm_pd_contract(scheduler, kv_args):
    """Describe actual buffers after allocation; capacities and pointers may differ.

    P and D may use different vision parallelism or MoE kernels. Their text
    configuration, KV layer ordering, per-page byte strides and OE semantics
    must agree before D advertises destination addresses.
    """
    config = scheduler.model_config.hf_config
    if not is_welm_vlm(config):
        return None
    if str(scheduler.model_config.dtype) != "torch.bfloat16":
        raise ValueError("WeLM-VL PD requires resolved BF16 model weights")
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    args = scheduler.server_args
    mapping = getattr(pool, "layers_mapping", None)
    if (
        mapping is None
        or getattr(pool, "_unified_kv", False)
        or hasattr(pool, "unified_buffer")
    ):
        raise ValueError("WeLM-VL PD requires the full/SWA paged KV pool")
    if [str(getattr(st, "value", st)) for st in kv_args.state_types] != ["swa"]:
        raise ValueError("WeLM-VL PD requires exactly one SWA page component")

    def layer_order(swa):
        return [
            layer
            for _, layer in sorted(
                (index, layer)
                for layer, (index, is_swa) in mapping.items()
                if is_swa == swa
            )
        ]

    full, swa = layer_order(False), layer_order(True)
    if len(kv_args.kv_item_lens) != 2 * len(full) or len(
        kv_args.state_item_lens[0]
    ) != 2 * len(swa):
        raise ValueError("WeLM-VL PD KV buffers do not match full/SWA layer order")
    # An item is a whole page of K or V, never a visual embedding or a ring row.
    for sub_pool, pointers, strides in (
        (pool.full_kv_pool, kv_args.kv_data_ptrs, kv_args.kv_item_lens),
        (pool.swa_kv_pool, kv_args.state_data_ptrs[0], kv_args.state_item_lens[0]),
    ):
        buffers = [
            (getter(i), dim)
            for getter, dim in (
                (sub_pool.get_key_buffer, sub_pool.head_dim),
                (sub_pool.get_value_buffer, sub_pool.v_head_dim),
            )
            for i in range(
                sub_pool.start_layer, sub_pool.start_layer + sub_pool.layer_num
            )
        ]
        for buf, dim in buffers:
            if (
                str(buf.dtype) != "torch.bfloat16"
                or not buf.is_contiguous()
                or buf.ndim != 4
                or buf.shape[1] not in (1, pool.page_size)
                or tuple(buf.shape[2:]) != (sub_pool.head_num, dim)
            ):
                raise ValueError(
                    "WeLM-VL PD requires contiguous BF16 token-major KV buffers"
                )
        if list(pointers) != [buf.data_ptr() for buf, _ in buffers]:
            raise ValueError("WeLM-VL PD KV addresses must follow K then V layer order")
        expected = [
            buf.element_size() * pool.page_size * sub_pool.head_num * dim
            for buf, dim in buffers
        ]
        if list(strides) != expected:
            raise ValueError("WeLM-VL PD requires contiguous token-major KV pages")
    contract = dict(
        version=1,
        architecture=list(config.architectures),
        config_digest=_digest(_semantic_config(config)),
        context_len=scheduler.model_config.context_len,
        attn_tp_size=scheduler.ps.attn_tp_size,
        kv_dtype=kv_args.kv_cache_dtype_str,
        page_size=kv_args.page_size,
        kv_mirror=args.enable_kv_mirror,
        over_encoding=args.enable_over_encoding,
        full_layers=full,
        swa_layers=swa,
        full_page_bytes=list(kv_args.kv_item_lens),
        swa_page_bytes=list(kv_args.state_item_lens[0]),
        aux_row_bytes=list(kv_args.aux_item_lens),
        heads=[
            [sub.head_num, sub.head_dim, sub.v_head_dim]
            for sub in (pool.full_kv_pool, pool.swa_kv_pool)
        ],
    )
    logging.getLogger(__name__).info(
        "WeLM-VL PD contract %s", json.dumps(contract, sort_keys=True)
    )
    return contract


def validate_welm_vlm_pd_contract(prefill, decode):
    if prefill is None and decode is None:
        return
    if prefill is None or decode is None:
        raise ValueError(
            "WeLM-VL PD requires matching contract support on P and D; "
            "update both workers to the same version and model"
        )
    if prefill != decode:
        differing = sorted(
            key
            for key in prefill.keys() | decode.keys()
            if prefill.get(key) != decode.get(key)
        )
        raise ValueError(f"WeLM-VL PD contract mismatch: {differing}")


def welm_vlm_prompt_signature(req):
    """Hash expanded cache IDs and ordered image metadata, independently of cache hits.

    No feature/device transfer or mutation. The processor's content hash and
    offsets distinguish duplicate images, order, grid and expansion lengths.
    Keep the original hash-pad IDs here; only the OE history uses logical IDs.
    """
    tokens = array("q", req.origin_input_ids)
    if sys.byteorder != "little":
        tokens.byteswap()
    digest = hashlib.sha256(tokens.tobytes())
    images = []
    mm = req.multimodal_inputs
    for item in mm.mm_items if mm is not None else ():
        grid = item.image_grid_thw
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        images.append(
            dict(
                modality=str(getattr(item.modality, "name", item.modality)),
                hash=int(item.hash),
                pad=int(item.pad_value),
                grid=grid,
                offsets=item.offsets,
            )
        )
    digest.update(json.dumps(images, sort_keys=True, separators=(",", ":")).encode())
    return (
        PROMPT_SIGNATURE_VERSION,
        len(tokens),
        *struct.unpack("<7i", digest.digest()[:28]),
    )


def init_welm_vlm_pd_request(req, model_config):
    if is_welm_vlm(model_config.hf_config) and not req.finished():
        req.welm_vlm_pd_signature = welm_vlm_prompt_signature(req)
