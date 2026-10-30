"""Opt-in eager WeLM prefill experiment using paged FIA v2 and sink LSE.

Keep CPU sequence lengths outside graph capture. FlashAttn's graph-only device
length buffers cannot be substituted for FIA's host length attributes.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def validate_fia_prefill_config(*, use_fia, prefill_backend, attn_cp_size):
    if not use_fia:
        raise ValueError("WELM_NPU_PREFILL_USE_FIA=1 requires ASCEND_USE_FIA=1")
    if prefill_backend != "disabled":
        raise ValueError(
            "WELM_NPU_PREFILL_USE_FIA=1 requires eager prefill. Set "
            "--disable-prefill-cuda-graph and remove explicit prefill graph "
            "backend/config overrides for BOTH FIA and FlashAttn A/B runs."
        )
    if attn_cp_size != 1:
        raise ValueError("The WeLM FIA prefill experiment does not support attention CP")


def apply_sink_lse(attn_out, softmax_lse, sinks, input_layout):
    """Add a zero-value sink to the denominator, preserving each head's logit.

    If Z = sum(exp(score)), FIA returns O and log(Z) without the sink. The
    desired output is O * Z/(Z + exp(sink)) = O * sigmoid(log(Z) - sink).
    This is the fd733ab69a28 FIA LSE path; do not drop or double-count sinks.
    """
    if input_layout == "TND":
        lse = softmax_lse.reshape(attn_out.shape[0], attn_out.shape[1], 1)
        sink = sinks.float().reshape(1, -1, 1)
    elif input_layout == "BSND":
        # FIA's non-TND LSE has layout [B,N,S,1].
        lse = softmax_lse.transpose(1, 2)
        sink = sinks.float().reshape(1, 1, -1, 1)
    else:
        raise ValueError(f"Unsupported FIA sink LSE layout: {input_layout}")
    return (attn_out.float() * torch.sigmoid(lse.float() - sink)).to(attn_out.dtype)


def _cpu_lengths(value, name):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"WeLM FIA prefill requires authoritative CPU {name}")
    if value.ndim != 1:
        raise ValueError(f"WeLM FIA prefill requires one-dimensional {name}")
    result = value.tolist()
    if any(int(n) != n or n < 0 for n in result):
        raise ValueError(f"Invalid WeLM FIA prefill {name}: {result}")
    return [int(n) for n in result]


def forward_fia_prefill(
    q,
    k_cache,
    v_cache,
    *,
    layer,
    page_size: int,
    block_table,
    q_lens,
    kv_lens,
    atten_mask,
    fia: Callable,
    use_tnd: bool,
    sinks: Optional[torch.Tensor] = None,
    mirror_prefill: bool = False,
):
    """Read existing Full/SWA KV pages without rewriting caches or metadata.

    Normal prefill has packed live Q rows followed by optional DP padding.
    Mirror consumers instead have one physical row per request. Drop inactive
    requests from FIA's lengths/tables, then restore the original output layout.
    """
    kv_lengths = _cpu_lengths(kv_lens, "KV lengths")
    q_lengths = (
        [int(n > 0) for n in kv_lengths]
        if mirror_prefill
        else _cpu_lengths(q_lens, "Q lengths")
    )
    if len(q_lengths) != len(kv_lengths) or any(
        nq > nk for nq, nk in zip(q_lengths, kv_lengths)
    ):
        raise ValueError("WeLM FIA prefill requires 0 <= Q length <= KV length per request")
    query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
    physical_rows = query.shape[0]
    if (
        sum(q_lengths) > physical_rows
        or block_table.shape[0] != len(q_lengths)
        or (mirror_prefill and physical_rows != len(q_lengths))
    ):
        raise ValueError("WeLM FIA prefill Q rows/page tables do not match CPU lengths")
    live = [i for i, n in enumerate(q_lengths) if n > 0]
    output_shape = (physical_rows, layer.tp_q_head_num * layer.v_head_dim)
    if not live:
        return q.new_zeros(output_shape)
    if sinks is not None and sinks.numel() != layer.tp_q_head_num:
        raise ValueError("WeLM FIA sink count must match the local Q head count")
    if sinks is not None and layer.qk_head_dim != layer.v_head_dim:
        raise ValueError("FIA sink correction requires equal QK/V head dimensions for LSE")

    indices = None
    if len(live) != len(q_lengths):
        indices = torch.tensor(live, dtype=torch.long, device=q.device)
        block_table = block_table.index_select(0, indices)
    if mirror_prefill and indices is not None:
        query = query.index_select(0, indices)
    else:
        query = query[: sum(q_lengths)]
    query = query.contiguous()
    q_lengths = [q_lengths[i] for i in live]
    kv_lengths = [kv_lengths[i] for i in live]
    key = k_cache.view(-1, page_size, layer.tp_k_head_num * layer.qk_head_dim)
    value = v_cache.view(-1, page_size, layer.tp_v_head_num * layer.v_head_dim)
    window = layer.sliding_window_size
    is_swa = window is not None and window >= 0
    common = dict(
        key=key,
        value=value,
        block_size=page_size,
        num_query_heads=layer.tp_q_head_num,
        num_key_value_heads=layer.tp_k_head_num,
        softmax_scale=layer.scaling,
        sparse_mode=4 if is_swa else 3,
        pre_tokens=window if is_swa else 2147483647,
        next_tokens=0 if is_swa else 2147483647,
        learnable_sink=None,
        return_softmax_lse=sinks is not None,
    )
    if use_tnd:
        cumulative, total = [], 0
        for length in q_lengths:
            total += length
            cumulative.append(total)
        result, lse = fia(
            query=query,
            block_table=block_table,
            input_layout="TND",
            atten_mask=atten_mask,
            actual_seq_qlen=cumulative,
            actual_seq_kvlen=kv_lengths,
            **common,
        )
        if sinks is not None:
            result = apply_sink_lse(result, lse, sinks, "TND")
    else:
        parts, offset = [], 0
        for i, length in enumerate(q_lengths):
            part, lse = fia(
                query=query[None, offset : offset + length],
                block_table=block_table[i : i + 1],
                input_layout="BSND",
                atten_mask=atten_mask.unsqueeze(0),
                actual_seq_qlen=[length],
                actual_seq_kvlen=[kv_lengths[i]],
                **common,
            )
            if sinks is not None:
                part = apply_sink_lse(part, lse, sinks, "BSND")
            parts.append(part[0])
            offset += length
        result = torch.cat(parts, dim=0)
    result = result.reshape(-1, output_shape[1])
    if result.shape[0] == physical_rows:
        return result
    output = q.new_zeros(output_shape)
    if mirror_prefill:
        output.index_copy_(0, indices, result)
    else:
        output[: result.shape[0]].copy_(result)
    return output
