#!/usr/bin/env python3
"""Replay a compact WeLM FlashAttn debug dump on one NPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch_npu


def to_npu(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=False)
    return value


def summarize(name: str, tensor: torch.Tensor) -> None:
    values = tensor.detach().float()
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} finite={torch.isfinite(values).all().item()} "
        f"min={values.min().item():.7g} max={values.max().item():.7g} "
        f"mean={values.mean().item():.7g}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, help="Path to a .pt replay dump")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--mode",
        choices=("metadata", "flash", "both"),
        default="both",
        help="Run metadata only, FlashAttn with dumped metadata, or both",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally save metadata and attention output on CPU",
    )
    parser.add_argument(
        "--preserve-cache-shape",
        action="store_true",
        help=(
            "Expand a compact dump back to the production cache block count; "
            "unreferenced blocks are zero-filled"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_npu.npu.set_device(args.device)

    # Import after selecting the device because the extension queries NPU
    # properties while registering FlashAttn.
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    attrs = payload["attributes"]
    q = to_npu(payload["q"], args.device)
    k = to_npu(payload["k"], args.device)
    v = to_npu(payload["v"], args.device)
    block_table = to_npu(payload["block_table"], args.device)
    cu_seqlens_q = to_npu(payload["cu_seqlens_q"], args.device)
    seqused_kv = to_npu(payload["seqused_kv"], args.device)
    sinks = to_npu(payload["sinks"], args.device)
    attn_mask = to_npu(payload["attn_mask"], args.device)
    dumped_metadata = to_npu(payload["metadata"], args.device)

    if args.preserve_cache_shape and payload.get("compact_kv", False):
        original_block_ids = payload["original_block_ids"].to(
            device=args.device, dtype=torch.long
        )
        original_cache_num_blocks = payload["original_cache_num_blocks"]
        full_k = torch.zeros(
            (original_cache_num_blocks, *k.shape[1:]),
            dtype=k.dtype,
            device=args.device,
        )
        full_v = torch.zeros(
            (original_cache_num_blocks, *v.shape[1:]),
            dtype=v.dtype,
            device=args.device,
        )
        full_k.index_copy_(0, original_block_ids, k)
        full_v.index_copy_(0, original_block_ids, v)
        block_table = original_block_ids[
            block_table.to(dtype=torch.long)
        ].to(dtype=torch.int32)
        k = full_k
        v = full_v
        print(
            f"restored production cache shape with "
            f"{original_cache_num_blocks} blocks",
            flush=True,
        )

    print(f"dump={args.dump}", flush=True)
    print(f"attributes={attrs}", flush=True)
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqused_kv", seqused_kv),
        ("sinks", sinks),
        ("attn_mask", attn_mask),
        ("dumped_metadata", dumped_metadata),
    ):
        if tensor is not None:
            print(
                f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} contiguous={tensor.is_contiguous()}",
                flush=True,
            )

    metadata = dumped_metadata
    if args.mode in ("metadata", "both"):
        begin = time.perf_counter()
        metadata = flash_attn_metadata(
            attrs["num_heads_q"],
            attrs["num_heads_kv"],
            attrs["head_dim"],
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            mask_mode=attrs["mask_mode"],
            win_left=attrs["win_left"],
            win_right=attrs["win_right"],
            layout_q=attrs["layout_q"],
            layout_kv=attrs["layout_kv"],
            layout_out=attrs["layout_out"],
        )
        torch_npu.npu.synchronize()
        print(
            f"flash_attn_metadata: PASS elapsed_ms="
            f"{(time.perf_counter() - begin) * 1000:.3f}",
            flush=True,
        )
        summarize("metadata", metadata)

    attention_output = None
    if args.mode in ("flash", "both"):
        begin = time.perf_counter()
        attention_output, softmax_lse = flash_attn(
            q,
            k,
            v,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            sinks=sinks,
            attn_mask=attn_mask,
            metadata=metadata,
            softmax_scale=attrs["softmax_scale"],
            mask_mode=attrs["mask_mode"],
            win_left=attrs["win_left"],
            win_right=attrs["win_right"],
            layout_q=attrs["layout_q"],
            layout_kv=attrs["layout_kv"],
            layout_out=attrs["layout_out"],
            return_softmax_lse=attrs["return_softmax_lse"],
        )
        torch_npu.npu.synchronize()
        print(
            f"flash_attn: PASS elapsed_ms="
            f"{(time.perf_counter() - begin) * 1000:.3f}",
            flush=True,
        )
        summarize("attention_output", attention_output)
        print(
            f"softmax_lse: shape={tuple(softmax_lse.shape)} "
            f"dtype={softmax_lse.dtype}",
            flush=True,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata.detach().cpu(),
                "attention_output": (
                    attention_output.detach().cpu()
                    if attention_output is not None
                    else None
                ),
            },
            args.output,
        )
        print(f"saved output to {args.output}", flush=True)


if __name__ == "__main__":
    main()
