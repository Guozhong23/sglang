#!/usr/bin/env python3
"""EP4 accuracy and performance probe for WeLM's Ascend MegaMoE shape.

Run with torchrun.  The semantic reference keeps only local experts on each
rank, accumulates their BF16 contributions for all source rows, then all-reduces
the partial results. MegaMoE additionally quantizes activations to MXFP8, so the
reported cosine/NRMSE thresholds intentionally measure quantization-level
agreement instead of bit equality.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

os.environ.setdefault("NPU_OPS_TRANSFORMER_OPS_IMPORT_MODE", "minimal")

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401

from npu_ops_transformer.ops.mega_moe import (
    get_symm_buffer_for_mega_moe,
    mega_moe,
)


@dataclass
class Metrics:
    max_abs: float
    mean_abs: float
    nrmse: float
    cosine: float


def parse_int_list(value: str) -> List[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def get_e8m0_dtype():
    dtype = getattr(torch, "float8_e8m0fnu", None)
    if dtype is None:
        dtype = getattr(torch_npu, "float8_e8m0fnu", None)
    if dtype is None:
        raise RuntimeError("torch/torch_npu does not expose float8_e8m0fnu")
    return dtype


def make_constant_scale(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    # E8M0 exponent byte 127 encodes scale 1.0. Reinterpret; do not cast.
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(
        get_e8m0_dtype()
    )


def make_fp8_weights(
    local_experts: int,
    hidden: int,
    intermediate: int,
    rank: int,
    device: torch.device,
    std: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fp8 = torch.float8_e4m3fn
    w13 = torch.empty(
        (local_experts, 2 * intermediate, hidden), dtype=fp8, device=device
    )
    w2 = torch.empty(
        (local_experts, hidden, intermediate), dtype=fp8, device=device
    )
    # Fill one expert at a time to avoid materializing a second ~768 MiB BF16
    # copy for WeLM's full local-E128 shape.
    for local_e in range(local_experts):
        global_e = rank * local_experts + local_e
        torch.manual_seed(1000003 + global_e)
        w13[local_e].copy_(
            (torch.randn(w13[local_e].shape, dtype=torch.bfloat16, device=device) * std).to(fp8)
        )
        torch.manual_seed(2000003 + global_e)
        w2[local_e].copy_(
            (torch.randn(w2[local_e].shape, dtype=torch.bfloat16, device=device) * std).to(fp8)
        )
    w13_sf = make_constant_scale(
        (local_experts, 2 * intermediate, (hidden + 63) // 64, 2), device
    )
    w2_sf = make_constant_scale(
        (local_experts, hidden, (intermediate + 63) // 64, 2), device
    )
    return w13, w2, w13_sf, w2_sf


def make_case(
    m: int,
    hidden: int,
    topk: int,
    num_experts: int,
    rank: int,
    device: torch.device,
    padded_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(3000001 + rank * 101 + m)
    x = (
        torch.randn((m, hidden), dtype=torch.bfloat16, device=device) * 0.25
    ).contiguous()
    source_rows = torch.arange(m, dtype=torch.int64, device=device) + rank * m
    offsets = torch.arange(topk, dtype=torch.int64, device=device) * 37
    # 37 is coprime with 512, so the WeLM K=10 ids are distinct per row.
    ids = ((source_rows[:, None] * 17 + offsets[None, :]) % num_experts).to(
        torch.int32
    )
    weights = (
        0.05
        + 0.90
        * (((source_rows[:, None] * 13 + offsets[None, :]) % 97).float() / 96.0)
    ).to(torch.bfloat16)
    if padded_rows:
        if padded_rows >= m:
            raise ValueError("padded_rows must be smaller than M")
        weights[-padded_rows:].zero_()
    return x, ids.contiguous(), weights.contiguous()


def all_gather_equal(input_tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    output = torch.empty(
        (input_tensor.shape[0] * world_size, *input_tensor.shape[1:]),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    dist.all_gather_into_tensor(output, input_tensor.contiguous())
    return output


@torch.no_grad()
def bf16_ep_reference(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    full_x = all_gather_equal(x, world_size)
    full_ids = all_gather_equal(ids, world_size)
    full_weights = all_gather_equal(weights, world_size)
    partial = torch.zeros_like(full_x)
    local_experts = w13.shape[0]
    first_expert = rank * local_experts

    for local_e in range(local_experts):
        global_e = first_expert + local_e
        rows_and_slots = torch.nonzero(full_ids == global_e, as_tuple=False)
        if rows_and_slots.numel() == 0:
            continue
        rows = rows_and_slots[:, 0]
        slots = rows_and_slots[:, 1]
        selected = full_x.index_select(0, rows)
        gate_up = torch.matmul(selected, w13[local_e].to(torch.bfloat16).t())
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_out = torch.matmul(activated, w2[local_e].to(torch.bfloat16).t())
        expert_out.mul_(full_weights[rows, slots, None])
        partial.index_add_(0, rows, expert_out)

    dist.all_reduce(partial)
    start = rank * x.shape[0]
    return partial[start : start + x.shape[0]]


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> Metrics:
    actual32 = actual.float()
    reference32 = reference.float()
    diff = actual32 - reference32
    rmse = torch.sqrt(torch.mean(diff * diff))
    reference_rms = torch.sqrt(torch.mean(reference32 * reference32)).clamp_min(1e-12)
    cosine = F.cosine_similarity(
        actual32.reshape(1, -1), reference32.reshape(1, -1), dim=1
    )[0]
    return Metrics(
        max_abs=float(diff.abs().max().item()),
        mean_abs=float(diff.abs().mean().item()),
        nrmse=float((rmse / reference_rms).item()),
        cosine=float(cosine.item()),
    )


def percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=parse_int_list, default=parse_int_list("1,16,128"))
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=512)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--padded-rows", type=int, default=0)
    parser.add_argument(
        "--reference-max-m",
        type=int,
        default=128,
        help="Skip the slower semantic reference above this M; 0 disables it.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.97)
    parser.add_argument("--max-nrmse", type=float, default=0.20)
    parser.add_argument("--max-padded-abs", type=float, default=1e-3)
    parser.add_argument("--csv", default="welm_megamoe_op.csv")
    parser.add_argument("--weight-std", type=float, default=0.015625)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"WeLM validation requires EP4, got world_size={world_size}")
    if args.num_experts % world_size:
        raise RuntimeError("num_experts must be divisible by world_size")

    torch_npu.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl")
    ranks = list(range(world_size))
    ep_group = dist.new_group(ranks=ranks, backend="hccl")
    ep_group._get_backend(device).get_hccl_comm_name(rank, init_comm=True)

    local_experts = args.num_experts // world_size
    w13, w2, w13_sf, w2_sf = make_fp8_weights(
        local_experts,
        args.hidden,
        args.intermediate,
        rank,
        device,
        args.weight_std,
    )
    capacity = max(args.m)
    sym_buffer = get_symm_buffer_for_mega_moe(
        ep_group,
        num_experts=args.num_experts,
        num_max_tokens_per_rank=capacity,
        num_topk=args.topk,
        hidden=args.hidden,
        intermediate_hidden=0,
        max_recv_token_num=0,
        dispatch_quant_mode=4,
        dispatch_quant_out_dtype=24,
        combine_quant_mode=0,
        comm_alg="",
    )

    rows = []
    failed = False
    try:
        for m in args.m:
            # Keep M=1 in the default decode smoke set. Padding validation is
            # meaningful only when the case contains at least one real row.
            case_padded_rows = min(args.padded_rows, max(m - 1, 0))
            x, ids, route_weights = make_case(
                m,
                args.hidden,
                args.topk,
                args.num_experts,
                rank,
                device,
                case_padded_rows,
            )

            def invoke():
                return mega_moe(
                    x,
                    ids,
                    route_weights,
                    [w13],
                    [w2],
                    sym_buffer,
                    l1_weights_sf=[w13_sf],
                    l2_weights_sf=[w2_sf],
                )[0]

            for _ in range(args.warmup):
                output = invoke()
            torch_npu.npu.synchronize()
            latencies_us = []
            for _ in range(args.repeat):
                start = time.perf_counter_ns()
                output = invoke()
                torch_npu.npu.synchronize()
                latencies_us.append((time.perf_counter_ns() - start) / 1000.0)

            result_metrics = None
            if args.reference_max_m > 0 and m <= args.reference_max_m:
                reference = bf16_ep_reference(
                    x, ids, route_weights, w13, w2, rank, world_size
                )
                torch_npu.npu.synchronize()
                result_metrics = metrics(output, reference)
                local_failed = (
                    result_metrics.cosine < args.min_cosine
                    or result_metrics.nrmse > args.max_nrmse
                )
                failure_tensor = torch.tensor(
                    [int(local_failed)], dtype=torch.int32, device=device
                )
                dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
                failed = failed or bool(failure_tensor.item())

            if case_padded_rows:
                padded_max = float(
                    output[-case_padded_rows:].float().abs().max().item()
                )
                if padded_max > args.max_padded_abs:
                    failed = True
            else:
                padded_max = 0.0

            row = {
                "rank": rank,
                "m": m,
                "mean_us": sum(latencies_us) / len(latencies_us),
                "p50_us": percentile(latencies_us, 0.50),
                "p90_us": percentile(latencies_us, 0.90),
                "p99_us": percentile(latencies_us, 0.99),
                "cosine": result_metrics.cosine if result_metrics else "",
                "nrmse": result_metrics.nrmse if result_metrics else "",
                "max_abs": result_metrics.max_abs if result_metrics else "",
                "mean_abs": result_metrics.mean_abs if result_metrics else "",
                "padded_rows": case_padded_rows,
                "padded_max_abs": padded_max,
            }
            rows.append(row)
            print("MEGAMOE_RESULT," + ",".join(f"{k}={v}" for k, v in row.items()), flush=True)

        failure_tensor = torch.tensor([int(failed)], dtype=torch.int32, device=device)
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
        failed = bool(failure_tensor.item())
        # all_gather_object is implemented through the all-gather collective,
        # which is available on HCCL; gather_object is not portable to every
        # torch_npu/HCCL release.
        gathered_rows = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_rows, rows)
        if rank == 0:
            flat_rows = [row for rank_rows in gathered_rows for row in rank_rows]
            with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(flat_rows[0].keys()))
                writer.writeheader()
                writer.writerows(flat_rows)
            print(f"CSV={os.path.abspath(args.csv)}", flush=True)
    finally:
        sym_buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()

    if failed:
        raise SystemExit("MegaMoE validation FAILED; inspect cosine/NRMSE/padding output")
    if rank == 0:
        print("MEGAMOE_VALIDATION=PASS", flush=True)


if __name__ == "__main__":
    main()
