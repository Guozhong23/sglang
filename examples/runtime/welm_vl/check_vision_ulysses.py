#!/usr/bin/env python3
"""HCCL/FIA smoke test of production Ulysses layouts; no model weights needed.

Run with torchrun --standalone --nproc-per-node=4. This checks real NPU
collectives, BF16 72-wide RoPE and segmented FIA, NOT end-to-end VLM accuracy
or a TP-vs-SP performance comparison. The numerical reference replicates
full-sequence computation on each rank purely for validation.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def load_source(name, filename):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        name, root / "python/sglang/srt/layers" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="4;12,16;3,2,6;2760",
        help="semicolon-separated packed segment lengths; odd lengths exercise padding",
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=72)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    cases = [[int(x) for x in case.split(",")] for case in args.cases.split(";")]
    if args.iters < 1 or args.warmup < 0 or args.head_dim <= 0 or args.head_dim % 2:
        parser.error("invalid iteration count or head dimension")
    if any(n <= 0 for case in cases for n in case):
        parser.error("segment lengths must be positive")

    import torch_npu

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    Plan = load_source(
        "welm_ulysses_smoke", "welmv4_vision_ulysses.py"
    ).VisionUlyssesPlan
    rope = load_source(
        "welm_rope_smoke", "welmv4_vision_op.py"
    ).welmv4_vision_apply_rope
    Plan(1, args.heads, world, rank)  # validate before collectives
    dist.init_process_group("hccl")
    group = SimpleNamespace(
        world_size=world,
        rank_in_group=rank,
        all_to_all_single=lambda out, inp: dist.all_to_all_single(out, inp),
        all_gather_into_tensor=lambda out, inp: dist.all_gather_into_tensor(out, inp),
    )

    def fia(qkv, ends):
        q, k, v = qkv
        return torch_npu.npu_fused_infer_attention_score(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            actual_seq_lengths=ends,
            actual_seq_lengths_kv=ends,
            num_heads=q.shape[1],
            num_key_value_heads=k.shape[1],
            scale=args.head_dim**-0.5,
            sparse_mode=0,
            input_layout="TND",
        )[0]

    try:
        with torch.inference_mode():
            for lengths in cases:
                n, hidden = sum(lengths), args.heads * args.head_dim
                plan = Plan(n, args.heads, world, rank)
                # Generate on CPU with identical seed so ranks share weights
                # and reference inputs without adding setup collectives.
                rng = torch.Generator().manual_seed(20261012)

                def random(shape, scale=1):
                    return (torch.randn(shape, generator=rng) * scale).to(
                        device, torch.bfloat16
                    )

                x = random((n, hidden))
                weight = random((3 * hidden, hidden), 0.02)
                bias = random((3 * hidden,), 0.02)
                out_weight = random((hidden, hidden), 0.02)
                out_bias = random((hidden,), 0.02)
                angle = torch.randn((n, args.head_dim // 2), generator=rng).to(device)
                cos, sin = angle.cos(), angle.sin()
                ends = torch.tensor(lengths).cumsum(0).tolist()

                def qkv_projection(rows, coefficients):
                    qkv = F.linear(rows, weight, bias)
                    q, k, v = [
                        y.reshape(-1, args.heads, args.head_dim).contiguous()
                        for y in qkv.chunk(3, -1)
                    ]
                    q, k = rope(q, k, *coefficients)
                    return q, k, v

                reference_qkv = qkv_projection(x, (cos, sin))
                reference = F.linear(
                    fia(reference_qkv, ends).flatten(1), out_weight, out_bias
                )
                local_x = plan.local_slice(x)
                local_coeff = (plan.local_slice(cos, 1), plan.local_slice(sin, 0))
                local_qkv = qkv_projection(local_x, local_coeff)

                # Exact exchange-only check independent of GEMM/FIA rounding.
                head_data = plan.sequence_to_heads(
                    *(plan.local_slice(y) for y in reference_qkv), group
                )
                start = rank * plan.local_heads
                layout_ok = all(
                    torch.equal(actual, full[:, start : start + plan.local_heads])
                    for actual, full in zip(head_data, reference_qkv)
                )
                round_trip = plan.heads_to_sequence(head_data[0], group)
                layout_ok = layout_ok and torch.equal(
                    round_trip, plan.local_slice(reference_qkv[0])
                )

                def attention_exchange():
                    heads = plan.sequence_to_heads(*local_qkv, group)
                    return plan.heads_to_sequence(fia(heads, ends), group)

                local_output = F.linear(
                    attention_exchange().flatten(1), out_weight, out_bias
                )
                actual = plan.gather_sequence(local_output, group)
                delta = actual.float() - reference.float()
                max_abs = delta.abs().max().item()
                relative_l2 = (
                    delta.norm() / reference.float().norm().clamp_min(1e-12)
                ).item()
                numerical_ok = bool(
                    torch.isfinite(actual).all().item()
                ) and torch.allclose(actual, reference, atol=args.atol, rtol=args.rtol)
                passed = torch.tensor(
                    [int(layout_ok and numerical_ok)], dtype=torch.int32, device=device
                )
                dist.all_reduce(passed, op=dist.ReduceOp.MIN)
                if not passed.item():
                    raise AssertionError(
                        f"rank={rank} lengths={lengths}: layout={layout_ok}, "
                        f"max_abs={max_abs}, relative_l2={relative_l2}"
                    )

                for _ in range(args.warmup):
                    attention_exchange()
                torch.npu.synchronize()
                dist.barrier()
                torch.npu.synchronize()
                started = time.perf_counter()
                for _ in range(args.iters):
                    attention_exchange()
                torch.npu.synchronize()
                elapsed = torch.tensor(
                    [(time.perf_counter() - started) * 1e6 / args.iters], device=device
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                if rank == 0:
                    print(
                        json.dumps(
                            dict(
                                lengths=lengths,
                                sp=world,
                                padded_tokens=plan.padded_tokens,
                                layout_exact=True,
                                max_abs_error=max_abs,
                                relative_l2=relative_l2,
                                a2a_fia_a2a_wall_us=elapsed.item(),
                                status="PASS",
                            )
                        ),
                        flush=True,
                    )
        if rank == 0:
            print(
                "ULYSSES_HCCL_FIA_SMOKE=PASS (not a model accuracy verdict)", flush=True
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
