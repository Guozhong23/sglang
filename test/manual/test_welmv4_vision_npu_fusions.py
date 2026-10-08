"""Standalone NPU correctness and eager-device-time checks for vision fusions.

python test/manual/test_welmv4_vision_npu_fusions.py --op all
torchrun --standalone --nproc_per_node=4 test/manual/test_welmv4_vision_npu_fusions.py --op mc2
"""

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as distributed
import torch.nn.functional as F


def _load_kernels():
    path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/layers/welmv4_vision_npu_kernels.py"
    )
    spec = importlib.util.spec_from_file_location(
        "welmv4_vision_fusion_manual_kernels", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()
    measurements = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000.0)
    measurements.sort()
    return {
        "mean_us": statistics.mean(measurements),
        "p50_us": statistics.median(measurements),
        "p90_us": measurements[
            min(len(measurements) - 1, math.ceil(len(measurements) * 0.9) - 1)
        ],
    }


def _check(actual, expected, args):
    difference = actual.float() - expected.float()
    result = {
        "max_abs_error": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "reference_rms": expected.float().square().mean().sqrt().item(),
    }
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    return result


def _report(label, shape, reference, fused, args, extra=None):
    expected = reference()
    actual = fused()
    error = None
    accuracy = {}
    try:
        accuracy = _check(actual, expected, args)
    except AssertionError as exc:
        error = exc
    is_distributed = distributed.is_initialized()
    if is_distributed:
        passed = torch.tensor(
            int(error is None), device=actual.device, dtype=torch.int32
        )
        distributed.all_reduce(passed, op=distributed.ReduceOp.MIN)
        if not passed.item():
            raise AssertionError(
                f"{label}: at least one TP rank failed accuracy; local error={error}"
            )
        distributed.barrier()
    elif error is not None:
        raise error
    baseline = _measure(reference, args.warmup, args.iterations)
    if is_distributed:
        distributed.barrier()
    optimized = _measure(fused, args.warmup, args.iterations)
    result = {
        "op": label,
        "rank": int(os.environ.get("RANK", "0")),
        "shape": shape,
        "status": "PASS",
        **accuracy,
        "reference": baseline,
        "fused": optimized,
        "speedup": baseline["mean_us"] / optimized["mean_us"],
    }
    if extra:
        result.update(extra)
    if is_distributed:
        maxima = torch.tensor(
            [baseline["mean_us"], optimized["mean_us"]],
            device=actual.device,
            dtype=torch.float32,
        )
        distributed.all_reduce(maxima, op=distributed.ReduceOp.MAX)
        maximum_reference, maximum_fused = maxima.tolist()
        result["tp_max_rank_mean_us"] = {
            "reference": maximum_reference,
            "fused": maximum_fused,
            "speedup": maximum_reference / maximum_fused,
        }
    print(json.dumps(result), flush=True)


def _test_gelu(kernels, args, device):
    kernels.validate_gelu_runtime()
    for rows, output_width, input_width, mode in (
        (2760, 1076, 1152, "tanh"),
        (690, 1152, 4608, "none"),
        (3, 1076, 72, "tanh"),
        (3, 1152, 72, "none"),
    ):
        hidden = torch.randn(rows, input_width, device=device, dtype=torch.bfloat16)
        weight = (
            torch.randn(output_width, input_width, device=device)
            / math.sqrt(input_width)
        ).bfloat16()
        bias = (torch.randn(output_width, device=device) * 0.1).bfloat16()
        reference = lambda: F.gelu(F.linear(hidden, weight, bias), approximate=mode)
        fused = lambda: kernels.linear_gelu(hidden, weight, bias, mode)
        _report(
            "matmul_bias_gelu_" + mode,
            [rows, output_width, input_width],
            reference,
            fused,
            args,
        )


def _test_add_norm(kernels, args, device):
    for rows in (2760, 3):
        width = 1152
        hidden = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
        residual = torch.randn_like(hidden)
        weight = (1 + 0.1 * torch.randn(width, device=device)).bfloat16()
        bias = (0.1 * torch.randn(width, device=device)).bfloat16()
        epsilon = 1e-6
        summed, _ = kernels.add_layer_norm(hidden, residual, weight, bias, epsilon)
        torch.testing.assert_close(summed, hidden + residual, rtol=0, atol=0)
        rounded = F.layer_norm(hidden + residual, (width,), weight, bias, epsilon)
        unrounded = F.layer_norm(
            hidden.float() + residual.float(),
            (width,),
            weight.float(),
            bias.float(),
            epsilon,
        ).bfloat16()
        if torch.equal(rounded, unrounded):
            raise AssertionError(
                "The rounding test fixture failed to distinguish FP32 and BF16 residual sums"
            )
        reference = lambda: F.layer_norm(
            hidden + residual, (width,), weight, bias, epsilon
        )
        fused = lambda: kernels.add_layer_norm(hidden, residual, weight, bias, epsilon)[
            1
        ]
        _report(
            "add_layer_norm",
            [rows, width],
            reference,
            fused,
            args,
            {"bf16_residual_exact": True},
        )


def _test_mc2(args, device, torch_npu):
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise ValueError(
            "MC2 validation requires torchrun --standalone --nproc_per_node=4"
        )
    distributed.init_process_group(backend="hccl")
    try:
        process_group = distributed.group.WORLD
        rank = distributed.get_rank()
        backend = process_group._get_backend(device)
        hcom = backend.get_hccl_comm_name(process_group.rank())
        operation = getattr(torch_npu, "npu_mm_all_reduce_base", None)
        if not callable(operation):
            raise RuntimeError("torch_npu.npu_mm_all_reduce_base is unavailable")
        for rows, output_width, input_width in (
            (2760, 1152, 288),
            (2760, 1152, 1076),
            (690, 2048, 1152),
        ):
            hidden = torch.randn(rows, input_width, device=device, dtype=torch.bfloat16)
            weight = (
                torch.randn(output_width, input_width, device=device)
                / math.sqrt(input_width)
            ).bfloat16()
            bias = torch.linspace(-0.25, 0.25, output_width, device=device).bfloat16()
            local_bias = bias if rank == 0 else None

            def reference():
                output = F.linear(hidden, weight, local_bias)
                distributed.all_reduce(output)
                return output

            def fused():
                return operation(
                    hidden,
                    weight.transpose(0, 1),
                    hcom,
                    reduce_op="sum",
                    bias=local_bias,
                    comm_turn=0,
                    comm_mode=args.comm_mode,
                )

            distributed.barrier()
            _report(
                "matmul_all_reduce",
                [rows, output_width, input_width],
                reference,
                fused,
                args,
                {
                    "world_size": 4,
                    "bias_policy": "rank0_only",
                    "comm_mode": args.comm_mode,
                },
            )
    finally:
        distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", "gelu", "norm", "mc2"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--comm-mode", choices=("ccu", "ai_cpu"), default="ccu")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.rtol < 0 or args.atol < 0:
        parser.error(
            "warmup/iterations must be positive; tolerances must be nonnegative"
        )
    import torch_npu

    local_rank = int(os.environ.get("LOCAL_RANK", str(args.device)))
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    torch.manual_seed(20261008 + local_rank)
    kernels = _load_kernels() if args.op != "mc2" else None
    with torch.inference_mode():
        if args.op in ("all", "gelu"):
            _test_gelu(kernels, args, device)
        if args.op in ("all", "norm"):
            _test_add_norm(kernels, args, device)
        if args.op == "mc2" or (
            args.op == "all" and int(os.environ.get("WORLD_SIZE", "1")) == 4
        ):
            _test_mc2(args, device, torch_npu)


if __name__ == "__main__":
    main()
