"""Standalone NPU correctness and eager-device-time checks for vision fusions.

python test/manual/test_welmv4_vision_npu_fusions.py --op all
python test/manual/test_welmv4_vision_npu_fusions.py --op norm --probe-native-add-layer-norm

No MC2 collectives are used. Timings synchronize NPU events. Passing tolerances
is a kernel smoke check, not a substitute for model-level accuracy evaluation.
"""

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch
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
    if isinstance(actual, tuple):
        return {
            f"output_{index}": _check(value, expected[index], args)
            for index, value in enumerate(actual)
        }
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
    if error is not None:
        raise error
    baseline = _measure(reference, args.warmup, args.iterations)
    optimized = _measure(fused, args.warmup, args.iterations)
    result = {
        "op": label,
        "shape": shape,
        "status": "PASS",
        **accuracy,
        "reference": baseline,
        "fused": optimized,
        "speedup": baseline["mean_us"] / optimized["mean_us"],
    }
    if extra:
        result.update(extra)
    print(json.dumps(result), flush=True)


def _test_gelu(kernels, args, device):
    kernels.validate_gelu_runtime()
    for rows, output_width, input_width, mode in (
        (2760, 1076, 1152, "tanh"),  # TP4 FFN
        (2760, 4304, 1152, "tanh"),  # DP / Ulysses full FFN
        (690, 1152, 4608, "none"),  # TP4 projector
        (690, 4608, 4608, "none"),  # DP full projector
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


def _test_rope(kernels, args, device):
    for tokens, heads in ((2760, 4), (2760, 16), (3, 4)):
        dim = 72
        qkv = torch.randn(tokens, heads, 3, dim, device=device, dtype=torch.bfloat16)
        query, key = qkv[:, :, 0, :], qkv[:, :, 1, :]
        angles = torch.randn(tokens, dim // 2, device=device, dtype=torch.float32)
        for full_width in (False, True):
            cos, sin = angles.cos(), angles.sin()
            if full_width:
                cos, sin = torch.cat((cos, cos), -1), torch.cat((sin, sin), -1)

            def reference():
                cosine, sine = (
                    cos[:, : dim // 2].unsqueeze(1),
                    sin[:, : dim // 2].unsqueeze(1),
                )

                def rotate(value):
                    lo, hi = value.float().chunk(2, dim=-1)
                    return torch.cat(
                        (lo * cosine - hi * sine, lo * sine + hi * cosine), -1
                    ).bfloat16()

                return rotate(query), rotate(key)

            def fused():
                return kernels.apply_rope(query, key, cos, sin)

            for actual, expected in zip(fused(), reference()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _report(
                "rope_qk_head72",
                [tokens, heads, dim],
                reference,
                fused,
                args,
                {
                    "full_width_coefficients": full_width,
                    "input_is_strided": not query.is_contiguous(),
                    "bf16_output_exact": True,
                },
            )


def _probe_native_add_layer_norm(args, device, torch_npu):
    """Diagnostic only: never selects the native operation for model execution."""
    operation = getattr(torch_npu, "npu_add_layer_norm", None)
    if not callable(operation):
        print(
            json.dumps({"op": "native_add_layer_norm_probe", "status": "UNAVAILABLE"}),
            flush=True,
        )
        return
    width = 1152
    hidden = torch.randn(11, width, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    weight = (1 + 0.1 * torch.randn(width, device=device)).bfloat16()
    bias = (0.1 * torch.randn(width, device=device)).bfloat16()
    rounded_sum = hidden + residual
    rounded = F.layer_norm(rounded_sum, (width,), weight, bias, 1e-6)
    unrounded = F.layer_norm(
        hidden.float() + residual.float(), (width,), weight.float(), bias.float(), 1e-6
    ).bfloat16()
    try:
        normalized, _, _, summed = operation(hidden, residual, weight, bias, 1e-6, True)
        torch.npu.synchronize()
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "op": "native_add_layer_norm_probe",
                    "status": "UNSUPPORTED",
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        return
    print(
        json.dumps(
            {
                "op": "native_add_layer_norm_probe",
                "status": "DIAGNOSTIC_ONLY",
                "residual_matches_bf16_sum": torch.equal(summed, rounded_sum),
                "normalized_exactly_matches_rounded": torch.equal(normalized, rounded),
                "max_error_vs_rounded": (normalized.float() - rounded.float())
                .abs()
                .max()
                .item(),
                "max_error_vs_unrounded": (normalized.float() - unrounded.float())
                .abs()
                .max()
                .item(),
                "rounded_and_unrounded_differ": not torch.equal(rounded, unrounded),
                "model_uses_native_operator": False,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", "gelu", "norm", "rope"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--probe-native-add-layer-norm", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.rtol < 0 or args.atol < 0:
        parser.error(
            "warmup/iterations must be positive; tolerances must be nonnegative"
        )
    import torch_npu

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    torch.manual_seed(20261008)
    kernels = _load_kernels()
    with torch.inference_mode():
        if args.op in ("all", "gelu"):
            _test_gelu(kernels, args, device)
        if args.op in ("all", "norm"):
            _test_add_norm(kernels, args, device)
        if args.op in ("all", "rope"):
            _test_rope(kernels, args, device)
        if args.probe_native_add_layer_norm:
            _probe_native_add_layer_norm(args, device, torch_npu)


if __name__ == "__main__":
    main()
