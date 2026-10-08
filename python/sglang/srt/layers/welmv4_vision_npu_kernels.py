from __future__ import annotations

import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cann import libdevice
except ImportError:
    libdevice = None


def validate_gelu_runtime():
    if not callable(getattr(tl, "erf", None)):
        raise RuntimeError("Vision fused GELU requires Triton Ascend tl.erf")
    if libdevice is None or not callable(getattr(libdevice, "tanh", None)):
        raise RuntimeError(
            "Vision fused GELU requires triton.language.extra.cann.libdevice.tanh"
        )


@triton.jit
def _linear_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows: tl.constexpr,
    output_width: tl.constexpr,
    input_width: tl.constexpr,
    has_bias: tl.constexpr,
    mode: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    column_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
    reduction_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), tl.float32)
    for start in range(tl.cdiv(input_width, block_k)):
        reduction = start * block_k + reduction_offsets
        left = tl.load(
            input_ptr + row_offsets[:, None] * input_width + reduction[None, :],
            (row_offsets[:, None] < rows) & (reduction[None, :] < input_width),
            0,
        )
        right = tl.load(
            weight_ptr + column_offsets[None, :] * input_width + reduction[:, None],
            (column_offsets[None, :] < output_width)
            & (reduction[:, None] < input_width),
            0,
        )
        accumulator = tl.dot(left, right, accumulator)
    if has_bias:
        bias = tl.load(bias_ptr + column_offsets, column_offsets < output_width, 0).to(
            tl.float32
        )
        accumulator = accumulator + bias[None, :]
    values = accumulator.to(output_ptr.dtype.element_ty).to(tl.float32)
    if mode == "tanh":
        argument = 0.7978845608028654 * (values + 0.044715 * values * values * values)
        activated = 0.5 * values * (1.0 + libdevice.tanh(argument))
    else:
        activated = 0.5 * values * (1.0 + tl.erf(values * 0.7071067811865476))
    tl.store(
        output_ptr + row_offsets[:, None] * output_width + column_offsets[None, :],
        activated,
        (row_offsets[:, None] < rows) & (column_offsets[None, :] < output_width),
    )


def linear_gelu(
    hidden: torch.Tensor, weight: torch.Tensor, bias, mode: str
) -> torch.Tensor:
    validate_gelu_runtime()
    rows = hidden.numel() // hidden.shape[-1]
    output = torch.empty(
        (*hidden.shape[:-1], weight.shape[0]), device=hidden.device, dtype=hidden.dtype
    )
    _linear_gelu_kernel[(triton.cdiv(rows, 32), triton.cdiv(weight.shape[0], 64))](
        hidden,
        weight,
        bias,
        output,
        rows,
        weight.shape[0],
        weight.shape[1],
        bias is not None,
        mode,
        32,
        64,
        128,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _add_layer_norm_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    sum_ptr,
    output_ptr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    hidden = tl.load(hidden_ptr + row * width + columns, columns < width, 0).to(
        tl.float32
    )
    residual = tl.load(residual_ptr + row * width + columns, columns < width, 0).to(
        tl.float32
    )
    summed = (hidden + residual).to(sum_ptr.dtype.element_ty)
    values = summed.to(tl.float32)
    mean = tl.sum(tl.where(columns < width, values, 0.0), 0) / width
    centered = tl.where(columns < width, values - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / width
    weight = tl.load(weight_ptr + columns, columns < width, 0).to(tl.float32)
    bias = tl.load(bias_ptr + columns, columns < width, 0).to(tl.float32)
    normalized = centered * tl.rsqrt(variance + epsilon) * weight + bias
    tl.store(sum_ptr + row * width + columns, summed, columns < width)
    tl.store(output_ptr + row * width + columns, normalized, columns < width)


def add_layer_norm(hidden, residual, weight, bias, epsilon):
    summed = torch.empty_like(hidden)
    output = torch.empty_like(hidden)
    width = hidden.shape[-1]
    _add_layer_norm_kernel[(hidden.numel() // width,)](
        hidden,
        residual,
        weight,
        bias,
        summed,
        output,
        width,
        epsilon,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return summed, output
