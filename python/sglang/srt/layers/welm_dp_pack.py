"""Fixed-storage DP staging. Invalid input rows, including NaNs, are not read."""

import torch


def pack_dp_slot(
    output, source, real_rows, dp_rank, offset, *, contribute=True, fused=False
):
    if output.dim() != 2 or source.dim() != 2 or output.shape[1] != source.shape[1]:
        raise ValueError("WeLM DP pack requires matching row-major hidden widths")
    if output.device != source.device or output.dtype != source.dtype:
        raise ValueError("WeLM DP pack requires matching device and dtype")
    if real_rows.device != source.device or not 0 <= dp_rank < real_rows.numel():
        raise ValueError("WeLM DP pack requires local device counts for this DP rank")
    if offset < 0 or offset + source.shape[0] > output.shape[0]:
        raise ValueError("WeLM DP pack source exceeds its output slot")
    if not output.is_contiguous() or not source.is_contiguous():
        raise ValueError("WeLM DP pack requires contiguous storage")
    if not output.numel():
        return output
    if fused and output.device.type != "cpu":
        from sglang.srt.layers.welm_dp_pack_triton import pack_kernel

        size = output.numel()
        pack_kernel[((size + 1023) // 1024,)](
            output,
            source,
            real_rows,
            size,
            source.shape[0],
            output.shape[1],
            dp_rank,
            offset,
            contribute,
            BLOCK=1024,
        )
    else:
        output.zero_()
        if contribute and source.shape[0]:
            rows = source.shape[0]
            valid = torch.arange(rows, device=source.device) < real_rows[dp_rank]
            output[offset : offset + rows].copy_(torch.where(valid[:, None], source, 0))
    return output
