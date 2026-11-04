# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# HCCL ctypes signatures adapted from vllm-ascend's pyhccl_wrapper.py.

"""Out-of-place HCCL all-reduce using an existing ProcessGroupHCCL.

The process group owns both the communicator and the communication stream. This
wrapper borrows them; it never creates or destroys an HCCL communicator.
"""

import ctypes
import os

import torch
import torch_npu
from torch.distributed import ProcessGroup


class PyHcclCommunicator:
    # HcclDataType values from hccl_types.h. Only floating-point SUM is needed
    # by SGLang's ordinary model all-reduce.
    _DTYPES = {torch.float16: 3, torch.float32: 4, torch.bfloat16: 11}

    def __init__(self, group: ProcessGroup, device: torch.device):
        self.group = group
        self.device = device
        backend = group._get_backend(device)
        for name in ("get_hccl_comm_name", "get_hccl_comm", "_get_stream_id"):
            if not callable(getattr(backend, name, None)):
                raise RuntimeError(
                    "SGLANG_NPU_ALLREDUCE_OUTPLACE requires ProcessGroupHCCL."
                    f"{name}; this torch_npu backend does not expose it."
                )

        library_path = os.environ.get("HCCL_SO_PATH") or "libhccl.so"
        try:
            # Keep the CDLL alive as long as the borrowed function pointers.
            self._library = ctypes.CDLL(library_path)
            self._all_reduce = self._library.HcclAllReduce
            self._all_reduce.restype = ctypes.c_int
            self._all_reduce.argtypes = [
                ctypes.c_void_p,  # sendBuf
                ctypes.c_void_p,  # recvBuf
                ctypes.c_uint64,  # count
                ctypes.c_int,  # HcclDataType
                ctypes.c_int,  # HcclReduceOp
                ctypes.c_void_p,  # HcclComm
                ctypes.c_void_p,  # aclrtStream
            ]
            self._error_string = self._library.HcclGetErrorString
            self._error_string.restype = ctypes.c_char_p
            self._error_string.argtypes = [ctypes.c_int]
        except (OSError, AttributeError) as exc:
            raise RuntimeError(
                "Cannot load HCCL for SGLANG_NPU_ALLREDUCE_OUTPLACE from "
                f"{library_path!r}. HCCL_SO_PATH must refer to the same HCCL "
                "library used by torch_npu."
            ) from exc

        with torch.npu.device(device):
            # Initialize through the configured PG before borrowing its handle;
            # get_hccl_comm alone can initialize with default HCCL options.
            backend.get_hccl_comm_name(device.index)
            self._comm = backend.get_hccl_comm(device.index)
            stream_id = backend._get_stream_id()
            if not self._comm or stream_id < 0:
                raise RuntimeError("ProcessGroupHCCL communicator/stream is unavailable")
            current_stream = torch.npu.current_stream(device)
            self._stream = torch.npu.Stream(
                stream_id=stream_id,
                device_index=device.index,
                device_type=current_stream.device_type,
            )

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self._comm is None:
            raise RuntimeError("PyHcclCommunicator is closed")
        if x.device != self.device or x.dtype not in self._DTYPES:
            raise ValueError(
                "Out-of-place HCCL all-reduce requires FP16/BF16/FP32 on "
                f"{self.device}, got {x.dtype} on {x.device}"
            )
        # Base ND/NCHW formats have no physical padding. Internal formats such
        # as NZ cannot be sent using logical numel without an explicit conversion.
        if (
            x.layout != torch.strided
            or not x.is_contiguous()
            or torch_npu.get_npu_format(x) not in (0, 2)
        ):
            raise ValueError(
                "Out-of-place HCCL all-reduce requires a contiguous ND/NCHW tensor"
            )

        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if x.numel() == 0:
            return out

        with torch.npu.device(self.device):
            current_stream = torch.npu.current_stream(self.device)
            different_stream = current_stream != self._stream
            if different_stream:
                self._stream.wait_stream(current_stream)
            x.record_stream(self._stream)
            out.record_stream(self._stream)

            # Read npu_stream on every call, AFTER the event/allocator work.
            # Its torch_npu getter drains the host task queue before returning
            # the raw ACL stream. Caching that pointer could let this direct
            # HCCL call overtake queued producers or stream-wait events.
            result = self._all_reduce(
                x.data_ptr(),
                out.data_ptr(),
                x.numel(),
                self._DTYPES[x.dtype],
                0,  # HCCL_REDUCE_SUM
                self._comm,
                self._stream.npu_stream,
            )
            if result != 0:
                detail = self._error_string(result)
                message = detail.decode("utf-8", errors="replace") if detail else ""
                raise RuntimeError(f"HcclAllReduce failed ({result}): {message}")
            if different_stream:
                current_stream.wait_stream(self._stream)
        return out

    def close(self) -> None:
        if self._comm is not None:
            # Direct calls are not WorkHCCL objects. Complete them before the
            # owning ProcessGroup destroys the borrowed native communicator.
            with torch.npu.device(self.device):
                self._stream.synchronize()
            self._comm = None
