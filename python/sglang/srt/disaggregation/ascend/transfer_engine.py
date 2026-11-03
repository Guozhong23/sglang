import json
import logging
import os
from contextlib import nullcontext
from typing import List

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.utils.network import NetworkAddress
from sglang.srt.utils.npu_pd_affinity import get_pd_thread_affinity

try:
    from memfabric_hybrid import TransferEngine

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)

_DEFAULT_PROTOCOL = "sdma"


class AscendTransferEngine(MooncakeTransferEngine):

    def __init__(
        self,
        hostname: str,
        npu_id: int,
        disaggregation_mode: DisaggregationMode,
    ):
        if import_error is not None:
            logger.warning(
                "Please install memfabric_hybrid, for details, see docs_new/docs/advanced_features/pd_disaggregation.mdx"
            )
            raise import_error

        self.engine = TransferEngine()
        self.hostname = hostname
        self.npu_id = npu_id

        # Centralized storage address of the AscendTransferEngine
        self.store_url = os.getenv("ASCEND_MF_STORE_URL")
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = "Prefill"
        elif disaggregation_mode == DisaggregationMode.DECODE:
            self.role = "Decode"
        else:
            logger.error(f"Unsupported DisaggregationMode: {disaggregation_mode}")
            raise ValueError(f"Unsupported DisaggregationMode: {disaggregation_mode}")
        rpc_port = self.engine.get_rpc_port()
        self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()
        self.initialize()
        if rpc_port == 0:
            rpc_port = self.engine.get_rpc_port()
            self.session_id = NetworkAddress(self.hostname, rpc_port).to_host_port_str()

    def initialize(self) -> None:
        from sglang.srt.distributed.parallel_state import (
            get_world_group,
            get_world_size,
        )

        transfer_protocol = self._get_transfer_protocol()
        trans_op_type = self._resolve_trans_op_type(transfer_protocol)
        hcom_url = os.getenv("ASCEND_MF_HCOM_URL", "")

        if transfer_protocol == "host_rdma":
            hcom_url = self._get_worker_hcom_url(
                hcom_url, get_world_group().rank_in_group
            )
        elif transfer_protocol == "device_rdma":
            tmp_tensor = torch.zeros(1, device="npu")
            output_tensor_list = [
                torch.empty_like(tmp_tensor) for _ in range(get_world_size())
            ]
            # Initialize hccl in advance through all_gather to avoid conflicts with rdma initialization.
            torch.distributed.all_gather(
                output_tensor_list, tmp_tensor, group=get_world_group().device_group
            )
        elif transfer_protocol in ("device_urma", "device_uboe"):
            # URMA selects the device fabric through the SDK. Do not pass a
            # stale Host RDMA NIC/port mapping from an older deployment.
            if hcom_url:
                logger.warning(
                    "Ignoring ASCEND_MF_HCOM_URL for %s; it configures the "
                    "Host RDMA endpoint, not the device fabric.",
                    transfer_protocol,
                )
            hcom_url = ""

        logger.info(
            "Initializing Ascend Transfer Engine: protocol=%s, op_type=%s, "
            "role=%s, npu_id=%d, session=%s",
            transfer_protocol,
            trans_op_type,
            self.role,
            self.npu_id,
            self.session_id,
        )
        # Decode owns the store. Prefill must remain a client so that its first
        # transfer lazily creates a connection to the Decode session.
        pd_affinity = (
            get_pd_thread_affinity() if transfer_protocol == "host_rdma" else None
        )
        with (
            pd_affinity.bind_initialization_thread()
            if pd_affinity is not None
            else nullcontext()
        ):
            ret_value = self.engine.initialize(
                self.store_url,
                self.session_id,
                self.role,
                self.npu_id,
                trans_op_type,
                "Decode",
                hcom_url,
            )
            if ret_value != 0:
                raise RuntimeError(
                    "Ascend Transfer Engine initialization failed: "
                    f"protocol={transfer_protocol}, role={self.role}, "
                    f"npu_id={self.npu_id}, return_code={ret_value}"
                )

    def batch_register(self, ptrs: List[int], lengths: List[int]):
        try:
            ret_value = self.engine.batch_register_memory(ptrs, lengths)
        except Exception as exc:
            raise RuntimeError(
                f"Ascend memory registration failed for {len(ptrs)} buffers."
            ) from exc
        if ret_value != 0:
            raise RuntimeError(
                f"Ascend memory registration failed for {len(ptrs)} buffers: "
                f"return_code={ret_value}"
            )

    @staticmethod
    def _get_transfer_protocol() -> str:
        protocol = os.getenv("ASCEND_MF_TRANSFER_PROTOCOL", "").strip().lower()
        return protocol or _DEFAULT_PROTOCOL

    @staticmethod
    def _resolve_trans_op_type(protocol: str):
        # Resolve only the requested protocol so older SDKs can still use
        # SDMA/RDMA without exposing the 950-series URMA enum members.
        op_type = getattr(TransferEngine.TransDataOpType, protocol.upper(), None)
        if op_type is None:
            supported = ", ".join(
                name.lower()
                for name in dir(TransferEngine.TransDataOpType)
                if name.isupper()
            )
            raise ValueError(
                f"ASCEND_MF_TRANSFER_PROTOCOL={protocol!r} is not supported by "
                "the installed memfabric_hybrid. Available protocols: "
                f"{supported}. Refusing to fall back to {_DEFAULT_PROTOCOL}."
            )
        return op_type

    def _get_worker_hcom_url(self, hcom_url: str, world_rank: int) -> str:
        """Select a URL by runtime NPU ID, then offset its port by world rank."""
        if not hcom_url:
            return hcom_url

        if hcom_url.lstrip().startswith("{"):
            try:
                npu_urls = json.loads(hcom_url)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON NPU-to-URL mapping in ASCEND_MF_HCOM_URL"
                ) from exc

            npu_key = str(self.npu_id)
            if npu_key not in npu_urls:
                raise ValueError(
                    "ASCEND_MF_HCOM_URL mapping has no entry for "
                    f"npu_id={self.npu_id}"
                )
            hcom_url = npu_urls[npu_key]
            if not isinstance(hcom_url, str) or not hcom_url.strip():
                raise ValueError(
                    "ASCEND_MF_HCOM_URL mapping requires a non-empty URL string "
                    f"for npu_id={self.npu_id}"
                )
            hcom_url = hcom_url.strip()

        address, separator, port_str = hcom_url.rpartition(":")
        if not separator or not address.startswith("tcp://"):
            raise ValueError(
                "ASCEND_MF_HCOM_URL must use tcp://<IPv4>:<port> or "
                "tcp://<IPv4>/<mask>:<port>, or a JSON object mapping NPU IDs "
                "to these URLs"
            )

        try:
            base_port = int(port_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid port in ASCEND_MF_HCOM_URL: {hcom_url!r}"
            ) from exc

        # MemFabric adds its group rank to this base port. Reserve eight ports
        # per worker for group ranks 0..7, allowing multiple Prefill peers to
        # join a Decode store. DP-attention does not change world_rank.
        port_stride = 8
        worker_port = base_port + world_rank * port_stride
        if not (1024 <= worker_port and worker_port + port_stride - 1 <= 65535):
            raise ValueError(
                "Resolved ASCEND_MF_HCOM_URL port is out of range: "
                f"base_port={base_port}, world_rank={world_rank}, "
                f"role={self.role}, "
                f"resolved_port_range={worker_port}-{worker_port + port_stride - 1}"
            )

        worker_hcom_url = f"{address}:{worker_port}"
        logger.info(
            "Resolved Ascend Host RDMA endpoint: role=%s, npu_id=%d, "
            "world_rank=%d, base=%s, endpoint=%s",
            self.role,
            self.npu_id,
            world_rank,
            hcom_url,
            worker_hcom_url,
        )
        return worker_hcom_url
