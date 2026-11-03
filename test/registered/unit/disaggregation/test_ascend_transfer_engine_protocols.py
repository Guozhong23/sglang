"""Exercise MemFabric initialization on CPU without torch, CANN or MemFabric.

Load the complete production class from its AST, as in the other source-only
PD tests. Only external SDK, collective and affinity dependencies are replaced.
"""

import ast
import importlib.util
import json
import logging
import os
import sys
import unittest
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import List
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
# Load the stdlib-only registry directly, without importing sglang's runtime.
spec = importlib.util.spec_from_file_location(
    "ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_register)
register_cpu_ci = ci_register.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class Mode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class Ops(Enum):
    SDMA = 0
    HOST_RDMA = 1
    DEVICE_RDMA = 2
    DEVICE_URMA = 3
    DEVICE_UBOE = 4


SOURCE = ROOT / "python/sglang/srt/disaggregation/ascend/transfer_engine.py"


def load_engine(namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    engine_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendTransferEngine"
    )
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    ]
    module = ast.Module(body=[*constants, engine_class], type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["AscendTransferEngine"]


class TestAscendTransferEngineProtocols(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ, {"ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24669"}, clear=True
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.native = Mock()
        self.native.initialize.return_value = 0
        self.native.get_rpc_port.return_value = 33101
        self.native.batch_register_memory.return_value = 0
        self.sdk = Mock(return_value=self.native)
        self.sdk.TransDataOpType = Ops
        self.affinity = Mock()
        self.tensor = object()
        self.torch = SimpleNamespace(
            zeros=Mock(return_value=self.tensor),
            empty_like=Mock(side_effect=lambda _: object()),
            distributed=SimpleNamespace(all_gather=Mock()),
        )
        self.group = SimpleNamespace(rank_in_group=2, device_group=object())
        parallel_state = ModuleType("sglang.srt.distributed.parallel_state")
        parallel_state.get_world_group = Mock(return_value=self.group)
        parallel_state.get_world_size = Mock(return_value=4)
        modules = patch.dict(sys.modules, {parallel_state.__name__: parallel_state})
        modules.start()
        self.addCleanup(modules.stop)
        self.Engine = load_engine(
            {
                "__name__": __name__,
                "MooncakeTransferEngine": object,
                "DisaggregationMode": Mode,
                "NetworkAddress": lambda host, port: SimpleNamespace(
                    to_host_port_str=lambda: f"{host}:{port}"
                ),
                "TransferEngine": self.sdk,
                "get_pd_thread_affinity": self.affinity,
                "import_error": None,
                "logger": logging.getLogger(__name__),
                "torch": self.torch,
                "json": json,
                "os": os,
                "nullcontext": nullcontext,
                "List": List,
            }
        )

    def create(self, protocol=None, mode=Mode.PREFILL):
        if protocol is None:
            os.environ.pop("ASCEND_MF_TRANSFER_PROTOCOL", None)
        else:
            os.environ["ASCEND_MF_TRANSFER_PROTOCOL"] = protocol
        return self.Engine("192.0.2.49", 5, mode)

    def assert_initialize(self, op, nic="", role="Prefill", session_port=33101):
        self.native.initialize.assert_called_once_with(
            "tcp://127.0.0.1:24669",
            f"192.0.2.49:{session_port}",
            role,
            5,
            op,
            "Decode",
            nic,
        )

    def test_urma_and_uboe_ignore_stale_host_mapping(self):
        # Even invalid Host RDMA configuration must not enter its parser.
        os.environ["ASCEND_MF_HCOM_URL"] = '{"obsolete": "invalid"}'
        for protocol, op in (
            (" DEVICE_URMA ", Ops.DEVICE_URMA),
            ("device_uboe", Ops.DEVICE_UBOE),
        ):
            with self.subTest(protocol=protocol):
                self.native.initialize.reset_mock()
                self.create(protocol)
                self.assert_initialize(op)
        self.torch.zeros.assert_not_called()
        self.torch.distributed.all_gather.assert_not_called()
        self.affinity.assert_not_called()

    def test_unset_and_blank_protocol_use_sdma(self):
        for protocol in (None, "", " \t", " SDMA "):
            with self.subTest(protocol=protocol):
                self.native.initialize.reset_mock()
                self.create(protocol)
                self.assert_initialize(Ops.SDMA)
        self.torch.distributed.all_gather.assert_not_called()
        self.affinity.assert_not_called()

    def test_unknown_or_unavailable_protocol_fails_before_sdk_initialize(self):
        old_ops = SimpleNamespace(
            SDMA=Ops.SDMA, HOST_RDMA=Ops.HOST_RDMA, DEVICE_RDMA=Ops.DEVICE_RDMA
        )
        self.sdk.TransDataOpType = old_ops
        for protocol in ("device_urma", "device_uboe", "device_urmaa", "clos"):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, protocol):
                    self.create(protocol)
        self.native.initialize.assert_not_called()
        self.torch.distributed.all_gather.assert_not_called()
        self.affinity.assert_not_called()

    def test_old_sdk_still_initializes_existing_protocols(self):
        self.sdk.TransDataOpType = SimpleNamespace(
            SDMA=Ops.SDMA, HOST_RDMA=Ops.HOST_RDMA, DEVICE_RDMA=Ops.DEVICE_RDMA
        )
        self.affinity.return_value = None
        for protocol, op in (
            ("sdma", Ops.SDMA),
            ("host_rdma", Ops.HOST_RDMA),
            ("device_rdma", Ops.DEVICE_RDMA),
        ):
            with self.subTest(protocol=protocol):
                self.native.initialize.reset_mock()
                self.create(protocol)
                self.assert_initialize(op)

    def test_host_rdma_keeps_device_mapping_stride_affinity_and_decode_role(self):
        os.environ["ASCEND_MF_HCOM_URL"] = json.dumps(
            {"2": "tcp://wrong-device:21000", "5": "tcp://192.0.2.49:21000"}
        )
        events = []

        @contextmanager
        def bind():
            events.append("bind")
            try:
                yield
            finally:
                events.append("unbind")

        self.affinity.return_value.bind_initialization_thread.side_effect = bind
        self.native.initialize.side_effect = lambda *args: events.append("init") or 0
        self.create("HOST_RDMA", mode=Mode.DECODE)
        # Runtime device 5 selects the URL; world rank 2 offsets its port by 16.
        self.assert_initialize(Ops.HOST_RDMA, "tcp://192.0.2.49:21016", "Decode")
        self.assertEqual(events, ["bind", "init", "unbind"])
        self.affinity.assert_called_once_with()
        self.torch.distributed.all_gather.assert_not_called()

    def test_device_rdma_prewarms_hccl_before_native_initialization(self):
        os.environ["ASCEND_MF_HCOM_URL"] = "legacy-nic-configuration"
        events = []
        self.torch.distributed.all_gather.side_effect = (
            lambda *args, **kwargs: events.append("all_gather")
        )
        self.native.initialize.side_effect = lambda *args: events.append("init") or 0
        self.create("device_rdma")
        self.assert_initialize(Ops.DEVICE_RDMA, "legacy-nic-configuration")
        self.assertEqual(events, ["all_gather", "init"])
        self.torch.zeros.assert_called_once_with(1, device="npu")
        args, kwargs = self.torch.distributed.all_gather.call_args
        self.assertEqual(len(args[0]), 4)
        self.assertIs(args[1], self.tensor)
        self.assertIs(kwargs["group"], self.group.device_group)
        self.affinity.assert_not_called()

    def test_sdma_preserves_existing_nic_argument(self):
        os.environ["ASCEND_MF_HCOM_URL"] = "legacy-nic-configuration"
        self.create("sdma")
        self.assert_initialize(Ops.SDMA, "legacy-nic-configuration")
        self.torch.distributed.all_gather.assert_not_called()

    def test_session_port_is_refreshed_after_sdk_initialization(self):
        self.native.get_rpc_port.side_effect = [0, 33102]
        engine = self.create("device_urma")
        self.assert_initialize(Ops.DEVICE_URMA, session_port=0)
        self.assertEqual(engine.session_id, "192.0.2.49:33102")
        self.assertEqual(self.native.get_rpc_port.call_count, 2)

    def test_sdk_initialization_failure_propagates(self):
        self.native.initialize.return_value = -7
        with self.assertRaises(RuntimeError):
            self.create("device_urma")
        self.assert_initialize(Ops.DEVICE_URMA)

    def test_memory_registration_preserves_addresses_and_rejects_error_code(self):
        engine = self.create("device_urma")
        ptrs, lengths = [2**48, 2**48 + 4096], [1024, 2048]
        engine.batch_register(ptrs, lengths)
        self.native.batch_register_memory.assert_called_once_with(ptrs, lengths)
        self.native.batch_register_memory.return_value = -13
        with self.assertRaisesRegex(RuntimeError, "-13"):
            engine.batch_register(ptrs, lengths)

    def test_memory_registration_exception_keeps_original_cause(self):
        engine = self.create("device_urma")
        original = OSError("URMA memory registration failed")
        self.native.batch_register_memory.side_effect = original
        with self.assertRaises(RuntimeError) as raised:
            engine.batch_register([2**48], [4096])
        self.assertIs(raised.exception.__cause__, original)


if __name__ == "__main__":
    unittest.main()
