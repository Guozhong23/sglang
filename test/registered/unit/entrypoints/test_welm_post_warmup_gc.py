"""Exercise real warmup lifecycle functions without importing the NPU server."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import requests

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/entrypoints/http_server.py"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def lifecycle():
    names = {"_freeze_gc_after_server_warmup", "_wait_and_warmup"}
    nodes = [
        n for n in ast.parse(SOURCE.read_text()).body if getattr(n, "name", "") in names
    ]
    assert {n.name for n in nodes} == names
    tree = ast.Module(
        body=ast.parse("from __future__ import annotations").body + nodes,
        type_ignores=[],
    )
    state = NS(tokenizer_manager=NS(server_status="starting"))
    scope = dict(
        requests=NS(post=Mock(return_value=Mock()), exceptions=requests.exceptions),
        logger=Mock(),
        _execute_server_warmup=Mock(return_value=True),
        _wait_weights_ready=Mock(),
        _global_state=state,
        ServerStatus=NS(Up="up", UnHealthy="unhealthy"),
    )
    exec(compile(tree, str(SOURCE), "exec"), scope)
    args = NS(
        api_key=None,
        admin_api_key=None,
        url=lambda: "https://127.0.0.1:7788",
        ssl_verify=lambda: "/test/ca.pem",
        checkpoint_engine_wait_weights_before_ready=False,
        is_ep_scale_joiner=False,
        skip_server_warmup=False,
        delete_ckpt_after_loading=False,
        debug_tensor_dump_input_file=None,
    )
    return NS(**scope), args


@pytest.mark.parametrize(
    "api_key,admin_key,headers",
    [
        (None, None, {}),
        ("api", None, {"Authorization": "Bearer api"}),
        ("api", "admin", {"Authorization": "Bearer admin"}),
    ],
)
def test_gc_request_uses_admin_auth_and_server_tls(
    lifecycle, api_key, admin_key, headers
):
    server, args = lifecycle
    args.api_key, args.admin_api_key = api_key, admin_key
    server._freeze_gc_after_server_warmup(args)
    server.requests.post.assert_called_once_with(
        args.url() + "/freeze_gc", headers=headers, timeout=10, verify=args.ssl_verify()
    )
    server.requests.post.return_value.raise_for_status.assert_called_once_with()


@pytest.mark.parametrize("skip", [False, True])
def test_freeze_runs_after_npu_finalization_and_before_launch_callback(lifecycle, skip):
    server, args = lifecycle
    args.skip_server_warmup = skip
    events = []

    def warmup(_):
        assert server._global_state.tokenizer_manager.server_status == "starting"
        events.append("warmup")
        return True

    def finalize():
        assert server._global_state.tokenizer_manager.server_status == "starting"
        events.append("affinity")

    def post(*_args, **_kwargs):
        assert server._global_state.tokenizer_manager.server_status == "up"
        events.append("freeze")
        return Mock()

    server.requests.post.side_effect = post
    server._wait_and_warmup(
        args,
        execute_warmup_func=warmup,
        finalize_npu_affinity=finalize,
        launch_callback=lambda: events.append("callback"),
    )
    assert events == ([] if skip else ["warmup"]) + ["affinity", "freeze", "callback"]


@pytest.mark.parametrize("failure", ["warmup", "affinity"])
def test_failed_startup_does_not_freeze_or_report_ready(lifecycle, failure):
    server, args = lifecycle
    warmup = Mock(return_value=failure != "warmup")
    finalize = Mock(
        side_effect=RuntimeError("binding failed") if failure == "affinity" else None
    )
    callback = Mock()
    server._wait_and_warmup(args, callback, warmup, finalize)
    server.requests.post.assert_not_called()
    callback.assert_not_called()
    assert server._global_state.tokenizer_manager.server_status != "up"
    if failure == "warmup":
        finalize.assert_not_called()


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_freeze_request_failure_does_not_undo_successful_startup(lifecycle, failure):
    server, args = lifecycle
    if failure == "timeout":
        server.requests.post.side_effect = requests.exceptions.Timeout(
            "freeze timed out"
        )
    else:
        server.requests.post.return_value.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("503")
        )
    callback = Mock()
    server._wait_and_warmup(args, launch_callback=callback)
    callback.assert_called_once_with()
    server.logger.warning.assert_called_once()
    assert server._global_state.tokenizer_manager.server_status == "up"
