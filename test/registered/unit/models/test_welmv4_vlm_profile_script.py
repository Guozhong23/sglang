"""Run the real profiling shell client against an isolated CPU HTTP server.

The fake server implements the public HTTP protocol only. No SGLang process,
checkpoint, profiler, or accelerator is loaded, and every request stays on a
random loopback port.
"""

import base64
import hashlib
import json
import os
import runpy
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SCRIPT = ROOT / "examples/runtime/welm_vl/profile_single_image.sh"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGPgEpEDAABo"
    "AD1UCKP3AAAAAElFTkSuQmCC"
)
BASE_MODEL = "welmv45-vl-test-base"
PROFILE_ROOT = "/tmp/server-profile"
MODEL_REPLY = {
    "object": "list",
    "data": [
        {"id": BASE_MODEL, "object": "model"},
        {"id": "another-model", "object": "model"},
    ],
}
CHAT_REPLY = {
    "id": "chatcmpl-profile-test",
    "object": "chat.completion",
    "model": BASE_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "一张测试图片。"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17},
}
MODELS = ("GET", "/v1/models")
FLUSH = ("POST", "/flush_cache")
START = ("POST", "/start_profile")
CHAT = ("POST", "/v1/chat/completions")
STOP = ("POST", "/stop_profile")


@pytest.fixture
def invoke_profile(tmp_path):
    servers = []
    invocations = []

    def invoke(*, fail=None, fail_status=500, chat_body=None, **environment):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _respond(self):
                parsed = urlsplit(self.path)
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                call = SimpleNamespace(
                    endpoint=(self.command, parsed.path),
                    query=parse_qs(parsed.query),
                    payload=json.loads(body) if body else None,
                    content_type=self.headers.get("Content-Type"),
                )
                calls.append(call)
                status = 200
                content_type = "text/plain; charset=utf-8"
                if call.endpoint == fail:
                    status = fail_status
                    reply = "Injected failure at " + parsed.path
                elif call.endpoint == MODELS:
                    reply = json.dumps(MODEL_REPLY)
                    content_type = "application/json"
                elif call.endpoint == CHAT:
                    reply = (
                        json.dumps(CHAT_REPLY, ensure_ascii=False)
                        if chat_body is None
                        else chat_body
                    )
                    content_type = "application/json"
                elif call.endpoint in (FLUSH, START, STOP):
                    # Actual SGLang control routes return plain text on success.
                    reply = "Control request completed."
                else:
                    status = 404
                    reply = "Unexpected endpoint: " + self.path
                encoded = reply.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            do_GET = _respond
            do_POST = _respond

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        workdir = tmp_path / f"invocation-{len(invocations)}"
        workdir.mkdir()
        image = workdir / "test image.png"
        image.write_bytes(PNG)
        client_dir = workdir / "client artifacts"
        url = f"http://127.0.0.1:{server.server_port}"
        env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHON_BIN": sys.executable,
            "IMAGE_PATH": str(image),
            "SGLANG_URL": url,
            "PROFILE_ROOT": PROFILE_ROOT,
            "CLIENT_OUTPUT_DIR": str(client_dir),
            "MAX_TOKENS": "37",
            "REQUEST_TIMEOUT": "5",
            "PROFILE_TIMEOUT": "5",
            **environment,
        }
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
        invocation = SimpleNamespace(
            result=result,
            calls=calls,
            endpoints=[call.endpoint for call in calls],
            client_dir=client_dir,
            image=image,
        )
        invocations.append(invocation)
        return invocation

    yield invoke
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _payload(invocation, endpoint):
    matching = [call for call in invocation.calls if call.endpoint == endpoint]
    assert len(matching) == 1, invocation.endpoints
    return matching[0].payload


def test_profile_single_request_uses_control_protocol_and_data_url(invoke_profile):
    invocation = invoke_profile()
    result = invocation.result
    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation.endpoints == [MODELS, FLUSH, START, CHAT, STOP]
    assert invocation.calls[1].query == {"timeout": ["30"]}

    profile = _payload(invocation, START)
    assert profile["activities"] == ["CPU", "GPU"]
    assert profile["with_stack"] is False
    assert profile["record_shapes"] is False
    assert profile["merge_profiles"] is False
    assert "num_steps" not in profile
    server_dir = Path(profile["output_dir"])
    assert server_dir.is_relative_to(PROFILE_ROOT)
    assert not server_dir.is_relative_to(invocation.client_dir)

    chat = _payload(invocation, CHAT)
    assert chat["model"] == BASE_MODEL
    assert chat["temperature"] == 0
    assert chat["stream"] is False
    assert chat["max_tokens"] == 37
    assert len(chat["messages"]) == 1
    assert chat["messages"][0]["role"] == "user"
    content = chat["messages"][0]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    text_parts = [part for part in content if part["type"] == "text"]
    assert len(image_parts) == len(text_parts) == 1
    assert text_parts[0]["text"].strip()
    data_url = image_parts[0]["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == PNG
    assert str(invocation.image) not in json.dumps(chat)
    assert invocation.client_dir.is_dir()
    artifact = invocation.client_dir
    assert json.loads((artifact / "request.json").read_text()) == chat
    assert json.loads((artifact / "profile_request.json").read_text()) == profile
    assert json.loads((artifact / "response.json").read_text()) == CHAT_REPLY
    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["model"] == BASE_MODEL
    assert metadata["server_profile_dir"] == str(server_dir)
    assert metadata["image_sha256"] == hashlib.sha256(PNG).hexdigest()
    assert metadata["inference_requests"] == 1
    assert metadata["flush_kv_cache"] is True
    summary = json.loads((artifact / "summary.json").read_text())
    assert summary["response_id"] == CHAT_REPLY["id"]
    assert summary["finish_reason"] == "stop"
    assert summary["usage"] == CHAT_REPLY["usage"]
    assert summary["server_profile_dir"] == str(server_dir)
    for name in ("flush_cache.txt", "start_profile.txt", "stop_profile.txt"):
        assert (artifact / name).read_text() == "Control request completed."
        assert (artifact / (name + ".http_status")).read_text().strip() == "200"
    for endpoint in (START, CHAT):
        call = next(call for call in invocation.calls if call.endpoint == endpoint)
        assert call.content_type == "application/json"
    assert "One request captured" in result.stdout


def _assert_failed_artifacts(invocation, failed_body, status_code=500):
    assert invocation.result.returncode != 0
    assert not (invocation.client_dir / "summary.json").exists()
    assert "One request captured" not in invocation.result.stdout
    assert "Injected failure" in (invocation.client_dir / failed_body).read_text()
    status = invocation.client_dir / (failed_body + ".http_status")
    assert status.read_text().strip() == str(status_code)


def test_inference_error_still_stops_profile_and_fails(invoke_profile):
    invocation = invoke_profile(fail=CHAT)
    _assert_failed_artifacts(invocation, "response.json")
    assert invocation.endpoints == [MODELS, FLUSH, START, CHAT, STOP]


def test_invalid_json_response_still_stops_before_validation_fails(invoke_profile):
    invocation = invoke_profile(chat_body="not a JSON response")
    assert invocation.result.returncode != 0
    assert invocation.endpoints == [MODELS, FLUSH, START, CHAT, STOP]
    artifact = invocation.client_dir
    assert (artifact / "response.json").read_text() == "not a JSON response"
    assert (artifact / "response.json.http_status").read_text().strip() == "200"
    assert (artifact / "stop_profile.txt.http_status").read_text().strip() == "200"
    assert not (artifact / "summary.json").exists()
    assert "One request captured" not in invocation.result.stdout


def test_failed_flush_never_starts_profiler_or_inference(invoke_profile):
    invocation = invoke_profile(fail=FLUSH)
    _assert_failed_artifacts(invocation, "flush_cache.txt")
    assert invocation.endpoints == [MODELS, FLUSH]


def test_failed_start_is_not_retried_and_never_sends_inference(invoke_profile):
    invocation = invoke_profile(fail=START)
    _assert_failed_artifacts(invocation, "start_profile.txt")
    assert invocation.endpoints == [MODELS, FLUSH, START]


def test_redirected_start_is_rejected_before_inference(invoke_profile):
    invocation = invoke_profile(fail=START, fail_status=302)
    _assert_failed_artifacts(invocation, "start_profile.txt", status_code=302)
    assert invocation.endpoints == [MODELS, FLUSH, START]


def test_failed_stop_is_not_retried_and_cannot_report_success(invoke_profile):
    invocation = invoke_profile(fail=STOP)
    _assert_failed_artifacts(invocation, "stop_profile.txt")
    assert invocation.endpoints == [MODELS, FLUSH, START, CHAT, STOP]


def test_flush_cache_can_be_explicitly_disabled(invoke_profile):
    invocation = invoke_profile(FLUSH_CACHE="0")
    result = invocation.result
    assert result.returncode == 0, result.stdout + result.stderr
    assert invocation.endpoints == [MODELS, START, CHAT, STOP]
    assert not (invocation.client_dir / "flush_cache.txt").exists()
    metadata = json.loads((invocation.client_dir / "metadata.json").read_text())
    assert metadata["flush_kv_cache"] is False
