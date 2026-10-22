"""CPU regression tests for request conversion without accelerator dependencies.

Execute the production handle_request method in isolation. Request containers and
response handlers are stand-ins; these tests verify scheduling and handoff, not
tokenizer output or end-to-end PD transport.
"""

import ast
import asyncio
import logging
import runpy
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[5]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class GenerateRequest(SimpleNamespace):
    pass


class EmbeddingRequest(SimpleNamespace):
    pass


class EncodingError(Exception):
    pass


def load_handle_request():
    path = ROOT / "python/sglang/srt/entrypoints/openai/serving_base.py"
    tree = ast.parse(path.read_text())
    owner = next(
        node for node in tree.body if getattr(node, "name", "") == "OpenAIServingBase"
    )
    method = next(
        node for node in owner.body if getattr(node, "name", "") == "handle_request"
    )
    tree.body = ast.parse("from __future__ import annotations").body + [method]
    scope = dict(
        asyncio=asyncio,
        monotonic_time=time.monotonic,
        GenerateReqInput=GenerateRequest,
        EmbeddingReqInput=EmbeddingRequest,
        HTTPException=HTTPException,
        DS32EncodingError=EncodingError,
        logger=logging.getLogger(__name__),
    )
    exec(compile(tree, str(path), "exec"), scope)
    return scope["handle_request"]


class Endpoint:
    handle_request = load_handle_request()

    def __init__(self, convert):
        self._convert_to_internal_request = convert
        self._validate_request = Mock(return_value=None)
        self.tokenizer_manager = SimpleNamespace(
            request_logger=SimpleNamespace(log_requests=False)
        )
        self.dispatches = []

    def create_error_response(
        self, message, err_type="BadRequestError", status_code=400
    ):
        return SimpleNamespace(message=message, type=err_type, status_code=status_code)

    async def _handle_streaming_request(self, adapted, processed, raw):
        self.dispatches.append(("stream", threading.get_ident()))
        return adapted, processed, raw

    async def _handle_non_streaming_request(self, adapted, processed, raw):
        self.dispatches.append(("nonstream", threading.get_ident()))
        return adapted, processed, raw


class TestRequestConversion(unittest.IsolatedAsyncioTestCase):
    async def test_pending_first_token_runs_during_conversion(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                loop = asyncio.get_running_loop()
                loop_thread = threading.get_ident()
                started = asyncio.Event()
                release = threading.Event()
                finished = threading.Event()
                worker_threads = []
                request = SimpleNamespace(stream=stream)
                raw_request = object()
                adapted = GenerateRequest(
                    image_data=["data:image/png;base64,fixture"],
                    bootstrap_host="prefill",
                    bootstrap_port=8998,
                    bootstrap_room=1234,
                    routed_dp_rank=1,
                    disagg_prefill_dp_rank=0,
                )

                def convert(received, raw):
                    self.assertIs(received, request)
                    self.assertIs(raw, raw_request)
                    worker_threads.append(threading.get_ident())
                    loop.call_soon_threadsafe(started.set)
                    if not release.wait(timeout=5):
                        raise TimeoutError("Conversion blocked the event loop")
                    finished.set()
                    return adapted, received

                endpoint = Endpoint(convert)
                received_after = time.monotonic()
                task = asyncio.create_task(
                    endpoint.handle_request(request, raw_request)
                )
                try:
                    await asyncio.wait_for(started.wait(), timeout=10)
                    # Model an already-running request's first-token callback. It
                    # must execute before the new request's conversion finishes.
                    first_token = loop.create_future()
                    loop.call_soon(first_token.set_result, "first token")
                    self.assertEqual(await first_token, "first token")
                    self.assertFalse(task.done())
                    self.assertFalse(finished.is_set())
                    self.assertNotEqual(worker_threads, [loop_thread])
                finally:
                    released_at = time.monotonic()
                    release.set()
                    result = await task

                self.assertIs(result[0], adapted)
                self.assertIs(result[1], request)
                self.assertIs(result[2], raw_request)
                self.assertGreaterEqual(adapted.received_time, received_after)
                self.assertLessEqual(adapted.received_time, released_at)
                self.assertEqual(
                    endpoint.dispatches,
                    [("stream" if stream else "nonstream", loop_thread)],
                )

    async def test_validation_precedes_conversion(self):
        convert = Mock()
        endpoint = Endpoint(convert)
        endpoint._validate_request.return_value = "Invalid prompt"
        response = await endpoint.handle_request(SimpleNamespace(stream=False), None)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.message, "Invalid prompt")
        convert.assert_not_called()
        self.assertEqual(endpoint.dispatches, [])

    async def test_worker_errors_preserve_http_status(self):
        cases = [
            (HTTPException(status_code=409, detail="Conflict"), 409, "409", "Conflict"),
            (ValueError("Invalid prompt"), 400, "BadRequest", "Invalid prompt"),
            (EncodingError("Invalid encoding"), 400, "BadRequest", "Invalid encoding"),
            (
                RuntimeError("Unexpected"),
                500,
                "InternalServerError",
                "Internal server error: Unexpected",
            ),
        ]
        for error, status, error_type, message in cases:
            with self.subTest(error=type(error).__name__):
                endpoint = Endpoint(Mock(side_effect=error))
                log_context = (
                    self.assertLogs(__name__, level="DEBUG")
                    if status == 500
                    else nullcontext()
                )
                with log_context:
                    response = await endpoint.handle_request(
                        SimpleNamespace(stream=False), None
                    )
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.type, error_type)
                self.assertEqual(response.message, message)
                self.assertEqual(endpoint.dispatches, [])

    async def test_cancellation_does_not_dispatch_generation(self):
        loop = asyncio.get_running_loop()
        started, finished = asyncio.Event(), asyncio.Event()
        release = threading.Event()

        def convert(request, raw):
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("Conversion blocked the event loop")
                return GenerateRequest(), request
            finally:
                loop.call_soon_threadsafe(finished.set)

        endpoint = Endpoint(convert)
        task = asyncio.create_task(
            endpoint.handle_request(SimpleNamespace(stream=True), None)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            # Cancelling the await does not terminate a running executor thread.
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=10)
        self.assertEqual(endpoint.dispatches, [])

    async def test_embedding_timing_is_preserved(self):
        request = SimpleNamespace(stream=False)
        adapted = EmbeddingRequest()
        endpoint = Endpoint(lambda request, raw: (adapted, request))
        before = time.monotonic()
        result = await endpoint.handle_request(request, None)
        self.assertIs(result[0], adapted)
        self.assertGreaterEqual(adapted.received_time, before)
        self.assertLessEqual(adapted.received_time, time.monotonic())


if __name__ == "__main__":
    unittest.main()
