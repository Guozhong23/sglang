"""Exercise the production adapter without importing workers or loading weights.

The class body is loaded unchanged; runtime-only base/model imports are omitted.
Media loading and native HF preprocessing are explicit test doubles, so these
tests validate adapter contracts, not real checkpoint preprocessing equivalence.
"""

import ast
import asyncio
import concurrent.futures
import hashlib
import logging
import runpy
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Union
from unittest.mock import AsyncMock, Mock, patch

import torch

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cache_hash_helpers():
    """Use the production SHA-based feature hash and hash-to-pad derivation."""
    srt = Path(__file__).resolve().parents[4] / "python/sglang/srt"
    namespace = dict(
        torch=torch,
        hashlib=hashlib,
        flatten_nested_list=lambda items: items,
        ShmPointerMMData=type("ShmPointerMMData", (), {}),
    )
    tree = ast.parse((srt / "managers/mm_utils.py").read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"data_hash", "tensor_hash", "hash_feature"}
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<MM hash>", "exec"), namespace
    )
    tree = ast.parse((srt / "managers/schedule_batch.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MM_PAD_SHIFT_VALUE"
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "_compute_pad_value":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "MultimodalDataItem":
            nodes.extend(
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef) and method.name == "set_hash"
            )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "<MM pad>", "exec"), namespace
    )
    return namespace


HASH_HELPERS = _cache_hash_helpers()


def _image_item(grid, pixels=None):
    item = SimpleNamespace(
        feature=torch.zeros(8, 3) if pixels is None else pixels,
        image_grid_thw=torch.tensor([grid]),
        modality="image",
        hash=None,
        pad_value=None,
        offsets=None,
    )
    item.set_hash = lambda value: HASH_HELPERS["set_hash"](item, value)
    return item


def _processor_class():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/multimodal/processors/welmv4_vlm.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = dict(
        asyncio=asyncio,
        threading=threading,
        time=time,
        torch=torch,
        Any=Any,
        List=List,
        Union=Union,
        BaseMultimodalProcessor=object,
        WeLMV4VLMForConditionalGeneration=object,
        logger=logging.getLogger(__name__),
        envs=SimpleNamespace(
            SGLANG_MM_SKIP_COMPUTE_HASH=SimpleNamespace(get=lambda: False)
        ),
        hash_feature=HASH_HELPERS["hash_feature"],
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["WeLMV4VLMImageProcessor"]


WeLMV4VLMImageProcessor = _processor_class()


class TestWeLMV4VLMProcessor(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Some socket-restricted sandboxes suppress asyncio's cross-thread
        # socketpair wakeups. A timer keeps executor completions observable.
        loop = asyncio.get_running_loop()

        def tick():
            self._tick_handle = loop.call_later(0.01, tick)

        self._tick_handle = loop.call_soon(tick)

    def make_processor(self):
        processor = object.__new__(WeLMV4VLMImageProcessor)
        processor._processor_lock = threading.Lock()
        processor.image_config = {"max_pixels": 4096}
        processor._processor = SimpleNamespace(
            resolve_tokenized_multimodal_inputs=Mock(
                return_value={
                    "images": ["first", "second"],
                    "video_timestamp_groups": [],
                    "image_resize_specs": [None, None],
                }
            ),
            process_resolved_tokenized_multimodal_prompt=Mock(
                return_value={"input_ids": torch.tensor([[1, 154752, 2]])}
            ),
        )
        processor._build_output_from_processor_result = Mock(side_effect=lambda x: x)
        processor._load_images = Mock(return_value=["decoded-first", "decoded-second"])
        return processor

    async def test_native_processor_gets_original_ids_and_image_order(self):
        processor = self.make_processor()
        ids = [1, 154752, 2, 154752, 3]
        images = ["first", "second"]
        result = await processor.process_mm_data_async(
            images, None, ids, SimpleNamespace(rid="images")
        )
        resolve = processor._processor.resolve_tokenized_multimodal_inputs
        resolve.assert_called_once_with(
            ids, images=images, normalized_videos=None, load_images=False
        )
        process = processor._processor.process_resolved_tokenized_multimodal_prompt
        call = process.call_args
        self.assertIs(call.args[0], ids)
        self.assertEqual(call.kwargs["images"], ["decoded-first", "decoded-second"])
        self.assertEqual(call.kwargs["image_resize_specs"], [None, None])
        self.assertEqual(
            call.kwargs["processor_kwargs"], {"images_kwargs": {"max_pixels": 4096}}
        )
        self.assertEqual(result["input_ids"].tolist(), [[1, 154752, 2]])

    async def test_cpu_processing_does_not_block_event_loop(self):
        processor = self.make_processor()
        entered = threading.Event()
        released = threading.Event()
        loop_thread = threading.get_ident()

        def process(ids, images):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            entered.set()
            if not released.wait(timeout=2):
                raise AssertionError("event loop did not release CPU processing")
            return {"input_ids": ids}

        processor._process_token_ids = process
        task = asyncio.create_task(
            processor.process_mm_data_async([], None, [1], SimpleNamespace())
        )
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            self.assertFalse(task.done())
        finally:
            released.set()
        self.assertEqual(await task, {"input_ids": [1]})

    async def test_output_hashing_runs_off_event_loop(self):
        processor = self.make_processor()
        loop_thread = threading.get_ident()

        def build_output(ret):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            return ret

        processor._build_output_from_processor_result = build_output
        await processor.process_mm_data_async([], None, [1], SimpleNamespace())

    async def test_shared_native_processor_is_serialized(self):
        processor = self.make_processor()
        entered = threading.Event()
        released = threading.Event()
        second_started = threading.Event()

        def process(ids, images):
            if ids == [1]:
                entered.set()
                if not released.wait(timeout=2):
                    raise AssertionError("first request was not released")
            else:
                second_started.set()
            return {"input_ids": ids}

        processor._process_token_ids = process
        first = asyncio.create_task(
            processor.process_mm_data_async([], None, [1], SimpleNamespace())
        )
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            second = asyncio.create_task(
                processor.process_mm_data_async([], None, [2], SimpleNamespace())
            )
            await asyncio.sleep(0)
            self.assertFalse(second_started.is_set())
        finally:
            released.set()
        await first
        if second is not None:
            self.assertEqual(await second, {"input_ids": [2]})

    async def test_text_fallback_awaits_media_loading(self):
        processor = self.make_processor()
        base_output = object()
        processor.mm_tokens = object()
        processor.load_mm_data = AsyncMock(return_value=base_output)
        processor.process_and_combine_mm_data = Mock(
            return_value=([], torch.tensor([7, 154752, 8]), {})
        )
        result = await processor.process_mm_data_async(
            ["first"], None, "native prompt", SimpleNamespace()
        )
        processor.load_mm_data.assert_awaited_once_with(
            prompt="native prompt",
            image_data=["first"],
            multimodal_tokens=processor.mm_tokens,
        )
        processor.process_and_combine_mm_data.assert_called_once_with(
            base_output, processor.mm_tokens
        )
        self.assertEqual(result["input_ids"].tolist(), [[7, 154752, 8]])

    async def test_cancelled_request_does_not_release_active_processor(self):
        processor = self.make_processor()
        entered = threading.Event()
        released = threading.Event()
        second_started = threading.Event()

        def process(ids, images):
            if ids == [1]:
                entered.set()
                if not released.wait(timeout=2):
                    raise AssertionError("cancelled worker was not released")
            else:
                second_started.set()
            return {"input_ids": ids}

        processor._process_token_ids = process
        first = asyncio.create_task(
            processor.process_mm_data_async([], None, [1], SimpleNamespace())
        )
        second = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                processor.process_mm_data_async([], None, [2], SimpleNamespace())
            )
            await asyncio.sleep(0)
            self.assertFalse(second_started.is_set())
        finally:
            released.set()
        if second is not None:
            self.assertEqual(await second, {"input_ids": [2]})

    def test_same_pixels_different_grids_get_distinct_cache_keys(self):
        wide = _image_item([1, 2, 4])
        tall = _image_item([1, 4, 2])
        self.assertEqual(
            HASH_HELPERS["hash_feature"]([wide.feature]),
            HASH_HELPERS["hash_feature"]([tall.feature]),
        )
        WeLMV4VLMImageProcessor._set_image_cache_hashes([wide, tall])
        self.assertNotEqual(wide.hash, tall.hash)
        self.assertNotEqual(wide.pad_value, tall.pad_value)
        for item in (wide, tall):
            self.assertEqual(
                item.pad_value, HASH_HELPERS["_compute_pad_value"](item.hash)
            )

    def test_identical_pixels_and_grid_reuse_cache_key(self):
        first = _image_item([1, 2, 4])
        repeated = _image_item([1, 2, 4], first.feature.clone())
        repeated.image_grid_thw = repeated.image_grid_thw.to(torch.int32)
        WeLMV4VLMImageProcessor._set_image_cache_hashes([first, repeated])
        self.assertEqual(first.hash, repeated.hash)
        self.assertEqual(first.pad_value, repeated.pad_value)
        repeated.feature[0, 0] = 1
        WeLMV4VLMImageProcessor._set_image_cache_hashes([repeated])
        self.assertNotEqual(first.hash, repeated.hash)

    def test_hash_opt_out_preserves_scheduler_uuid_path(self):
        item = _image_item([1, 2, 4])
        item.hash, item.pad_value = 1, 2
        namespace = WeLMV4VLMImageProcessor._set_image_cache_hashes.__globals__
        flag = namespace["envs"].SGLANG_MM_SKIP_COMPUTE_HASH
        with patch.object(flag, "get", return_value=True):
            WeLMV4VLMImageProcessor._set_image_cache_hashes([item])
        self.assertIsNone(item.hash)
        self.assertIsNone(item.pad_value)

    def test_output_hashes_each_expanded_image(self):
        processor = object.__new__(WeLMV4VLMImageProcessor)
        processor.FEATURE_NAMES = ["pixel_values"]
        processor.image_token_id = 154752
        processor.vision_start_token_id = 7
        processor.vision_end_token_id = 8
        bundled = _image_item([1, 2, 4])
        expanded = [_image_item([1, 2, 4]), _image_item([1, 4, 2])]
        processor.collect_mm_items_from_processor_output = Mock(return_value=[bundled])
        processor.get_mm_items_offset = Mock(return_value=[(1, 2), (4, 5)])
        expand = Mock(return_value=expanded)
        namespace = processor._build_output_from_processor_result.__globals__
        with patch.dict(
            namespace,
            {
                "Modality": SimpleNamespace(IMAGE="image"),
                "get_new_expanded_mm_items": expand,
                "MultimodalProcessorOutput": SimpleNamespace,
            },
        ):
            result = processor._build_output_from_processor_result(
                {
                    "input_ids": torch.tensor([[1, 154752, 154752, 2, 154752, 154752]]),
                    "pixel_values": torch.zeros(16, 3),
                }
            )
        expand.assert_called_once_with([bundled])
        self.assertIsNone(bundled.hash)
        self.assertIs(result.mm_items, expanded)
        self.assertNotEqual(expanded[0].hash, expanded[1].hash)
        self.assertNotEqual(expanded[0].pad_value, expanded[1].pad_value)

    async def test_text_without_images_does_not_reject_external_hash_field(self):
        processor = self.make_processor()
        await processor.process_mm_data_async(
            [], None, [1], SimpleNamespace(mm_hashes=["unused"])
        )
        processor._processor.resolve_tokenized_multimodal_inputs.assert_called_once()

    async def test_unsupported_media_fail_before_native_processing(self):
        cases = (
            (["image"], None, SimpleNamespace(mm_hashes=["00"]), "mm_hashes"),
            ([], None, SimpleNamespace(video_data=["video.mp4"]), "video and audio"),
            ([], ["audio.wav"], SimpleNamespace(), "video and audio"),
            (
                [{"format": "precomputed_embedding"}],
                None,
                SimpleNamespace(),
                "precomputed embeddings",
            ),
            (
                [{"format": "processor_output"}],
                None,
                SimpleNamespace(),
                "processor_output",
            ),
        )
        for images, audio, request, message in cases:
            with self.subTest(message=message):
                processor = self.make_processor()
                with self.assertRaisesRegex(ValueError, message):
                    await processor.process_mm_data_async(images, audio, [1], request)
                resolve = processor._processor.resolve_tokenized_multimodal_inputs
                resolve.assert_not_called()

    def test_image_loading_preserves_order_and_preloaded_images(self):
        processor = object.__new__(WeLMV4VLMImageProcessor)
        preloaded = object()
        processor._load_image_item = Mock(side_effect=lambda value: "decoded-" + value)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            processor.io_executor = executor
            self.assertEqual(
                processor._load_images(["first", preloaded, "second"]),
                ["decoded-first", preloaded, "decoded-second"],
            )

    def test_missing_custom_processor_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "resolve_tokenized_multimodal_inputs"):
            WeLMV4VLMImageProcessor._validate_checkpoint_processor(
                SimpleNamespace(tokenizer=object()), SimpleNamespace()
            )

    def test_missing_template_requires_native_checkpoint_template(self):
        processor = self.make_processor()._processor
        processor.tokenizer = SimpleNamespace(chat_template=None)
        with self.assertRaisesRegex(ValueError, "chat_template.jinja"):
            WeLMV4VLMImageProcessor._validate_checkpoint_processor(
                processor, SimpleNamespace(chat_template=None)
            )
        WeLMV4VLMImageProcessor._validate_checkpoint_processor(
            processor, SimpleNamespace(chat_template="/model/chat_template.jinja")
        )

    def test_missing_special_token_is_not_silently_used_as_unknown(self):
        tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: 0, unk_token_id=0
        )
        with self.assertRaisesRegex(ValueError, "required token"):
            WeLMV4VLMImageProcessor._require_token_id(tokenizer, "<|image_pad|>")

    def test_processor_native_template_is_used_by_chat_tokenizer(self):
        processor = self.make_processor()._processor
        processor.tokenizer = SimpleNamespace(chat_template=None)
        processor.chat_template = "native VL template"
        WeLMV4VLMImageProcessor._validate_checkpoint_processor(
            processor, SimpleNamespace(chat_template=None)
        )
        self.assertEqual(processor.tokenizer.chat_template, "native VL template")

    def test_configured_image_id_must_match_tokenizer(self):
        processor = self.make_processor()._processor
        processor.tokenizer = SimpleNamespace(
            chat_template="native template",
            convert_tokens_to_ids=lambda token: 154752,
            unk_token_id=None,
        )
        with self.assertRaisesRegex(ValueError, "image_token_id does not match"):
            WeLMV4VLMImageProcessor(
                SimpleNamespace(image_token_id=10), SimpleNamespace(), processor
            )


if __name__ == "__main__":
    unittest.main()
