# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""WeLM visual preprocessing using the checkpoint's native processor."""

import asyncio
import threading
import time
from typing import Any, List, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import get_new_expanded_mm_items, hash_feature
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.welmv4_vlm import WeLMV4VLMForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.utils import logger


class WeLMV4VLMImageProcessor(BaseMultimodalProcessor):
    models = [WeLMV4VLMForConditionalGeneration]
    gpu_image_decode = False
    # Preserve template token IDs through both OpenAI chat and TokenizerManager.
    prompt_input_type = "token_ids"
    prefer_tokenized_input = True
    _supports_native_video = False

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self._validate_checkpoint_processor(_processor, server_args)
        tokenizer = _processor.tokenizer
        self.image_token = getattr(hf_config, "image_token", "<|image_pad|>")
        self.image_token_id = self._require_token_id(tokenizer, self.image_token)
        configured_image_id = getattr(hf_config, "image_token_id", None)
        if (
            configured_image_id is not None
            and configured_image_id != self.image_token_id
        ):
            raise ValueError(
                "WeLM-v4.5-VL image_token_id does not match the checkpoint tokenizer: "
                f"config={configured_image_id}, tokenizer={self.image_token_id}. "
                "Use the tokenizer distributed with this VL checkpoint."
            )
        (
            self.vision_start_token_id,
            self.vision_end_token_id,
        ) = self._resolve_vision_boundary_tokens(tokenizer, self.image_token)
        self.video_token = None
        self.video_token_id = None
        # Actor represents video frames as timestamped IMAGE token runs. Keep
        # older checkpoints (including the MoonViT subclass) image-only unless
        # their native processor explicitly provides that normalization API.
        self._supports_native_video = callable(
            getattr(_processor, "normalize_video_item", None)
        ) and bool(getattr(_processor, "video_token", None))
        if self._supports_native_video:
            self.video_token = _processor.video_token
            self.video_token_id = self._require_token_id(tokenizer, self.video_token)
            configured_video_id = getattr(hf_config, "video_token_id", None)
            if (
                configured_video_id is not None
                and configured_video_id != self.video_token_id
            ):
                raise ValueError(
                    "WeLM video_token_id does not match the checkpoint tokenizer: "
                    f"config={configured_video_id}, tokenizer={self.video_token_id}."
                )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.IM_START_TOKEN_ID = self.vision_start_token_id
        self.IM_END_TOKEN_ID = self.vision_end_token_id
        self.IM_TOKEN_ID = self.image_token_id
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            image_token_id=self.image_token_id,
            video_token=self.video_token,
            video_token_id=self.video_token_id,
        ).build(_processor)
        # The checkpoint processor can hold mutable preprocessing state. Offload
        # CPU patchification without concurrently invoking that shared object.
        # Hold the lock inside the worker, so cancelling an HTTP coroutine does
        # not release it while that worker is still using the native processor.
        self._processor_lock = threading.Lock()

    @staticmethod
    def _validate_checkpoint_processor(processor, server_args):
        required_methods = (
            "resolve_tokenized_multimodal_inputs",
            "process_resolved_tokenized_multimodal_prompt",
        )
        missing = [
            name
            for name in required_methods
            if not callable(getattr(processor, name, None))
        ]
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None or missing:
            if tokenizer is None:
                missing.append("tokenizer")
            raise ValueError(
                "WeLM-v4.5-VL requires its checkpoint's custom HF processor "
                f"and tokenizer. Missing: {', '.join(missing)}. "
                "Provide the complete VL checkpoint, including processor config, "
                "its Python files and tokenizer files, and use --trust-remote-code. "
                "A text-only WeLM tokenizer or config.json alone is insufficient."
            )
        if not getattr(tokenizer, "chat_template", None) and getattr(
            processor, "chat_template", None
        ):
            # HF processors may own the native template, but SGLang's OpenAI
            # entrypoint renders through the processor's tokenizer.
            tokenizer.chat_template = processor.chat_template
        if not getattr(tokenizer, "chat_template", None) and not getattr(
            server_args, "chat_template", None
        ):
            raise ValueError(
                "WeLM-v4.5-VL is missing its native chat template. "
                "Include the checkpoint's "
                "chat_template.jinja (or tokenizer chat_template), or pass it "
                "template with --chat-template."
            )

    @staticmethod
    def _require_token_id(tokenizer, token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            not isinstance(token_id, int)
            or token_id < 0
            or token_id == getattr(tokenizer, "unk_token_id", None)
        ):
            raise ValueError(
                f"WeLM-v4.5-VL tokenizer is missing the required token {token!r}. "
                "Use the tokenizer distributed with the VL checkpoint."
            )
        return token_id

    @classmethod
    def _resolve_vision_boundary_tokens(cls, tokenizer, image_token):
        prefixes = (
            ("welm_", "") if image_token.startswith("<|welm_") else ("", "welm_")
        )
        for prefix in prefixes:
            try:
                return (
                    cls._require_token_id(tokenizer, f"<|{prefix}vision_start|>"),
                    cls._require_token_id(tokenizer, f"<|{prefix}vision_end|>"),
                )
            except ValueError:
                continue
        raise ValueError(
            "WeLM tokenizer is missing a matching pair of vision boundary tokens. "
            "Use the tokenizer distributed with this VL checkpoint."
        )

    @staticmethod
    def _is_token_id_input(input_text) -> bool:
        return isinstance(input_text, list) and (
            not input_text or isinstance(input_text[0], int)
        )

    @staticmethod
    def _is_precomputed_item(item: Any) -> bool:
        return isinstance(item, dict) and str(item.get("format", "")).lower() in {
            "processor_output",
            "precomputed_embedding",
        }

    def _load_image_item(self, image_item):
        return self.__class__._load_single_item(
            image_item, Modality.IMAGE, discard_alpha_channel=True
        )

    def _load_images(self, image_data):
        loaded_images = [None] * len(image_data)
        pending = []
        for idx, image_item in enumerate(image_data):
            if isinstance(image_item, (str, bytes)) or hasattr(image_item, "url"):
                pending.append((idx, image_item))
            else:
                loaded_images[idx] = image_item
        futures = [
            self.io_executor.submit(self._load_image_item, image_item)
            for _, image_item in pending
        ]
        for (idx, _), future in zip(pending, futures):
            loaded_images[idx] = future.result()
        return loaded_images

    def _processor_kwargs(self):
        processor_kwargs = {}
        if self.image_config:
            processor_kwargs["images_kwargs"] = dict(self.image_config)
        if getattr(self, "video_config", None):
            processor_kwargs["videos_kwargs"] = dict(self.video_config)
        return processor_kwargs

    def _normalize_videos(self, video_data, processor_kwargs):
        if not video_data:
            return None
        # Decode separate videos concurrently while preserving their order.
        # This runs under _processor_lock, just like image patchification.
        futures = [
            self.io_executor.submit(
                self._processor.normalize_video_item,
                video,
                processor_kwargs=processor_kwargs,
            )
            for video in video_data
        ]
        return [future.result() for future in futures]

    def _process_token_ids(self, input_ids, image_data, video_data=None):
        processor_kwargs = self._processor_kwargs()
        normalized_videos = self._normalize_videos(video_data, processor_kwargs)
        resolved = self._processor.resolve_tokenized_multimodal_inputs(
            input_ids,
            images=image_data,
            normalized_videos=normalized_videos,
            load_images=False,
        )
        resolved_images = resolved["images"]
        loaded_images = self._load_images(resolved_images) if resolved_images else None
        return self._processor.process_resolved_tokenized_multimodal_prompt(
            input_ids,
            images=loaded_images,
            video_timestamp_groups=resolved["video_timestamp_groups"],
            image_resize_specs=resolved["image_resize_specs"],
            return_tensors="pt",
            processor_kwargs=processor_kwargs,
        )

    def _process_video_prompt(self, input_text, image_data, video_data):
        # A string prompt is already rendered by the caller. Tokenize once,
        # then let the checkpoint preserve timestamps and interleaved ordering.
        # Generic video preprocessing uses a different frame/token layout.
        if not self._is_token_id_input(input_text):
            input_text = self._processor.tokenizer.encode(
                input_text, add_special_tokens=False
            )
        return self._process_token_ids(input_text, image_data, video_data)

    async def _fallback_text_processor(self, input_text, image_data):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        _, input_ids, ret = await asyncio.to_thread(
            self._run_processor_locked,
            self.process_and_combine_mm_data,
            base_output,
            self.mm_tokens,
        )
        ret["input_ids"] = input_ids.unsqueeze(0)
        return ret

    def _run_processor_locked(self, function, *args):
        with self._processor_lock:
            return function(*args)

    @staticmethod
    def _set_image_cache_hashes(mm_items):
        for item in mm_items:
            if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
                # Leave UUID generation to the scheduler's existing opt-out.
                item.hash = item.pad_value = None
                continue
            grid = torch.as_tensor(item.image_grid_thw, dtype=torch.int64, device="cpu")
            # Identical patch pixels can have different spatial layouts. Both
            # learned positions and vision RoPE depend on the grid, so it must
            # participate in the embedding-cache and RadixAttention keys.
            item.set_hash(hash_feature([grid, item.feature]))

    def _build_output_from_processor_result(self, ret):
        # CPU patchification and CPU feature transport are shared by the NPU
        # path and the standard MM path; do not use the deprecated device flag.
        for feature_name in self.FEATURE_NAMES:
            if isinstance(ret.get(feature_name), torch.Tensor):
                ret[feature_name] = ret[feature_name].to("cpu")
        input_ids = ret["input_ids"].flatten()
        mm_items = self.collect_mm_items_from_processor_output(ret)
        for mm_item in mm_items:
            if mm_item.modality != Modality.IMAGE:
                raise ValueError("WeLM-v4.5-VL currently supports image inputs only.")
            mm_item.offsets = self.get_mm_items_offset(input_ids, self.image_token_id)
        token_type_ids = ret.get("mm_token_type_ids")
        if token_type_ids is None:
            token_type_ids = ret.get("token_type_ids")
        mm_items = get_new_expanded_mm_items(mm_items)
        self._set_image_cache_hashes(mm_items)
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_start_id=self.vision_start_token_id,
            im_end_id=self.vision_end_token_id,
            im_token_id=self.image_token_id,
            video_token_id=getattr(self, "video_token_id", None),
            token_type_ids=token_type_ids,
        )

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data,
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        video_data = getattr(request_obj, "video_data", None)
        if (image_data or video_data) and getattr(request_obj, "mm_hashes", None):
            raise ValueError(
                "WeLM-v4.5-VL does not support caller-supplied mm_hashes yet: "
                "external image hashes have not been validated to include the "
                "image grid. Omit mm_hashes so the server computes a key from "
                "the processed pixels and grid."
            )
        if audio_data or getattr(request_obj, "audio_data", None):
            raise ValueError(
                "WeLM-v4.5-VL audio inputs are not supported by this implementation."
            )
        if video_data and not self._supports_native_video:
            raise ValueError(
                "This WeLM checkpoint supports image inputs only; video requires "
                "a native processor with normalize_video_item and video tokens."
            )
        if any(
            self._is_precomputed_item(item)
            for items in (image_data, video_data)
            for item in (items or [])
        ):
            raise ValueError(
                "WeLM-v4.5-VL requires raw visual inputs; precomputed embeddings "
                "and processor_output inputs are not supported."
            )
        entry_time = time.perf_counter()
        use_token_ids = self._is_token_id_input(input_text)
        if video_data:
            ret = await asyncio.to_thread(
                self._run_processor_locked,
                self._process_video_prompt,
                input_text,
                image_data,
                video_data,
            )
        elif use_token_ids:
            ret = await asyncio.to_thread(
                self._run_processor_locked,
                self._process_token_ids,
                input_text,
                image_data,
            )
        else:
            ret = await self._fallback_text_processor(input_text, image_data)
        logger.debug(
            "WeLM-v4.5-VL preprocessing rid=%s path=%s duration_ms=%.2f",
            getattr(request_obj, "rid", "anonymous_rid"),
            "video" if video_data else "token_ids" if use_token_ids else "text",
            (time.perf_counter() - entry_time) * 1000,
        )
        # Hashing full-resolution CPU pixels can be expensive; keep it off the
        # event loop just like the checkpoint's patchification above.
        return await asyncio.to_thread(self._build_output_from_processor_result, ret)
