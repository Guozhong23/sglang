# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""MoonViT image patches with token-ID-preserving WeLM placeholder expansion."""

import asyncio

import torch

from sglang.srt.models.welm_moonvit_vl import WeLMMoonViTVLForConditionalGeneration
from sglang.srt.multimodal.processors.welmv4_vlm import WeLMV4VLMImageProcessor


def expand_moonvit_image_tokens(input_ids, image_token_id, grids, merge_kernel):
    kh, kw = merge_kernel
    if sum(token == image_token_id for token in input_ids) != len(grids):
        raise ValueError(
            "MoonViT requires exactly one image placeholder per supplied image"
        )
    counts = []
    for t, h, w in grids:
        if t != 1 or h <= 0 or w <= 0 or h % kh or w % kw:
            raise ValueError("Invalid MoonViT image grid or merge size")
        counts.append((h // kh) * (w // kw))
    expanded = []
    image_index = 0
    for token in input_ids:
        if token == image_token_id:
            expanded.extend([image_token_id] * counts[image_index])
            image_index += 1
        else:
            expanded.append(token)
    return expanded


class WeLMMoonViTVLImageProcessor(WeLMV4VLMImageProcessor):
    models = [WeLMMoonViTVLForConditionalGeneration]

    @staticmethod
    def _validate_checkpoint_processor(processor, server_args):
        tokenizer = getattr(processor, "tokenizer", None)
        image_processor = getattr(processor, "image_processor", None)
        if tokenizer is None or not callable(image_processor):
            raise ValueError(
                "WeLM MoonViT requires the complete checkpoint's processor and "
                "tokenizer, including preprocessor_config.json and Python files; "
                "use --trust-remote-code."
            )
        if getattr(image_processor, "patch_size", None) != 14:
            raise ValueError("WeLM MoonViT requires its patch14 image processor")
        if not getattr(tokenizer, "chat_template", None):
            tokenizer.chat_template = getattr(processor, "chat_template", None)
        if not tokenizer.chat_template and not getattr(
            server_args, "chat_template", None
        ):
            raise ValueError("WeLM MoonViT requires the checkpoint's VL chat template")

    def _process_token_ids(self, input_ids, image_data):
        images = self._load_images(image_data or [])
        if not images:
            ids = expand_moonvit_image_tokens(
                input_ids, self.image_token_id, [], (2, 2)
            )
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        # Call only the image processor. The checkpoint's combined __call__
        # decodes/retokenizes text and silently substitutes grey images on I/O
        # failure. SGLang has already loaded valid images and rendered tokens.
        image_processor = self._processor.image_processor
        result = image_processor(
            images=images, return_tensors="pt", **self.image_config
        )
        grid_hw = torch.as_tensor(result["image_grid_hws"], dtype=torch.long)
        if grid_hw.ndim != 2 or grid_hw.shape != (len(images), 2):
            raise ValueError(
                "MoonViT image processor returned an invalid image_grid_hws"
            )
        grid_thw = torch.cat((torch.ones_like(grid_hw[:, :1]), grid_hw), dim=1)
        merge = tuple(image_processor.merge_kernel_size)
        ids = expand_moonvit_image_tokens(
            input_ids, self.image_token_id, grid_thw.tolist(), merge
        )
        pixels = result["pixel_values"]
        if pixels.shape[0] != int(grid_hw.prod(-1).sum()):
            raise ValueError("MoonViT image pixels do not match image grids")
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "pixel_values": pixels,
            "image_grid_thw": grid_thw,
        }

    async def _fallback_text_processor(self, input_text, image_data):
        input_ids = self._processor.tokenizer.encode(
            input_text, add_special_tokens=False
        )
        return await asyncio.to_thread(
            self._run_processor_locked, self._process_token_ids, input_ids, image_data
        )
