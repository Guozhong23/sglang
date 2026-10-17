# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""WeLM vision mode selection, shared by serving and portable preflight."""

import logging
import os


def vision_parallel_mode(mm_enable_dp_encoder=False, *, supports_ulysses=True):
    def flag(name):
        value = os.getenv(name, "0").strip().lower()
        if value not in ("0", "1", "false", "true"):
            raise ValueError(f"{name} must be 0, 1, false or true")
        return value in ("1", "true")

    sp = flag("SGLANG_VLM_VISION_USE_ULYSSES")
    if flag("SGLANG_VLM_VISION_USE_DP"):
        message = (
            "SGLANG_VLM_VISION_USE_DP is removed; unset it and use "
            "--mm-enable-dp-encoder for image DP"
        )
        if not mm_enable_dp_encoder:
            raise ValueError(message)
        logging.getLogger(__name__).warning(message)
    if sp and mm_enable_dp_encoder:
        raise ValueError("Vision Ulysses and image DP are mutually exclusive")
    if sp and not supports_ulysses:
        raise ValueError("MoonViT does not support SGLANG_VLM_VISION_USE_ULYSSES")
    return "dp" if mm_enable_dp_encoder else "ulysses" if sp else "tp"
