"""Inspect a complete MoonViT checkpoint before allocating model/host OE memory."""

import argparse
import json
from pathlib import Path

ARCH = "WeLMMoonViTVLForConditionalGeneration"


def check_checkpoint(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    if config.get("architectures") != [ARCH]:
        raise ValueError(f"Expected architectures=[{ARCH!r}]")
    if config.get("model_type") != "welm_moonvit_vl":
        raise ValueError("Expected model_type=welm_moonvit_vl")
    vision = config["vision_config"]
    if vision.get("patch_size") != 14 or vision.get("model_type") != "moonvit":
        raise ValueError("Expected MoonViT patch14 vision_config")
    if vision.get("merge_kernel_size") != config.get("projector_merge_kernel_size"):
        raise ValueError("Encoder/projector merge sizes differ")
    if not (path / "tokenizer_config.json").is_file():
        raise ValueError(
            "Missing tokenizer_config.json: use the complete VL checkpoint"
        )
    if not (path / "preprocessor_config.json").is_file():
        raise ValueError(
            "Missing preprocessor_config.json: use the complete VL checkpoint"
        )
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text())["weight_map"]
        missing = sorted({f for f in weights.values() if not (path / f).is_file()})
        if missing:
            raise ValueError(f"Missing weight shards: {missing[:5]}")
        for prefix in (
            "visual.patch_embed.",
            "visual.encoder.",
            "visual.merger.",
            "language_model.",
        ):
            if not any(name.startswith(prefix) for name in weights):
                raise ValueError(f"Weight index has no tensors under {prefix}")
        print(f"Weight index: {len(weights)} tensors; all referenced shards exist")
    elif not (path / "model.safetensors").is_file():
        raise ValueError("Expected model.safetensors or model.safetensors.index.json")
    print(f"Checkpoint architecture: {ARCH}")
    return config


def check_runtime(path, config):
    import sglang
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.models.welm_moonvit_vl import WeLMMoonViTVLForConditionalGeneration
    from sglang.srt.multimodal.processors.welm_moonvit_vl import (
        WeLMMoonViTVLImageProcessor,
    )
    from sglang.srt.utils.hf_transformers_utils import get_processor

    print(f"SGLang import: {sglang.__file__}")
    cls, _ = ModelRegistry.resolve_model_cls([ARCH])
    if cls is not WeLMMoonViTVLForConditionalGeneration:
        raise ValueError(f"Native MoonViT model not registered; resolved {cls}")
    processor = get_processor(str(path), trust_remote_code=True)
    from types import SimpleNamespace

    WeLMMoonViTVLImageProcessor._validate_checkpoint_processor(
        processor, SimpleNamespace(chat_template=None)
    )
    for token, key in (
        ("<|image_pad|>", "image_token_id"),
        ("<|vision_start|>", "vision_start_token_id"),
        ("<|vision_end|>", "vision_end_token_id"),
    ):
        actual = WeLMMoonViTVLImageProcessor._require_token_id(
            processor.tokenizer, token
        )
        if actual != config[key]:
            raise ValueError(
                f"Tokenizer/config mismatch for {token}: {actual} != {config[key]}"
            )
    print(f"Native model: {cls.__module__}.{cls.__name__}")
    print(f"Processor: {type(processor).__name__}")
    print("Runtime imports and processor passed; NPU execution has not been tested")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    try:
        config = check_checkpoint(args.model_path)
        if args.runtime:
            check_runtime(args.model_path, config)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        raise SystemExit(f"MoonViT preflight failed: {exc}") from exc
