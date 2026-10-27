"""Check local WeLM-VL artifacts before starting distributed NPU workers."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path


def vision_parallel_mode(mm_enable_dp_encoder=False, *, supports_ulysses=True) -> str:
    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/layers/welmv4_vision_parallel.py"
    )
    spec = importlib.util.spec_from_file_location(
        "welm_vision_parallel_preflight", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.vision_parallel_mode(
        mm_enable_dp_encoder, supports_ulysses=supports_ulysses
    )


def attention_tp_size(tp: int, dp: int = 1) -> int:
    if tp <= 0 or dp <= 0 or tp % dp:
        raise ValueError(
            "TP and DP sizes must be positive and TP must be divisible by DP"
        )
    return tp // dp


def check_parallel_dimensions(
    config: dict, tp: int, mm_enable_dp_encoder=False, dp: int = 1
) -> str:
    moonvit = "WeLMMoonViTVLForConditionalGeneration" in config.get("architectures", [])
    mode = vision_parallel_mode(mm_enable_dp_encoder, supports_ulysses=not moonvit)
    attn_tp = attention_tp_size(tp, dp)
    dimensions = [(config["text_config"], "num_attention_heads")]
    # DP has full heads/MLPs on each worker; Ulysses only shards heads during
    # attention, while its block MLP weights are replicated.
    if mode != "dp":
        dimensions.append((config["vision_config"], "num_attention_heads"))
    if mode == "tp":
        dimensions.append((config["vision_config"], "intermediate_size"))
    for cfg, field in dimensions:
        if cfg[field] % attn_tp:
            raise ValueError(
                f"{field}={cfg[field]} is not divisible by attention TP={attn_tp}."
            )
    return mode


def check_deployment_profile(profile, tp, dp, mm_enable_dp_encoder):
    attention_tp_size(tp, dp)
    if profile == "dp4ep4":
        if (tp, dp) != (4, 4):
            raise ValueError("The dp4ep4 profile requires TP=4, DP=4 (attention TP=1)")
        if not mm_enable_dp_encoder:
            raise ValueError("The dp4ep4 profile requires --mm-enable-dp-encoder")


def check_artifacts(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text())
    if "WeLMV4VLMForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError(
            "Expected a WeLM-VL checkpoint, not a text-only WeLM checkpoint."
        )

    missing = []
    for cfg in (config, config.get("text_config", {})):
        auto_config = cfg.get("auto_map", {}).get("AutoConfig")
        if auto_config:
            module = auto_config.split("--")[-1].rsplit(".", 1)[0]
            code_path = model_path / (module.replace(".", "/") + ".py")
            if not code_path.is_file():
                missing.append(str(code_path.relative_to(model_path)))
    if not (model_path / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json and tokenizer assets")
    if not any(
        (model_path / name).is_file()
        for name in ("processor_config.json", "preprocessor_config.json")
    ):
        missing.append(
            "processor_config.json / preprocessor_config.json and custom processor code"
        )
    weights = list(model_path.glob("*.safetensors")) + list(
        model_path.glob("pytorch_model*.bin")
    )
    if not weights:
        missing.append("model weights (*.safetensors or pytorch_model*.bin)")
    for index_path in model_path.glob("*.index.json"):
        index = json.loads(index_path.read_text())
        for shard in sorted(set(index.get("weight_map", {}).values())):
            if not (model_path / shard).is_file():
                missing.append(shard)
    if missing:
        raise ValueError(
            "Incomplete model directory; config.json alone cannot serve this model. "
            "Missing:\n  " + "\n  ".join(dict.fromkeys(missing))
        )
    return config


def check_optimization_dependencies(
    tp: int, profile: str = "baseline", dp: int = 1
) -> None:
    """Import enabled extensions without allocating distributed runtime buffers."""
    if profile not in ("baseline", "optimized", "dp4ep4"):
        raise ValueError(f"Unknown WeLM-VL profile: {profile}")

    required = []
    if profile in ("optimized", "dp4ep4"):
        required.append(("deep_ep", ("Buffer", "Config"), f"{profile} DeepEP profile"))
    # Match the production flags and supported attention topologies. The
    # specialized projection supports TP4 and the separately enabled TP2DP2.
    if os.getenv("WELM_NPU_USE_MEGAMOE", "0").lower() in ("1", "true", "yes", "y"):
        required.append(
            (
                "npu_ops_transformer.ops.mega_moe",
                ("mega_moe", "get_symm_buffer_for_mega_moe"),
                "WELM_NPU_USE_MEGAMOE (npu_ops_transformer/custom_transformer)",
            )
        )
    if os.getenv("WELM_NPU_USE_FLASH_ATTN", "0").lower() in ("1", "true"):
        required.append(
            (
                "cann_ops_transformer",
                ("flash_attn", "flash_attn_metadata"),
                "WELM_NPU_USE_FLASH_ATTN",
            )
        )
    attn_tp = attention_tp_size(tp, dp)
    fused_qkv_topology = attn_tp == 4 or (
        (tp, dp) == (4, 2)
        and os.getenv("SGLANG_NPU_WELMV4_DP_FUSED_QKV", "0") == "1"
    )
    if fused_qkv_topology and os.getenv("SGLANG_NPU_WELMV4_FUSED_QKV", "0") == "1":
        # Import the real kernel module: importing cannbotdsl alone would miss
        # incompatible cannbotdsl 0.7 RingBuffer/memory/register APIs.
        required.append(
            (
                "sglang.srt.layers.fused_qkv_proj_norm_rope_cache",
                ("compile_aot", "compile_aot_rank_chunk"),
                "SGLANG_NPU_WELMV4_FUSED_QKV (cannbotdsl)",
            )
        )
    for module_name, symbols, reason in required:
        try:
            module = importlib.import_module(module_name)
            missing = [
                name for name in symbols if not callable(getattr(module, name, None))
            ]
            if missing:
                raise ValueError("missing callable API: " + ", ".join(missing))
        except Exception as exc:
            raise ValueError(
                f"{reason} requires a compatible {module_name} extension: "
                f"{type(exc).__name__}: {exc}"
            ) from exc


def check_runtime(
    model_path: Path,
    tp: int,
    base_device: int,
    chat_template: str | None,
    profile: str = "baseline",
    mm_enable_dp_encoder: bool = False,
    dp: int = 1,
):
    mode = vision_parallel_mode(mm_enable_dp_encoder)
    import torch
    import torch_npu  # noqa: F401

    from sglang.srt.layers.welmv4_vision_npu import VisionNPUFusionConfig
    from sglang.srt.utils.hf_transformers_utils import get_config, get_processor

    if not torch.npu.is_available():
        raise ValueError("torch_npu is installed but no Ascend NPU is available.")
    if base_device < 0 or base_device + tp > torch.npu.device_count():
        raise ValueError(
            f"Need {tp} visible NPUs starting at {base_device}; "
            f"only {torch.npu.device_count()} are visible."
        )
    with torch.npu.device(torch.device(f"npu:{base_device}")):
        VisionNPUFusionConfig.from_env().validate_runtime(
            torch.device(f"npu:{base_device}")
        )
        check_optimization_dependencies(tp, profile, dp)
    config = get_config(str(model_path), trust_remote_code=True, local_files_only=True)
    processor = get_processor(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    for method in (
        "resolve_tokenized_multimodal_inputs",
        "process_resolved_tokenized_multimodal_prompt",
    ):
        if not callable(getattr(processor, method, None)):
            raise ValueError(
                f"Checkpoint processor is missing required method: {method}"
            )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Checkpoint processor does not expose its tokenizer.")
    if not (
        chat_template
        or getattr(tokenizer, "chat_template", None)
        or getattr(processor, "chat_template", None)
    ):
        raise ValueError(
            "Missing model-native chat template; supply CHAT_TEMPLATE=/path/to/template.jinja."
        )
    token_id = tokenizer.convert_tokens_to_ids(config.image_token)
    if token_id != config.image_token_id:
        raise ValueError(
            f"Image token mismatch: tokenizer={token_id}, config={config.image_token_id}"
        )
    print(
        f"Runtime: torch={torch.__version__}, NPU={torch.npu.get_device_name(base_device)}"
    )
    print(f"Processor: {type(processor).__name__}; image token ID: {token_id}")
    # Probe the exact vision layout before loading the large text backbone.
    # Zero Q/K yield a known mean within each image, so this also checks that
    # the cumulative TND sequence boundaries keep the two images isolated.
    vision = config.vision_config
    if isinstance(vision, dict):
        heads, width = vision["num_attention_heads"], vision["hidden_size"]
    else:
        heads, width = vision.num_attention_heads, vision.hidden_size
    head_dim = width // heads
    attn_tp = attention_tp_size(tp, dp)
    attention_heads = heads if mode == "dp" else heads // attn_tp
    device = torch.device(f"npu:{base_device}")
    with torch.npu.device(device), torch.inference_mode():
        q = torch.zeros(
            (8, attention_heads, head_dim), device=device, dtype=torch.bfloat16
        )
        values = torch.arange(8, device=device, dtype=torch.float32).to(torch.bfloat16)
        values = values[:, None, None].expand_as(q).contiguous()
        output = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=q,
            value=values,
            actual_seq_lengths=[4, 8],
            actual_seq_lengths_kv=[4, 8],
            scale=head_dim**-0.5,
            num_heads=attention_heads,
            num_key_value_heads=attention_heads,
            sparse_mode=0,
            input_layout="TND",
        )[0]
        expected = torch.full_like(q, 1.5)
        expected[4:] = 5.5
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
    print(
        f"NPU vision attention probe passed: BF16, head_dim={head_dim}, "
        f"vision={mode}, local_heads={attention_heads}, "
        f"LLM attention TP={attn_tp}, DP={dp}, world TP={tp}."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--base-device", type=int, default=0)
    parser.add_argument("--chat-template")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--mm-enable-dp-encoder", action="store_true")
    parser.add_argument(
        "--profile", choices=("baseline", "optimized", "dp4ep4"), default="baseline"
    )
    args = parser.parse_args()
    try:
        config = check_artifacts(args.model_path)
        if args.tp not in (1, 2, 4, 8):
            raise ValueError(
                "This checkpoint's basic TP path supports TP=1, 2, 4 or 8."
            )
        check_deployment_profile(
            args.profile, args.tp, args.dp, args.mm_enable_dp_encoder
        )
        check_parallel_dimensions(config, args.tp, args.mm_enable_dp_encoder, args.dp)
        if args.chat_template and not Path(args.chat_template).is_file():
            raise ValueError(f"Chat template does not exist: {args.chat_template}")
        if args.runtime:
            check_runtime(
                args.model_path,
                args.tp,
                args.base_device,
                args.chat_template,
                args.profile,
                args.mm_enable_dp_encoder,
                args.dp,
            )
    except (
        ValueError,
        OSError,
        ImportError,
        KeyError,
        RuntimeError,
        AssertionError,
    ) as exc:
        parser.exit(2, f"WeLM-VL preflight failed: {exc}\n")
    print(f"WeLM-VL artifact checks passed: {args.model_path}")


if __name__ == "__main__":
    main()
