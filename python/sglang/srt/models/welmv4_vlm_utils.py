"""Device-independent contracts for the WeLM-V4.5 vision-language backbone."""

import copy


def welmv4_vlm_visual_actor_match(config):
    """Select the native WeLM visual math contract from checkpoint metadata.

    Actor checkpoints use the namespaced image token and Actor-compatible
    rounding boundaries. Older checkpoints retain their existing visual math.
    An explicit vision-config boolean overrides this default without changing
    either the checkpoint config or the separate MoonViT model's behavior.
    """
    missing = object()

    def read(value, name, default=None):
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    vision_config = read(config, "vision_config")
    override = read(vision_config, "welm_visual_actor_match", missing)
    if override is not missing:
        if not isinstance(override, bool):
            raise ValueError("vision_config.welm_visual_actor_match must be a boolean")
        return override
    return read(config, "image_token") == "<|welm_image_pad|>"


def welmv4_vlm_target_config(config, speculative_algorithm=None):
    """Build the non-MTP execution view without mutating checkpoint metadata.

    The VL checkpoint has three draft consumers of source layer zero. Their
    extra K/V banks are never consumed by a target-only forward. The current
    NPU backbone supports one consumer per source, so omit those inactive
    pairs instead of pretending its projection layout supports MTP fan-out.
    Text-only models retain their existing configuration and validation.
    """
    if not getattr(config, "_welmv4_vlm_target_only", False):
        return config
    if speculative_algorithm is not None:
        raise ValueError("WeLM-V4.5-VL does not yet support speculative decoding.")

    target_layers = int(config.num_hidden_layers)
    nextn_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    consumers = list(getattr(config, "kv_mirror_layers", []) or [])
    sources = list(getattr(config, "kv_mirror_imitated_layers", []) or [])
    if target_layers <= 0 or nextn_layers < 0:
        raise ValueError("WeLM-V4.5-VL has invalid target/NextN layer counts.")
    if len(consumers) != len(sources):
        raise ValueError("WeLM-V4.5-VL KV-mirror source/consumer counts differ.")
    if len(set(consumers)) != len(consumers):
        raise ValueError("WeLM-V4.5-VL KV-mirror consumer IDs must be unique.")
    for consumer, source in zip(consumers, sources):
        if not 0 <= source < min(consumer, target_layers):
            raise ValueError("WeLM-V4.5-VL KV-mirror source is invalid.")
        if consumer >= target_layers + nextn_layers:
            raise ValueError("WeLM-V4.5-VL KV-mirror consumer is out of range.")

    pairs = [
        (consumer, source)
        for consumer, source in zip(consumers, sources)
        if consumer < target_layers
    ]
    active_sources = [source for _, source in pairs]
    if len(set(active_sources)) != len(active_sources):
        raise ValueError(
            "WeLM-V4.5-VL target KV-mirror fan-out is not supported by this backbone."
        )
    runtime_config = copy.copy(config)
    runtime_config.kv_mirror_layers = [consumer for consumer, _ in pairs]
    runtime_config.kv_mirror_imitated_layers = active_sources
    if hasattr(runtime_config, "kv_mirror_repeat"):
        runtime_config.kv_mirror_repeat = [1] * len(pairs)
    return runtime_config


def normalize_welmv4_image_tokens(tokens, multimodal_inputs, image_token_id):
    """Map request-local image cache placeholders to logical OE token IDs.

    Keep the request's original IDs unchanged: their image hashes distinguish
    radix-cache prefixes and locate image embeddings. Only the n-gram history
    table needs the logical image token, including history before a chunk.
    """
    if image_token_id is None or multimodal_inputs is None:
        return tokens
    image_pads = {
        item.pad_value
        for item in multimodal_inputs.mm_items
        if item.is_image() and item.pad_value is not None
    }
    if not image_pads:
        return tokens
    return [image_token_id if token in image_pads else token for token in tokens]
