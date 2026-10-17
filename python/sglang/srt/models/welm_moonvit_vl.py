# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Native WeLM + MoonViT serving; reuse WeLM's TP/EP and eager OE boundary."""

from __future__ import annotations

import copy
import math
import os
import re

import torch
from torch import nn
from transformers.activations import PytorchGELUTanh

from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.welmv4_vision_dp import VisionDPInputSpec, create_vision_dp_group
from sglang.srt.models.kimi_vl_moonvit import (
    MoonVisionPatchEmbed,
    MoonVitEncoderLayer,
    patch_merger,
)
from sglang.srt.models.welmv4_vlm import (
    WeLMV4VLMForConditionalGeneration,
    _as_config,
    _vision_dp_tp_kwargs,
    _vision_max_patches,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


def moonvit_rope_coefficients(grids, head_dim, device):
    """Match checkpoint complex RoPE: adjacent pairs, alternating x/y phases."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 4, device=device).float() / head_dim)
    )
    phases = []
    for h, w in grids:
        x = torch.arange(w, device=device).repeat(h).float()
        y = torch.arange(h, device=device).repeat_interleave(w).float()
        phases.append(
            torch.stack(
                (torch.outer(x, inv_freq), torch.outer(y, inv_freq)), -1
            ).flatten(1)
        )
    phases = torch.cat(phases)
    return phases.cos(), phases.sin()


def moonvit_apply_rope(q, k, position_embeddings, x_shape=None):
    # NPU complex tensors are unnecessary: expand the exact complex multiply
    # in FP32 and round only the final result back to the activation dtype.
    cos, sin = (x.unsqueeze(1) for x in position_embeddings)

    def rotate(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs.unbind(-1)
        return (
            torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1)
            .flatten(-2)
            .to(x.dtype)
        )

    return rotate(q), rotate(k)


class WeLMMoonViTEncoder(nn.Module):
    def __init__(self, config, prefix="", use_data_parallel=False):
        super().__init__()
        self.config = config
        self.vision_dp_max_patches = _vision_max_patches()
        self.patch_size = config.patch_size
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.head_dim = config.hidden_size // config.num_attention_heads
        tp = 1 if use_data_parallel else get_parallel().attn_tp_size
        if config.num_attention_heads % tp or config.intermediate_size % tp:
            raise ValueError("MoonViT attention heads and MLP size must divide TP size")
        if config.hidden_size % config.num_attention_heads or self.head_dim % 4:
            raise ValueError("MoonViT head dimension must be divisible by 4")
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )
        # Match the checkpoint's visual.encoder.* namespace and final norm.
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList(
            MoonVitEncoderLayer(
                num_heads=config.num_attention_heads,
                hidden_dim=config.hidden_size,
                mlp_dim=config.intermediate_size,
                activation=PytorchGELUTanh(),
                attn_bias=True,
                use_data_parallel=use_data_parallel,
                prefix=add_prefix(f"encoder.blocks.{i}", prefix),
            )
            for i in range(config.num_hidden_layers)
        )
        for block in self.encoder.blocks:
            block.attn.customized_position_embedding_applier = moonvit_apply_rope
        self.encoder.final_layernorm = nn.LayerNorm(config.hidden_size, eps=1e-5)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self):
        return self.patch_embed.proj.weight.device

    @property
    def vision_dp_input_spec(self):
        return VisionDPInputSpec(
            merge=self.merge_kernel_size,
            pixel_tail_shape=(3, self.patch_size, self.patch_size),
            output_hidden_size=self.config.out_hidden_size,
            image_only=True,
            grid_limit=512,
        )

    def clear_metadata_cache(self):
        self.patch_embed.pos_emb._interpolated_pos_emb_cache.clear()

    def _apply(self, fn, recurse=True):
        self.clear_metadata_cache()
        return super()._apply(fn, recurse=recurse)

    def forward(self, pixel_values, image_grid_thw):
        grid = image_grid_thw.to(device="cpu", dtype=torch.long)
        if grid.ndim != 2 or grid.shape[1] != 3 or not torch.all(grid[:, 0] == 1):
            raise ValueError("MoonViT requires image_grid_thw=[N,3] with t=1")
        grids = grid[:, 1:].tolist()
        kh, kw = self.merge_kernel_size
        if not grids or any(
            not (0 < h < 512 and 0 < w < 512) or h % kh or w % kw for h, w in grids
        ):
            raise ValueError(
                "MoonViT grids must fit RoPE and be divisible by merge size"
            )
        lengths = [h * w for h, w in grids]
        if pixel_values.shape != (sum(lengths), 3, self.patch_size, self.patch_size):
            raise ValueError(
                "MoonViT pixels must be [sum(h*w),3,patch_size,patch_size]"
            )
        weight = self.patch_embed.proj.weight
        pixels = pixel_values.to(device=weight.device, dtype=weight.dtype)
        hidden = self.patch_embed(pixels, grid[:, 1:])
        rope = moonvit_rope_coefficients(grids, self.head_dim, hidden.device)
        # Ascend's eager TND attention consumes CPU cumulative sequence lengths.
        # Avoid a repeated device->host sync inside every encoder block.
        cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        for block in self.encoder.blocks:
            hidden = block(hidden, cu, rope_freqs_cis=rope, max_seqlen=max(lengths))
        hidden = self.encoder.final_layernorm(hidden)
        return torch.cat(patch_merger(hidden, grid[:, 1:], self.merge_kernel_size))


class WeLMMoonViTProjector(nn.Module):
    def __init__(self, config, prefix="", use_data_parallel=False):
        super().__init__()
        self.hidden_size = config.hidden_size * math.prod(config.merge_kernel_size)
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=add_prefix("linear_1", prefix),
            **_vision_dp_tp_kwargs(use_data_parallel),
        )
        self.linear_2 = RowParallelLinear(
            self.hidden_size,
            config.out_hidden_size,
            bias=True,
            prefix=add_prefix("linear_2", prefix),
            **_vision_dp_tp_kwargs(use_data_parallel),
        )

    def forward(self, hidden):
        hidden = self.pre_norm(hidden).reshape(-1, self.hidden_size)
        hidden, _ = self.linear_1(hidden)
        hidden = torch.nn.functional.gelu(hidden, approximate="none")
        hidden, _ = self.linear_2(hidden)
        return hidden


def convert_moonvit_weights(weights, text_config):
    """Stream checkpoint tensors into the existing TP/EP-aware weight loaders."""
    for name, weight in weights:
        if name.startswith("visual.merger."):
            yield name.replace("visual.merger.", "vision_projector.", 1), weight
        elif name.startswith("visual."):
            name = name.replace("visual.", "vision_encoder.", 1)
            name = name.replace(".wqkv.", ".attn.qkv_proj.").replace(
                ".wo.", ".attn.proj."
            )
            yield name, weight
        elif name.startswith("language_model."):
            name = name[len("language_model.") :]
            if name == "model.oe_up_proj.weight":
                name = "model.oe_gate_up_proj.weight"
            if ".mlp.stacked_" not in name:
                yield name, weight
                continue
            stem, kind = name.split(".mlp.stacked_", 1)
            projections = {
                "gate_weights": "gate_proj",
                "up_weights": "up_proj",
                "down_weights": "down_proj",
            }
            if kind not in projections:
                raise ValueError(f"Unknown MoonViT stacked expert tensor: {name}")
            h, m = text_config.hidden_size, text_config.moe_intermediate_size
            expected = (h, m) if kind == "down_weights" else (m, h)
            if weight.ndim != 3 or weight.shape[0] != text_config.num_experts:
                raise ValueError(
                    f"Invalid stacked expert shape for {name}: {tuple(weight.shape)}"
                )
            if tuple(weight.shape[1:]) == expected:
                transpose = False
            elif tuple(weight.shape[1:]) == expected[::-1] and h != m:
                transpose = True
            else:
                raise ValueError(
                    f"Invalid stacked expert matrix for {name}: {tuple(weight.shape)}"
                )
            for expert in range(text_config.num_experts):
                matrix = weight[expert]
                if transpose:
                    matrix = matrix.t().contiguous()
                yield f"{stem}.mlp.experts.{expert}.{projections[kind]}.weight", matrix
        else:
            raise ValueError(f"Unexpected MoonViT checkpoint weight prefix: {name}")


def checked_moonvit_language_weights(weights, parameter_names, num_experts):
    """Reject missing/unmapped BF16 text weights before serving random parameters."""
    required = set()
    for name in parameter_names:
        if name.startswith("vision_"):
            continue
        if ".experts.w13_weight" in name or ".experts.w2_weight" in name:
            parts = ("gate_proj", "up_proj") if ".w13_" in name else ("down_proj",)
            for expert in range(num_experts):
                for part in parts:
                    required.add(
                        re.sub(
                            r"experts\.w(?:13|2)_weight",
                            f"experts.{expert}.{part}.weight",
                            name,
                        )
                    )
        elif ".self_attn.qkv_proj." in name:
            required.update(
                name.replace(".qkv_proj.", f".{part}_proj.") for part in ("q", "k", "v")
            )
        elif ".gate_up_proj." in name:
            required.update(
                name.replace(".gate_up_proj.", f".{part}_proj.")
                for part in ("gate", "up")
            )
        else:
            required.add(name)
    for name, weight in weights:
        if name.startswith("vision_"):
            yield name, weight
            continue
        if name.endswith("rotary_emb.inv_freq"):
            continue
        names = {name}
        if ".self_attn.qkv_proj." in name:
            names = {
                name.replace(".qkv_proj.", f".{part}_proj.") for part in ("q", "k", "v")
            }
        elif ".gate_up_proj." in name:
            names = {
                name.replace(".gate_up_proj.", f".{part}_proj.")
                for part in ("gate", "up")
            }
        if not names <= required:
            raise ValueError(f"Unmapped or duplicate MoonViT language weight: {name}")
        required.difference_update(names)
        yield name, weight
    if required:
        raise ValueError(
            f"Missing MoonViT language weights ({len(required)}): {sorted(required)[:20]}"
        )


class WeLMMoonViTVLForConditionalGeneration(WeLMV4VLMForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix=""):
        unsupported = [
            flag
            for flag in (
                "SGLANG_VLM_VISION_USE_ULYSSES",
                "SGLANG_VLM_NPU_USE_MC2",
                "SGLANG_VLM_NPU_FUSE_GELU",
                "SGLANG_VLM_NPU_FUSE_ADD_LAYERNORM",
                "SGLANG_VLM_NPU_FUSE_ROPE",
            )
            if os.getenv(flag, "0").lower() not in ("0", "false", "")
        ]
        if unsupported:
            raise ValueError(
                "MoonViT supports vision TP or --mm-enable-dp-encoder. Set these old-ViT "
                f"options to 0: {', '.join(unsupported)}"
            )
        config = copy.copy(config)
        config.vision_config = copy.copy(_as_config(config.vision_config))
        vision = config.vision_config
        vision.out_hidden_size = _as_config(config.text_config).hidden_size
        if tuple(vision.merge_kernel_size) != tuple(config.projector_merge_kernel_size):
            raise ValueError("MoonViT encoder/projector merge sizes must agree")
        super().__init__(config, quant_config=quant_config, prefix=prefix)

    def _init_vision_modules(self, vision_config, prefix):
        self._vision_dp_group = create_vision_dp_group(
            vision_config, max_patches=_vision_max_patches()
        )
        use_dp = self._vision_dp_group is not None
        self.vision_encoder = WeLMMoonViTEncoder(
            vision_config,
            prefix=add_prefix("vision_encoder", prefix),
            use_data_parallel=use_dp,
        )
        self.vision_projector = WeLMMoonViTProjector(
            vision_config,
            prefix=add_prefix("vision_projector", prefix),
            use_data_parallel=use_dp,
        )

    def load_weights(self, weights, is_nextn=False):
        return super().load_weights(
            checked_moonvit_language_weights(
                convert_moonvit_weights(weights, self.config),
                set(dict(self.named_parameters())),
                self.config.num_experts,
            ),
            is_nextn=is_nextn,
        )


EntryClass = WeLMMoonViTVLForConditionalGeneration
