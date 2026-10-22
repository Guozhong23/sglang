# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Equal-count Ulysses exchanges for the BF16 WeLM vision encoder.

Only the sequence is padded, at its GLOBAL tail. After sequence-to-head
exchange the tail is removed BEFORE attention; zero keys must never enter
softmax. Keeping all padding at the global tail also makes arbitrary image
boundaries compatible with the original packed cumulative sequence lengths.

This module has no import-time serving dependencies, so layouts can be tested
on CPU. ``group`` implements GroupCoordinator's all_to_all_single and
all_gather_into_tensor APIs. Collectives are synchronous from the consumer's
stream perspective; never fall back to a different collective after failure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch


def resolve_vision_parallel_group(config, use_data_parallel=False, max_patches=0):
    """Agree on image-DP / Ulysses / legacy TP before loading vision weights."""
    from sglang.srt.layers.welmv4_vision_parallel import vision_parallel_mode
    from sglang.srt.runtime_context import get_mm
    from sglang.srt.utils import is_npu

    raw_sp = os.environ.get("SGLANG_VLM_VISION_USE_ULYSSES", "0").strip().lower()
    raw_dp = os.environ.get("SGLANG_VLM_VISION_USE_DP", "0").strip().lower()
    dp = get_mm().mm_enable_dp_encoder
    if not is_npu():
        if vision_parallel_mode(dp) != "tp":
            raise ValueError("WeLM vision Ulysses / image DP requires the NPU runtime")
        return None, None

    from sglang.srt.distributed import get_attn_tp_group, get_tp_group
    from sglang.srt.environ import envs

    tp_group = get_tp_group()
    attn_group = get_attn_tp_group()
    mode = vision_parallel_mode(dp)
    if list(attn_group.ranks) != list(tp_group.ranks) and mode != "dp":
        raise ValueError(
            "WeLM attention DP requires --mm-enable-dp-encoder; "
            "vision TP and Ulysses across different request groups are unsupported"
        )
    # Each attention-DP shard owns different requests. Settings, cache consensus,
    # ownership and embedding gather must all stay within that request's group.
    group = attn_group if mode == "dp" else tp_group
    graph = envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
    settings = (
        raw_sp,
        raw_dp,
        max_patches,
        config.num_attention_heads,
        config.hidden_size,
        config.num_hidden_layers,
        graph,
        use_data_parallel,
        get_mm().mm_enable_dp_encoder,
    )
    if group.world_size > 1:
        peer_settings = [None] * group.world_size
        torch.distributed.all_gather_object(
            peer_settings, settings, group=group.cpu_group
        )
        if any(peer != settings for peer in peer_settings):
            raise ValueError("WeLM vision parallel settings differ across TP ranks")
    if mode == "tp":
        return None, None
    if graph or (mode == "ulysses" and use_data_parallel):
        raise ValueError(
            "Vision parallel modes do not support vision graphs or Ulysses with replicated encoder DP"
        )
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("Vision hidden_size must be divisible by attention heads")
    return group, mode


def create_vision_ulysses_group(config, use_data_parallel=False, max_patches=0):
    group, mode = resolve_vision_parallel_group(config, use_data_parallel, max_patches)
    if mode != "ulysses":
        return None
    VisionUlyssesPlan(
        1, config.num_attention_heads, group.world_size, group.rank_in_group
    )
    logging.getLogger(__name__).info(
        "WeLM vision Ulysses enabled: SP=%d rank=%d, replicated block weights; "
        "block AllReduce bypassed, projector retains TP. BF16 eager only.",
        group.world_size,
        group.rank_in_group,
    )
    return group


def validate_vision_ulysses_input(hidden: torch.Tensor):
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16:
        raise ValueError("WeLM vision Ulysses requires BF16 NPU activations")


@dataclass(frozen=True)
class VisionUlyssesPlan:
    tokens: int
    heads: int
    world_size: int
    rank: int

    def __post_init__(self):
        if self.tokens <= 0 or self.heads <= 0 or self.world_size <= 0:
            raise ValueError("Ulysses tokens, heads and world_size must be positive")
        if self.heads % self.world_size:
            raise ValueError("Vision attention heads must be divisible by SP size")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid Ulysses rank")

    @property
    def local_tokens(self) -> int:
        return (self.tokens + self.world_size - 1) // self.world_size

    @property
    def padded_tokens(self) -> int:
        return self.local_tokens * self.world_size

    @property
    def local_heads(self) -> int:
        return self.heads // self.world_size

    def local_slice(self, tensor: torch.Tensor, pad_value=0) -> torch.Tensor:
        """Select consecutive global rows, including all-padding ranks.

        Rotary cos uses pad_value=1 and sin uses zero. Padding hidden states
        need not stay zero through biased linears: they are excluded from FIA.
        """
        if tensor.shape[0] != self.tokens:
            raise ValueError("Global tensor length does not match Ulysses plan")
        start = self.rank * self.local_tokens
        part = tensor[start : start + self.local_tokens]
        missing = self.local_tokens - part.shape[0]
        if missing:
            part = torch.cat(
                (part, tensor.new_full((missing, *tensor.shape[1:]), pad_value)), dim=0
            )
        return part.contiguous()

    def pack_qkv(self, q, k, v) -> torch.Tensor:
        """[L,H,D] x3 -> [destination,L,QKV,H/P,D]."""
        if (
            q.ndim != 3
            or q.shape[:2] != (self.local_tokens, self.heads)
            or k.shape != q.shape
            or v.shape != q.shape
            or not (q.dtype == k.dtype == v.dtype)
            or not (q.device == k.device == v.device)
        ):
            raise ValueError("Invalid local QKV shape, dtype or device")
        return (
            torch.stack((q, k, v), dim=1)
            .reshape(
                self.local_tokens, 3, self.world_size, self.local_heads, q.shape[2]
            )
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )

    def unpack_qkv(self, received: torch.Tensor):
        """Receive order is [source,L,QKV,H/P,D], not head-major order."""
        if received.ndim != 5 or received.shape[:4] != (
            self.world_size,
            self.local_tokens,
            3,
            self.local_heads,
        ):
            raise ValueError("Invalid received QKV layout")
        global_qkv = received.reshape(
            self.padded_tokens, 3, self.local_heads, received.shape[-1]
        )[: self.tokens]
        # FIA needs contiguous TND tensors, and MUST NOT see padded keys.
        return tuple(global_qkv[:, i].contiguous() for i in range(3))

    def pack_output(self, output: torch.Tensor) -> torch.Tensor:
        """[N,H/P,D] -> [destination,L,H/P,D], padding only global tail."""
        if output.ndim != 3 or output.shape[:2] != (self.tokens, self.local_heads):
            raise ValueError("Invalid head-sharded attention output")
        missing = self.padded_tokens - self.tokens
        if missing:
            output = torch.cat(
                (output, output.new_zeros((missing, *output.shape[1:]))), dim=0
            )
        return output.reshape(
            self.world_size, self.local_tokens, self.local_heads, output.shape[-1]
        ).contiguous()

    def unpack_output(self, received: torch.Tensor) -> torch.Tensor:
        """[source,L,H/P,D] -> [L,H,D]; source rank owns a head range."""
        if received.ndim != 4 or received.shape[:3] != (
            self.world_size,
            self.local_tokens,
            self.local_heads,
        ):
            raise ValueError("Invalid received attention output layout")
        return (
            received.permute(1, 0, 2, 3)
            .contiguous()
            .reshape(self.local_tokens, self.heads, received.shape[-1])
        )

    def _exchange(self, packed, group):
        self._check_group(group)
        if self.world_size == 1:
            return packed
        received = torch.empty_like(packed)
        # dim0 is destination on send, source on receive. All ranks have the
        # same element count, even when a rank has no real sequence rows.
        group.all_to_all_single(received, packed)
        return received

    def _check_group(self, group):
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise ValueError("Ulysses plan and communication group do not match")

    def sequence_to_heads(self, q, k, v, group):
        return self.unpack_qkv(self._exchange(self.pack_qkv(q, k, v), group))

    def heads_to_sequence(self, output, group):
        return self.unpack_output(self._exchange(self.pack_output(output), group))

    def gather_sequence(self, local: torch.Tensor, group) -> torch.Tensor:
        """Recover original patch order before the TP spatial-merge projector."""
        self._check_group(group)
        if local.shape[0] != self.local_tokens:
            raise ValueError("Invalid local sequence length for AllGather")
        if self.world_size == 1:
            return local
        result = local.new_empty((self.padded_tokens, *local.shape[1:]))
        group.all_gather_into_tensor(result, local.contiguous())
        return result[: self.tokens]
