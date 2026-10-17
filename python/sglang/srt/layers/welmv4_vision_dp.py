# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Whole-image WeLM vision DP over the text TP group.

Each owner runs a replicated encoder and projector, without layer collectives.
Only projected embeddings are gathered. Padding belongs to communication buffers
and never enters vision attention or spatial merging. Layout helpers run on CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

import torch


def create_vision_dp_group(config, max_patches=0):
    from sglang.srt.layers.welmv4_vision_ulysses import resolve_vision_parallel_group

    group, mode = resolve_vision_parallel_group(config, max_patches=max_patches)
    if mode != "dp":
        return None
    logging.getLogger(__name__).info(
        "WeLM image DP enabled: DP=%d rank=%d; encoder and projector replicated, "
        "whole images assigned to owners, projected embeddings gathered once. "
        "A single image uses one rank. BF16 eager only.",
        group.world_size,
        group.rank_in_group,
    )
    return group


def _image_grid(item):
    grid = item.image_grid_thw
    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().tolist()
    if len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("WeLM image DP requires one grid row per image item")
    return tuple(int(value) for value in grid[0])


def synchronize_vision_dp_cache(items, cached_mask, *, group):
    """Use the union of misses even if a rank's LRU cache differs.

    All ranks call once per multimodal batch, including all-hit/empty-overlap
    batches. Validate image identity/order before any device collective. This
    control-plane synchronization also makes a divergent request fail clearly.
    """
    manifest = tuple(
        (item.hash, _image_grid(item), tuple(item.feature.shape)) for item in items
    )
    if len(cached_mask) != len(items):
        raise ValueError("Image DP cache mask does not match image items")
    state = manifest, tuple(bool(hit) for hit in cached_mask)
    peers = [state]
    if group.world_size > 1:
        peers = [None] * group.world_size
        torch.distributed.all_gather_object(peers, state, group=group.cpu_group)
    if any(peer[0] != manifest or len(peer[1]) != len(items) for peer in peers):
        raise ValueError("WeLM image DP image order/grids differ across TP ranks")
    return [any(not peer[1][index] for peer in peers) for index in range(len(items))]


@dataclass(frozen=True)
class VisionDPInputSpec:
    merge: tuple[int, int]
    pixel_tail_shape: tuple[int, ...]
    output_hidden_size: int
    image_only: bool = False
    grid_limit: int | None = None

    def validate(self, item, grid, count):
        t, h, w = grid
        if self.image_only and t != 1:
            raise ValueError("MoonViT image DP requires t=1")
        if self.grid_limit is not None and max(h, w) >= self.grid_limit:
            raise ValueError("MoonViT image DP grid exceeds RoPE limit")
        if tuple(item.feature.shape) != (count, *self.pixel_tail_shape):
            raise ValueError("Image DP pixel shape does not match its grid")


@dataclass(frozen=True)
class VisionDPPlan:
    grids: tuple[tuple[int, int, int], ...]
    merge: int | tuple[int, int]
    world_size: int
    rank: int

    def __post_init__(self):
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid image DP communication group")
        if isinstance(self.merge, int):
            object.__setattr__(self, "merge", (self.merge, self.merge))
        if not self.grids or len(self.merge) != 2 or min(self.merge) <= 0:
            raise ValueError("Image DP requires nonempty grids and positive merge size")
        if any(
            t <= 0 or h <= 0 or w <= 0 or h % self.merge[0] or w % self.merge[1]
            for t, h, w in self.grids
        ):
            raise ValueError("Image DP grids must be positive and merge-aligned")

    @cached_property
    def patch_counts(self):
        return tuple(t * h * w for t, h, w in self.grids)

    @cached_property
    def token_counts(self):
        return tuple(
            count // (self.merge[0] * self.merge[1]) for count in self.patch_counts
        )

    @cached_property
    def assignments(self):
        loads = [0] * self.world_size
        assignments = [[] for _ in range(self.world_size)]
        # Stable longest-first scheduling; ties use original image/rank order.
        for index in sorted(
            range(len(self.grids)), key=lambda i: (-self.patch_counts[i], i)
        ):
            owner = min(range(self.world_size), key=lambda rank: (loads[rank], rank))
            assignments[owner].append(index)
            loads[owner] += self.patch_counts[index]
        return tuple(tuple(sorted(indices)) for indices in assignments)

    @cached_property
    def rows_per_rank(self):
        return tuple(
            sum(self.token_counts[i] for i in indices) for indices in self.assignments
        )

    @cached_property
    def capacity(self):
        return max(self.rows_per_rank)

    def gather(self, local, group):
        if group.world_size != self.world_size or group.rank_in_group != self.rank:
            raise ValueError("Image DP plan and communication group differ")
        if local.ndim != 2 or local.shape[0] != self.rows_per_rank[self.rank]:
            raise ValueError("Image DP projected output has incorrect shape")
        if self.world_size == 1:
            gathered = local
        else:
            padded = local.new_zeros((self.capacity, local.shape[1]))
            padded[: local.shape[0]].copy_(local)
            gathered = local.new_empty(
                (self.world_size * self.capacity, local.shape[1])
            )
            group.all_gather_into_tensor(gathered, padded)
        images = [None] * len(self.grids)
        for rank, indices in enumerate(self.assignments):
            offset = rank * self.capacity
            for index in indices:
                length = self.token_counts[index]
                # Cache each image's own storage; a small surviving entry must
                # not pin the full cross-rank gather buffer after other eviction.
                images[index] = gathered[offset : offset + length].clone()
                offset += length
        return images


def encode_vision_dp(items, encoder, projector, group):
    """Validate on all ranks, transfer only owned images, gather after projection."""
    spec = encoder.vision_dp_input_spec
    plan = VisionDPPlan(
        tuple(_image_grid(item) for item in items),
        spec.merge,
        group.world_size,
        group.rank_in_group,
    )
    # Validate every image before any rank starts device computation/collectives.
    for item, grid, count in zip(items, plan.grids, plan.patch_counts):
        spec.validate(item, grid, count)
    indices = plan.assignments[plan.rank]
    if indices:
        # A cap limits owner activation memory, never splits a single image's
        # attention. Outputs retain original order and use one final AllGather.
        cap = getattr(encoder, "vision_dp_max_patches", 0)
        batches, batch, patches = [], [], 0
        for index in indices:
            if cap and batch and patches + plan.patch_counts[index] > cap:
                batches.append(batch)
                batch, patches = [], 0
            batch.append(index)
            patches += plan.patch_counts[index]
        batches.append(batch)
        outputs = []
        for batch in batches:
            pixels = torch.cat([items[index].feature for index in batch], dim=0)
            grids = torch.tensor(
                [plan.grids[index] for index in batch], dtype=torch.int64
            )
            outputs.append(projector(encoder(pixels, grids)))
        local = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
    else:
        # Idle ranks enter the same AllGather without running empty attention.
        local = torch.empty(
            (0, spec.output_hidden_size), dtype=encoder.dtype, device=encoder.device
        )
    if local.ndim != 2 or local.shape[1] != spec.output_hidden_size:
        raise ValueError(
            "Image DP projector output width does not match text hidden size"
        )
    return plan.gather(local, group)
