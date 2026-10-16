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
class VisionDPPlan:
    grids: tuple[tuple[int, int, int], ...]
    merge: int
    world_size: int
    rank: int

    def __post_init__(self):
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid image DP communication group")
        if not self.grids or self.merge <= 0:
            raise ValueError("Image DP requires nonempty grids and positive merge size")
        if any(
            t <= 0 or h <= 0 or w <= 0 or h % self.merge or w % self.merge
            for t, h, w in self.grids
        ):
            raise ValueError("Image DP grids must be positive and merge-aligned")

    @cached_property
    def patch_counts(self):
        return tuple(t * h * w for t, h, w in self.grids)

    @cached_property
    def token_counts(self):
        return tuple(count // self.merge**2 for count in self.patch_counts)

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
    """Only owner pixels cross to the device, via the encoder's normal cast."""
    plan = VisionDPPlan(
        tuple(_image_grid(item) for item in items),
        encoder.config.spatial_merge_size,
        group.world_size,
        group.rank_in_group,
    )
    expected_width = (
        encoder.config.in_channels
        * encoder.config.temporal_patch_size
        * encoder.config.patch_size**2
    )
    # All ranks validate the full inexpensive shape manifest before any rank
    # starts the model. Invalid non-owned input cannot strand its peers in AG.
    for item, count in zip(items, plan.patch_counts):
        if tuple(item.feature.shape) != (count, expected_width):
            raise ValueError("Image DP pixel shape does not match its grid")
    indices = plan.assignments[plan.rank]
    if indices:
        # Deferred mm_utils transfer keeps the common CPU storage intact.
        pixels = torch.cat([items[index].feature for index in indices], dim=0)
        grids = torch.tensor(
            [plan.grids[index] for index in indices], dtype=torch.int64
        )
        local = projector(encoder(pixels, grids))
    else:
        # Do not send an empty image through FIA/position embedding. Idle ranks
        # still enter the same AllGather with the same positive buffer extent.
        local = torch.empty(
            (0, encoder.config.out_hidden_size),
            dtype=encoder.dtype,
            device=encoder.device,
        )
    return plan.gather(local, group)
