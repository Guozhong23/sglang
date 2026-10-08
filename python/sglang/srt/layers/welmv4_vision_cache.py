from __future__ import annotations

import os
from collections import OrderedDict

import torch


def vision_grid_key(grid) -> tuple[tuple[int, int, int], ...]:
    rows = grid.tolist() if isinstance(grid, torch.Tensor) else grid
    return tuple(tuple(int(value) for value in row) for row in rows)


class VisionMetadataCache:
    """Per-encoder, byte-bounded LRU for immutable inference metadata tensors."""

    def __init__(self, max_entries: int = 16, max_bytes: int = 128 * 1024**2):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("Vision metadata cache limits must be nonnegative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.current_bytes = 0
        self.signature = None

    @classmethod
    def from_env(cls):
        enabled = os.getenv("SGLANG_VLM_VISION_METADATA_CACHE", "1").strip().lower()
        if enabled not in {"0", "1", "false", "true"}:
            raise ValueError("SGLANG_VLM_VISION_METADATA_CACHE must be 0 or 1")
        return cls(
            max_entries=(
                int(os.getenv("SGLANG_VLM_VISION_METADATA_CACHE_ENTRIES", "16"))
                if enabled in {"1", "true"}
                else 0
            ),
            max_bytes=int(os.getenv("SGLANG_VLM_VISION_METADATA_CACHE_MB", "128"))
            * 1024**2,
        )

    def clear(self):
        self.entries.clear()
        self.current_bytes = 0
        self.signature = None

    def get_or_build(self, key, signature, build, *, enabled: bool):
        if not enabled or not self.max_entries or not self.max_bytes:
            return build()
        if signature != self.signature:
            self.clear()
            self.signature = signature
        if key in self.entries:
            tensors, size = self.entries.pop(key)
            self.entries[key] = (tensors, size)
            return tensors
        tensors = build()
        size = sum(t.numel() * t.element_size() for t in tensors)
        if size > self.max_bytes:
            return tensors
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self.current_bytes + size > self.max_bytes
        ):
            _, (_, evicted_size) = self.entries.popitem(last=False)
            self.current_bytes -= evicted_size
        self.entries[key] = (tensors, size)
        self.current_bytes += size
        return tensors
