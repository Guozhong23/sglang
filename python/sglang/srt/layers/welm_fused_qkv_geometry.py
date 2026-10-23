"""Compile-time BF16 WeLM QKV geometry, independent of device libraries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedQkvGeometry:
    num_q_heads: int = 6
    num_kv_heads: int = 1
    head_dim: int = 256
    has_mirror: bool = False

    def __post_init__(self):
        if (
            self.num_q_heads not in (6, 12)
            or self.num_kv_heads != 1
            or self.head_dim != 256
        ):
            raise ValueError("WeLM fused BF16 QKV supports Q6/Q12, KV1, head_dim=256")

    @property
    def q_width(self):
        return self.num_q_heads * self.head_dim

    @property
    def qkv_width(self):
        return (self.num_q_heads + 2 + 2 * self.has_mirror) * self.head_dim

    @property
    def units(self):
        q = self.num_q_heads
        return dict(k=q, v=q + 1, mirror_k=q + 2, mirror_v=q + 3)

    @classmethod
    def from_width(cls, width):
        layouts = {
            2048: (6, False),
            2560: (6, True),
            3584: (12, False),
            4096: (12, True),
        }
        if width not in layouts:
            raise ValueError(f"Unsupported WeLM BF16 QKV projection width: {width}")
        heads, mirror = layouts[width]
        return cls(num_q_heads=heads, has_mirror=mirror)
