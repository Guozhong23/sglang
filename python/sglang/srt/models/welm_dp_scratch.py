"""Event-protected scratch leases for serial/overlapped eager WeLM prefills.

Captured graphs retain their own row views. Only completed eager leases may be
reused; real-row vectors always belong to the current batch.
"""

from contextlib import contextmanager
from types import SimpleNamespace


def warmup_eager_scratch(model, plan, *, capacities, dtype, device):
    """Allocate/initialize short and long eager scratch for all DP roles.

    No model, attention, KV access or collective is executed here. Device
    kernel/communication warmup remains the graph/server warmup's job.
    """
    import torch

    from sglang.srt.layers.dp_attention import DpPaddingMode
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardMode,
        build_welm_dp_moe_row_layout,
    )

    attn_tp_size = int(plan.attn_tp_size)
    dp_size = int(plan.outer_target_dp_size)
    if (attn_tp_size, dp_size) not in ((2, 2), (1, 4)):
        raise ValueError(
            "WeLM scratch warmup requires attention TP2 x DP2 or TP1 x DP4"
        )
    pool = model.welm_dp_scratch_pool
    executor = model.layers[0]._welm_dp_executor
    for capacity in sorted(set(capacities)):
        capacity = (
            (max(attn_tp_size, capacity) + attn_tp_size - 1)
            // attn_tp_size
            * attn_tp_size
        )
        for request_capacity in (1, 16):
            for mask in range(1, 1 << dp_size):
                tokens = [capacity if mask & (1 << d) else 0 for d in range(dp_size)]
                requests = [min(request_capacity, t) for t in tokens]
                layout = build_welm_dp_moe_row_layout(
                    raw_rows_cpu=tokens,
                    request_rows_cpu=requests,
                    local_dp_rank=plan.outer_target_dp_rank,
                    device=device,
                    padding_mode=DpPaddingMode.MAX_LEN,
                    current_slot_alignment=attn_tp_size,
                )
                batch = SimpleNamespace(
                    welm_dp_moe_row_layout=layout,
                    is_extend_in_batch=True,
                    forward_mode=ForwardMode.EXTEND,
                    enable_kv_mirror=plan.first_mirror_consumer is not None,
                )
                # Only width/dtype/device are read; avoid allocating fake token
                # embeddings or touching the business request/KV pools.
                like = torch.empty(
                    0, model.config.hidden_size, dtype=dtype, device=device
                )
                with pool.scope(batch):
                    executor.prepare_forward_scratch(
                        forward_batch=batch, hidden_states=like
                    )
                # Startup only: leases must be available before reuse/serving.
                for entry in pool.entries:
                    entry["event"].synchronize()


class WelmDpScratchPool:
    def __init__(self, event_factory, max_entries=8):
        self.event_factory = event_factory
        self.max_entries = max_entries
        self.entries = []

    def acquire(self, key):
        for entry in self.entries:
            if entry["key"] == key and not entry["busy"] and entry["event"].query():
                entry["busy"] = True
                return entry
        # In-flight entries remain strongly owned until their events finish.
        # The cache limit applies to retained completed entries, not live work.
        for entry in list(self.entries):
            if len(self.entries) < self.max_entries:
                break
            if not entry["busy"] and entry["event"].query():
                self.entries.remove(entry)
        entry = dict(key=key, tensors={}, busy=True, event=None)
        self.entries.append(entry)
        return entry

    @contextmanager
    def scope(self, batch):
        lease = ScratchLease(self)
        previous = getattr(batch, "welm_dp_scratch_lease", None)
        batch.welm_dp_scratch_lease = lease
        try:
            yield
        finally:
            batch.welm_dp_scratch_lease = previous
            lease.release()


class ScratchLease:
    def __init__(self, pool):
        self.pool = pool
        self.bindings = []

    def bind(self, row_view, key):
        entry = self.pool.acquire(key)
        for name in list(vars(row_view)):
            if name.startswith("_"):
                setattr(row_view, name, None)
        for name, value in entry["tensors"].items():
            setattr(row_view, name, value)
        self.bindings.append((row_view, entry))

    def capture(self, row_view):
        entry = next(entry for view, entry in self.bindings if view is row_view)
        entry["tensors"] = {
            name: value
            for name, value in vars(row_view).items()
            if name.startswith("_") and value is not None
        }

    def release(self):
        if not self.bindings:
            return
        event = self.pool.event_factory()
        event.record()
        for _, entry in self.bindings:
            entry["event"] = event
            entry["busy"] = False
