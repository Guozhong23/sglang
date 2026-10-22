# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""Fixed-capacity WeLM attention TP2 x DP2 target prefill graphs.

The scheduler/runner owns the unanimous admission ticket. This adapter never
votes, and never infers real rows from generic MAX-padded token counters.
Prompt[Tcap] and Mirror[Bcap] retain independent physical row views and share
only the fixed-address live T/B vectors. Device capture still requires NPU
validation; the host contracts do not establish HCCL or kernel correctness.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    build_welm_dp_moe_row_layout,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner.welm_prefill_graph import WelmPrefillGraphAdapter
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def validate_dp_graph_rows(token_rows, request_rows, token_capacity, request_capacity):
    """CPU-only validation, suitable for the shared pre-dispatch vote."""
    tokens = tuple(int(x) for x in token_rows)
    requests = tuple(int(x) for x in request_rows)
    if len(tokens) != 2 or len(requests) != 2:
        raise ValueError("WeLM DP prefill requires exactly two T/B row counts")
    if token_capacity <= 0 or token_capacity % 2 or request_capacity <= 0:
        raise ValueError("WeLM DP prefill requires positive, TP2-aligned token slots")
    if any(
        not 0 < b <= request_capacity or not b <= t <= token_capacity
        for t, b in zip(tokens, requests)
    ):
        raise ValueError(
            "WeLM DP prefill requires nonempty EXTEND rows within both buckets"
        )
    return tokens, requests


class WelmDpPrefillGraphAdapter(WelmPrefillGraphAdapter):
    is_dp = True
    kernel_variant = "bf16-normal-ordinary-oproj"

    def __init__(self, runner):
        parallel = get_parallel()
        if (
            not parallel.enable_dp_attention
            or parallel.attn_dp_size != 2
            or parallel.attn_tp_size != 2
            or parallel.tp_size != 4
        ):
            raise ValueError(
                "WeLM DP prefill graph requires attention TP2 x DP2 on four ranks"
            )
        if runner.quant_config is not None:
            raise ValueError(
                "WeLM DP prefill graph currently requires unquantized BF16"
            )
        self.dp_size = int(parallel.attn_dp_size)
        self.dp_rank = int(parallel.attn_dp_rank)
        self.attn_tp_size = int(parallel.attn_tp_size)
        self.attn_tp_rank = int(parallel.attn_tp_rank)
        self.ep_size = int(parallel.moe_ep_size)
        super().__init__(runner)
        self.prompt_real_rows_gpu = torch.zeros(
            2, dtype=torch.int32, device=self.device
        )
        self.request_real_rows_gpu = torch.zeros_like(self.prompt_real_rows_gpu)
        self.prompt_local_valid_rows = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self.request_local_valid_rows = torch.zeros_like(self.prompt_local_valid_rows)
        self._token_rows = (0, 0)
        self._request_rows = (0, 0)
        self._row_layouts = {}
        self._creating_capture = False
        logger.info(
            "WeLM TP2 x DP2 prefill graph: kernel=%s, MoE=%s; common T/B buckets, "
            "two nonempty EXTEND groups required; MegaMoE and fused/pipeline OProj "
            "remain eager-only. Visual/OE composition and logits stay eager.",
            self.kernel_variant,
            "DeepEP NORMAL AG / mirror local-sort AR" if self.ep else "TP",
        )

    def key(self, capacity, batch_size=None):
        return ShapeKey(
            size=capacity,
            variant_label=f"welm:dp2:tp2:bf16-normal:prompt:mirror={int(self.prune)}",
        )

    def mirror_key(self, batch_size):
        return ShapeKey(
            size=batch_size, variant_label="welm:dp2:tp2:bf16-normal:mirror"
        )

    def manifest(self):
        return {
            "kind": "welm-dp-prefill-v1",
            "kernel_variant": self.kernel_variant,
            "dp_size": self.dp_size,
            "attn_tp_size": self.attn_tp_size,
            "ep_size": self.ep_size,
            "prompt_capacities": sorted(
                k.size for k in self.capture_keys if k == self.key(k.size)
            ),
            "request_capacities": sorted(
                k.size for k in self.capture_keys if k == self.mirror_key(k.size)
            ),
            "max_requests": self.max_requests,
            "mirror": self.prune,
            "first_mirror": self.first_mirror,
            "layer_count": len(self.model.layers),
            "eager_input_embeddings": self.eager_input_embeddings,
        }

    def local_can_run(self, batch, token_capacity, request_capacity):
        """No collective and no graph execution: called before unanimous admission."""
        if batch.forward_mode != ForwardMode.EXTEND:
            return self.reject("DP graph only captures two ordinary EXTEND groups")
        if token_capacity % self.attn_tp_size:
            return self.reject("DP graph prompt bucket is not attention-TP aligned")
        if not 0 < batch.batch_size <= request_capacity <= self.max_requests:
            return self.reject("DP graph request bucket does not cover the live batch")
        if self.prune and self.mirror_key(request_capacity) not in self.capture_keys:
            return self.reject("common DP mirror request bucket was not captured")
        # load_batch stages every request tensor below before replay. Reject
        # incomplete metadata in the unanimous vote, never after peers commit
        # to executing the graph and its EP collectives.
        for name in (
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "seq_lens",
            "seq_lens_cpu",
            "req_pool_indices",
        ):
            value = getattr(batch, name, None)
            if value is None or value.dim() != 1 or value.numel() != batch.batch_size:
                return self.reject(f"DP graph request metadata shape mismatch: {name}")
        orig_seq_lens = getattr(batch, "orig_seq_lens", None)
        if orig_seq_lens is not None and (
            orig_seq_lens.dim() != 1 or orig_seq_lens.numel() != batch.batch_size
        ):
            return self.reject("DP graph request metadata shape mismatch: orig_seq_lens")
        if not super().can_run(batch, token_capacity):
            return False
        real_tokens = sum(batch.extend_seq_lens_cpu)
        for name in ("out_cache_loc", "positions"):
            value = getattr(batch, name, None)
            if value is None or value.dim() != 1 or value.numel() < real_tokens:
                return self.reject(
                    f"DP graph token metadata is shorter than real rows: {name}"
                )
        needed = [("prompt", token_capacity, self.key(token_capacity))]
        if self.prune:
            needed.append(
                ("mirror", request_capacity, self.mirror_key(request_capacity))
            )
        if any(
            (phase, capacity) not in self._row_layouts or key not in self.flash_metadata
            for phase, capacity, key in needed
        ):
            return self.reject("DP graph fixed row/Flash metadata is not captured")
        return True

    def can_run(self, batch, capacity):
        # All failure-producing checks belong in the shared pre-dispatch vote.
        # In particular, do not put a collective in this local EXTEND branch.
        ticket = getattr(batch, "welm_prefill_graph_ticket", None)
        return bool(
            ticket is not None
            and ticket.allowed
            and ticket.token_capacity == capacity
            and self.key(capacity) in self.capture_keys
            and (
                not self.prune
                or self.mirror_key(ticket.request_capacity) in self.capture_keys
            )
        )

    def _copy_live_rows(self, token_rows, request_rows):
        # Fresh immutable pinned staging prevents overwriting an in-flight H2D
        # source when large/small/large requests replay the same graph storage.
        self._token_rows = tuple(token_rows)
        self._request_rows = tuple(request_rows)
        for values, target in (
            (token_rows, self.prompt_real_rows_gpu),
            (request_rows, self.request_real_rows_gpu),
        ):
            staging = torch.tensor(
                values, dtype=torch.int32, device="cpu", pin_memory=True
            )
            target.copy_(staging, non_blocking=True)
        self.prompt_local_valid_rows.copy_(
            self.prompt_real_rows_gpu[self.dp_rank : self.dp_rank + 1]
        )
        self.request_local_valid_rows.copy_(
            self.request_real_rows_gpu[self.dp_rank : self.dp_rank + 1]
        )

    def _update_cpu_rows(self, view, rows):
        view.real_rows_cpu = list(rows)
        view.local_real_rows = int(rows[self.dp_rank])

    def _layout(self, phase, capacity, *, create=False):
        key = (phase, capacity)
        if key not in self._row_layouts:
            if not create:
                raise RuntimeError("WeLM DP graph row layout was not captured")
            mirror = phase == "mirror"
            if not mirror and capacity % self.attn_tp_size:
                raise ValueError("WeLM DP prompt capture bucket must be TP2 aligned")
            self._row_layouts[key] = build_welm_dp_moe_row_layout(
                raw_rows_cpu=list(self._request_rows if mirror else self._token_rows),
                request_rows_cpu=list(self._request_rows),
                local_dp_rank=self.dp_rank,
                device=self.device,
                padding_mode=DpPaddingMode.MAX_LEN,
                current_width=1,
                current_real_rows_gpu=(
                    self.request_real_rows_gpu if mirror else self.prompt_real_rows_gpu
                ),
                current_slot_rows_cpu=[capacity] * self.dp_size,
                request_real_rows_gpu=self.request_real_rows_gpu,
                request_slot_rows_cpu=[capacity if mirror else self.max_requests]
                * self.dp_size,
                current_slot_alignment=1 if mirror else self.attn_tp_size,
                request_slot_alignment=1,
            )
        layout = self._row_layouts[key]
        self._update_cpu_rows(
            layout.current,
            self._request_rows if phase == "mirror" else self._token_rows,
        )
        self._update_cpu_rows(layout.request, self._request_rows)
        return layout

    def _install_layout(self, batch, phase, capacity, *, create=False):
        layout = self._layout(phase, capacity, create=create)
        batch.welm_dp_moe_row_layout = layout
        batch.welm_dp_raw_rows_cpu = list(layout.current.real_rows_cpu)
        batch.global_num_requests_cpu = list(self._request_rows)
        batch.dp_padding_mode = DpPaddingMode.MAX_LEN
        batch.global_num_tokens_cpu = list(layout.current.slot_rows_cpu)
        batch.global_dp_buffer_len = layout.current.global_slot_rows
        batch.num_token_non_padded_cpu = layout.current.local_real_rows
        batch.is_extend_in_batch = True
        batch.global_forward_mode = ForwardMode.EXTEND
        return layout

    def _bind(self, batch, capacity):
        super()._bind(batch, capacity)
        row_capacity = batch.batch_size if self.capture_phase == "mirror" else capacity
        self._install_layout(
            batch, self.capture_phase, row_capacity, create=self._creating_capture
        )

    def prepare_capture(self, batch, capacity):
        # Dummy startup input is symmetric. Its constants are copied outside
        # capture; before_capture_forward never fills either live real vector.
        self._copy_live_rows((capacity,) * 2, (batch.batch_size,) * 2)
        self._creating_capture = True
        try:
            super().prepare_capture(batch, capacity)
        finally:
            self._creating_capture = False

    def before_capture_forward(self, batch):
        from sglang.srt.models.welmv4 import KVMirrorManager

        KVMirrorManager.activations_dict_kv.clear()
        phase = batch.welm_prefill_graph_phase
        capacity = batch.batch_size if phase == "mirror" else batch.positions.numel()
        key = self.mirror_key(capacity) if phase == "mirror" else self.key(capacity)
        self.current_flash_metadata = self.flash_metadata[key]
        self.current_flash_metadata.welm_flash_schedules.clear()
        batch.model_specific_states = None
        batch.__dict__.pop("custom_last_index", None)
        batch.welmv4_npu_deepep_scattered = False
        batch.welmv4_npu_deepep_full_mirror = False
        layout = self._install_layout(batch, phase, capacity)
        # These are physical generic slot counters only, never true T/B. The
        # two live vectors remain independent from these stage-local writes.
        if batch.global_num_tokens_gpu is not None:
            batch.global_num_tokens_gpu.fill_(capacity)
        source = (
            self.request_local_valid_rows
            if phase == "mirror"
            else self.prompt_local_valid_rows
        )
        if batch.num_token_non_padded is not None:
            batch.num_token_non_padded.copy_(source)
        batch.num_token_non_padded_cpu = layout.current.local_real_rows
        self.current_batch = batch

    def prepare_replay(self, live, static, capacity):
        ticket = getattr(live, "welm_prefill_graph_ticket", None)
        if not self.can_run(live, capacity):
            raise RuntimeError("WeLM DP replay requires the unanimous admission ticket")
        tokens, requests = validate_dp_graph_rows(
            ticket.token_rows, ticket.request_rows, capacity, ticket.request_capacity
        )
        if (
            tokens[self.dp_rank] != sum(live.extend_seq_lens_cpu)
            or requests[self.dp_rank] != live.batch_size
        ):
            raise RuntimeError(
                "WeLM DP admission rows changed before graph input staging"
            )
        self.capture_phase = "prompt"
        self._creating_capture = False
        self._copy_live_rows(tokens, requests)
        if self.prune:
            self._layout("mirror", ticket.request_capacity)
        super().prepare_replay(live, static, capacity)
        static.welm_prefill_graph_ticket = ticket
        self._install_layout(static, "prompt", capacity)
        # The base adapter's scalar is shared with pure-TP metadata. DP graph
        # heads restore independent prompt/request scalars instead.
        self.local_valid_rows.copy_(self.prompt_local_valid_rows)
        self.current_batch = static

    def _prepare_mirror_handoff(self, first, hidden_states, residual, batch):
        return first._welm_dp_executor.prepare_graph_mirror_handoff(
            layer=first,
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=batch,
            tail_indices=self.tail_indices,
            request_valid_mask=self.mirror_q_used.bool(),
        )

    def begin_mirror(self, batch):
        # The private capture batch has fixed Bcap, unlike the live serving
        # batch. DP executor handles the already-normalized first consumer and
        # reconstructs a partial residual only on the attention TP subgroup.
        bs = batch.batch_size
        self._install_layout(batch, "mirror", bs)
        return (
            self.mirror_hidden[:bs].clone(),
            self.mirror_residual[:bs].clone(),
            self.mirror_positions[:bs],
        )

    def replay(self, key, batch, **kwargs):
        ticket = getattr(batch, "welm_prefill_graph_ticket", None)
        if not self.can_run(batch, key.size) or key != self.key(ticket.token_capacity):
            raise RuntimeError("WeLM DP graph admission changed before replay")
        mirror_key = self.mirror_key(ticket.request_capacity) if self.prune else None
        # Both keys are verified BEFORE the prompt writes any business KV.
        if mirror_key is not None and mirror_key not in self.capture_keys:
            raise RuntimeError("WeLM DP mirror graph is missing before prompt replay")
        result = self.runner.backend.replay(key, batch, **kwargs)
        if mirror_key is not None:
            result = self.runner.backend.replay(mirror_key, batch, **kwargs)
        self.publish_state(key, batch)
        return result
