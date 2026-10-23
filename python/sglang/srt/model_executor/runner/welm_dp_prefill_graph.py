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

import copy
import logging
from contextlib import nullcontext

import torch

from sglang.srt.environ import envs
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
        not ((t == b == 0) or (0 < b <= request_capacity and b <= t <= token_capacity))
        for t, b in zip(tokens, requests)
    ):
        raise ValueError(
            "WeLM DP prefill requires nonempty EXTEND rows within both buckets"
        )
    if not any(tokens):
        raise ValueError("WeLM DP prefill cannot replay an all-IDLE batch")
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
        self.active_mask = 3
        self.profile_ranges = envs.SGLANG_WELMV4_DP_PROFILE_RANGES.get()
        self.active_masks = (
            (1, 2, 3) if envs.SGLANG_WELMV4_DP_PREFILL_IDLE_GRAPH.get() else (3,)
        )
        plan = runner.model_runner.runner_parallel_plan
        self.variant_id = int(plan.fused_pack) + 2 * int(
            plan.replica_dp_group is not None
        )
        self.transport_variant = (
            "replica-ag" if plan.replica_dp_group is not None else "world-sum"
        )
        super().__init__(runner)
        self.idle_prompt_hidden = (
            torch.zeros(
                (self.max_tokens, self.model.config.hidden_size),
                dtype=runner.model_runner.dtype,
                device=self.device,
            )
            if len(self.active_masks) > 1
            else None
        )
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
            "static active masks=%s; MegaMoE and fused/pipeline OProj "
            "remain eager-only. Visual/OE composition and logits stay eager.",
            self.kernel_variant,
            "DeepEP NORMAL AG / mirror local-sort AR" if self.ep else "TP",
            self.active_masks,
        )

    def profile_range(self, name):
        if getattr(self, "profile_ranges", False):
            return torch.profiler.record_function(name)
        return nullcontext()

    def is_idle_role(self, batch=None):
        mask = getattr(
            batch, "welm_dp_graph_active_mask", getattr(self, "active_mask", 3)
        )
        return not bool(mask & (1 << self.dp_rank))

    def key(self, capacity, batch_size=None, *, active_mask=None):
        mask = getattr(self, "active_mask", 3) if active_mask is None else active_mask
        return ShapeKey(
            size=capacity,
            variant_label=f"welm:dp2:tp2:v{getattr(self, 'variant_id', 0)}:mask={mask}:prompt:mirror={int(self.prune)}",
        )

    def mirror_key(self, batch_size, *, active_mask=None):
        mask = getattr(self, "active_mask", 3) if active_mask is None else active_mask
        return ShapeKey(
            size=batch_size,
            variant_label=f"welm:dp2:tp2:v{getattr(self, 'variant_id', 0)}:mask={mask}:mirror",
        )

    def manifest(self):
        return {
            "kind": "welm-dp-prefill-v2",
            "active_masks": list(self.active_masks),
            "transport": self.transport_variant,
            "variant_id": self.variant_id,
            "kernel_variant": self.kernel_variant,
            "dp_size": self.dp_size,
            "attn_tp_size": self.attn_tp_size,
            "ep_size": self.ep_size,
            "prompt_capacities": sorted(
                {
                    k.size
                    for k in self.capture_keys
                    if any(
                        k == self.key(k.size, active_mask=m) for m in self.active_masks
                    )
                }
            ),
            "request_capacities": sorted(
                {
                    k.size
                    for k in self.capture_keys
                    if any(
                        k == self.mirror_key(k.size, active_mask=m)
                        for m in self.active_masks
                    )
                }
            ),
            "max_requests": self.max_requests,
            "mirror": self.prune,
            "first_mirror": self.first_mirror,
            "layer_count": len(self.model.layers),
            "eager_input_embeddings": self.eager_input_embeddings,
        }

    def local_can_run(self, batch, token_capacity, request_capacity, *, active_mask=3):
        """No collective and no graph execution: called before unanimous admission."""
        if active_mask not in getattr(self, "active_masks", (3,)):
            return self.reject("DP graph active role was not captured")
        prompt_key = self.key(token_capacity, active_mask=active_mask)
        mirror_key = self.mirror_key(request_capacity, active_mask=active_mask)
        if prompt_key not in self.capture_keys or (
            self.prune and mirror_key not in self.capture_keys
        ):
            return self.reject("DP graph role/bucket was not captured")
        if not active_mask & (1 << self.dp_rank):
            needed = [("prompt", token_capacity, prompt_key)]
            if self.prune:
                needed.append(("mirror", request_capacity, mirror_key))
            if any(
                (phase, size, active_mask) not in self._row_layouts
                or key not in self.flash_metadata
                for phase, size, key in needed
            ):
                return self.reject("IDLE graph metadata was not captured")
            return (
                batch.forward_mode == ForwardMode.IDLE
                and batch.batch_size == 0
                and not batch.return_logprob
                and batch.spec_info is None
                and batch.enable_kv_mirror == self.prune
            )
        if batch.forward_mode != ForwardMode.EXTEND:
            return self.reject("active DP graph role requires ordinary EXTEND")
        if token_capacity % self.attn_tp_size:
            return self.reject("DP graph prompt bucket is not attention-TP aligned")
        if not 0 < batch.batch_size <= request_capacity <= self.max_requests:
            return self.reject("DP graph request bucket does not cover the live batch")
        if self.prune and mirror_key not in self.capture_keys:
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
            return self.reject(
                "DP graph request metadata shape mismatch: orig_seq_lens"
            )
        self.active_mask = active_mask
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
            (phase, capacity, active_mask) not in self._row_layouts
            or key not in self.flash_metadata
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
            and self.key(capacity, active_mask=getattr(ticket, "active_mask", 3))
            in self.capture_keys
            and (
                not self.prune
                or self.mirror_key(
                    ticket.request_capacity,
                    active_mask=getattr(ticket, "active_mask", 3),
                )
                in self.capture_keys
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
        key = (phase, capacity, getattr(self, "active_mask", 3))
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
        batch.welm_dp_graph_active_mask = getattr(self, "active_mask", 3)
        super()._bind(batch, capacity)
        row_capacity = batch.batch_size if self.capture_phase == "mirror" else capacity
        self._install_layout(
            batch, self.capture_phase, row_capacity, create=self._creating_capture
        )

    def prepare_capture(self, batch, capacity):
        # Dummy startup input is symmetric. Its constants are copied outside
        # capture; before_capture_forward never fills either live real vector.
        self._copy_live_rows(
            tuple(capacity if self.active_mask & (1 << d) else 0 for d in range(2)),
            tuple(
                batch.batch_size if self.active_mask & (1 << d) else 0 for d in range(2)
            ),
        )
        self._creating_capture = True
        try:
            if self.is_idle_role():
                self._bind(batch, capacity)
                self.prepare_idle_graph_inputs(batch, capacity)
                key = (
                    self.mirror_key(batch.batch_size)
                    if self.capture_phase == "mirror"
                    else self.key(capacity)
                )
                metadata = copy.copy(self.flash_inputs)
                metadata.welm_flash_schedules = {}
                self.flash_metadata[key] = metadata
                self.current_flash_metadata = metadata
            else:
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
            tokens[self.dp_rank] != sum(live.extend_seq_lens_cpu or [])
            or requests[self.dp_rank] != live.batch_size
        ):
            raise RuntimeError(
                "WeLM DP admission rows changed before graph input staging"
            )
        self.active_mask = getattr(ticket, "active_mask", 3)
        self.capture_phase = "prompt"
        self._creating_capture = False
        self._copy_live_rows(tokens, requests)
        if self.prune:
            self._layout("mirror", ticket.request_capacity)
        if self.is_idle_role():
            self._bind(static, capacity)
            self.prepare_idle_graph_inputs(static, capacity)
        else:
            super().prepare_replay(live, static, capacity)
        static.welm_prefill_graph_ticket = ticket
        self._install_layout(static, "prompt", capacity)
        # The base adapter's scalar is shared with pure-TP metadata. DP graph
        # heads restore independent prompt/request scalars instead.
        self.local_valid_rows.copy_(self.prompt_local_valid_rows)
        self.current_batch = static

    def prepare_idle_graph_inputs(self, batch, capacity):
        self.valid_rows[:capacity].zero_()
        self.local_valid_rows.zero_()
        self.full_write_locs[:capacity].fill_(-1)
        self.swa_write_locs[:capacity].fill_(-1)
        self.tail_indices.zero_()
        self.mirror_q_used.zero_()
        self.oe_ids.zero_()
        batch.input_ids.zero_()
        batch.positions.zero_()
        batch.out_cache_loc.fill_(-1)
        batch.model_specific_states = None
        batch.__dict__.pop("custom_last_index", None)
        batch.welmv4_npu_deepep_scattered = False
        batch.welmv4_npu_deepep_full_mirror = False
        if batch.num_token_non_padded is not None:
            batch.num_token_non_padded.zero_()
        self.current_batch = batch

    def finish_prompt(self, hidden_states, residual, positions, batch):
        if not self.is_idle_role(batch):
            return super().finish_prompt(hidden_states, residual, positions, batch)
        if self.mirror_hidden is None:
            shape = (self.max_requests, hidden_states.shape[-1])
            self.mirror_hidden = hidden_states.new_empty(shape)
            self.mirror_residual = hidden_states.new_empty(shape, dtype=torch.float32)
        self.mirror_hidden.zero_()
        self.mirror_residual.zero_()
        self.mirror_positions.zero_()
        self.mirror_residual_is_partial = False
        batch.model_specific_states = None
        return None

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
        self.active_mask = getattr(ticket, "active_mask", 3)
        if not self.can_run(batch, key.size) or key != self.key(ticket.token_capacity):
            raise RuntimeError("WeLM DP graph admission changed before replay")
        mirror_key = self.mirror_key(ticket.request_capacity) if self.prune else None
        # Both keys are verified BEFORE the prompt writes any business KV.
        if mirror_key is not None and mirror_key not in self.capture_keys:
            raise RuntimeError("WeLM DP mirror graph is missing before prompt replay")
        with self.profile_range("welm.dp.prompt_replay"):
            result = self.runner.backend.replay(key, batch, **kwargs)
        if mirror_key is not None:
            with self.profile_range("welm.dp.mirror_replay"):
                result = self.runner.backend.replay(mirror_key, batch, **kwargs)
        self.publish_state(key, batch)
        return result
