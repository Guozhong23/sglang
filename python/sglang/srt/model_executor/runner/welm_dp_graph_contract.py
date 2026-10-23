# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Host-only admission contract for WeLM attention-TP2/DP2 prefill graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence


@dataclass(frozen=True)
class WelmPrefillGraphTicket:
    allowed: bool
    token_capacity: int = 0
    request_capacity: int = 0
    token_rows: tuple[int, ...] = ()
    request_rows: tuple[int, ...] = ()
    reason: str = ""
    active_mask: int = 0
    phases: tuple[int, ...] = ()
    kernel_variant: int = 0

    def vote_record(self, local_allowed: bool) -> tuple[int, ...]:
        # Fixed width even for missing metadata: every participating rank must
        # enter the same small CPU collective before dispatching a graph.
        tokens = self.token_rows if len(self.token_rows) == 2 else (0, 0)
        requests = self.request_rows if len(self.request_rows) == 2 else (0, 0)
        return (
            int(self.allowed and local_allowed),
            self.token_capacity,
            self.request_capacity,
            self.active_mask,
            self.kernel_variant,
            *(self.phases if len(self.phases) == 2 else (0, 0)),
            *tokens,
            *requests,
        )


def select_welm_dp_prefill_ticket(
    *,
    token_rows: Sequence[int] | None,
    request_rows: Sequence[int] | None,
    all_extend: bool,
    token_buckets: Sequence[int],
    request_buckets: Sequence[int],
    token_alignment: int = 2,
    phases: Sequence[int] | None = None,
    kernel_variant: int = 0,
) -> WelmPrefillGraphTicket:
    """Choose shared capacities from unpadded scheduler counts, never local T/B."""
    tokens = tuple(int(x) for x in token_rows) if token_rows is not None else ()
    requests = tuple(int(x) for x in request_rows) if request_rows is not None else ()
    modes = (
        tuple(int(x) for x in phases)
        if phases is not None
        else ((1, 1) if all_extend else (0, 0))
    )
    ticket = WelmPrefillGraphTicket(
        False,
        token_rows=tokens,
        request_rows=requests,
        phases=modes,
        kernel_variant=kernel_variant,
    )
    if len(tokens) != 2 or len(requests) != 2:
        return replace(ticket, reason="missing_dp_counts")
    # ForwardMode.EXTEND=1 and IDLE=4 are serialized by the scheduler.
    if len(modes) != 2 or any(mode not in (1, 4) for mode in modes):
        return replace(ticket, reason="mixed_phase")
    active_mask = 0
    for d, (t, b, mode) in enumerate(zip(tokens, requests, modes)):
        if mode == 1 and 0 < b <= t:
            active_mask |= 1 << d
        elif not (mode == 4 and t == b == 0):
            return replace(ticket, reason="invalid_real_counts")
    ticket = replace(ticket, active_mask=active_mask)
    if not active_mask:
        return replace(ticket, reason="all_idle")
    if not all_extend:
        return replace(ticket, reason="scheduler_ineligible")
    if token_alignment <= 0:
        raise ValueError("WeLM DP graph token alignment must be positive")
    tcap = next(
        (
            t
            for t in sorted(set(token_buckets))
            if t >= max(tokens) and t % token_alignment == 0
        ),
        0,
    )
    bcap = next((b for b in sorted(set(request_buckets)) if b >= max(requests)), 0)
    if not tcap or not bcap:
        return replace(ticket, reason="capacity")
    return replace(ticket, allowed=True, token_capacity=tcap, request_capacity=bcap)


def agree_welm_dp_prefill_ticket(
    ticket: WelmPrefillGraphTicket,
    votes: Sequence[Sequence[int]],
) -> WelmPrefillGraphTicket:
    """Require all four ranks to agree on eligibility, shape and real counts."""
    if len(votes) != 4 or any(len(vote) != 11 for vote in votes):
        raise ValueError("WeLM TP2DP2EP4 admission requires four fixed-width votes")
    expected = ticket.vote_record(True)
    if any(tuple(vote[1:]) != expected[1:] for vote in votes):
        return replace(ticket, allowed=False, reason="rank_metadata_mismatch")
    if not all(vote[0] == 1 for vote in votes):
        return replace(ticket, allowed=False, reason=ticket.reason or "rank_ineligible")
    if not ticket.allowed:
        return replace(ticket, reason=ticket.reason or "rank_ineligible")
    return ticket
