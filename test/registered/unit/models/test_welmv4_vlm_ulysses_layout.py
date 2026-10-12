"""CPU tests of the production Ulysses layouts, without importing SGLang."""

import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
spec = importlib.util.spec_from_file_location(
    "vision_ulysses_layout",
    ROOT / "python/sglang/srt/layers/welmv4_vision_ulysses.py",
)
ULYSSES = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ULYSSES
spec.loader.exec_module(ULYSSES)
Plan = ULYSSES.VisionUlyssesPlan


def exchange_all(send):
    # Independent model of equal-count AllToAll: dst receives its chunk from
    # every src, in src order. Do NOT implement this with production permutes.
    return [torch.stack([sender[dst] for sender in send]) for dst in range(len(send))]


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 3, 4, 7, 17, 28, 690])
def test_exact_sequence_head_round_trip(world, tokens):
    heads, dim = 16, 4
    full = torch.arange(tokens * heads * dim).reshape(tokens, heads, dim).float()
    plans = [Plan(tokens, heads, world, rank) for rank in range(world)]
    send = [
        p.pack_qkv(*(p.local_slice(full + i * 100000) for i in range(3))) for p in plans
    ]
    received = exchange_all(send)
    outputs = []
    for rank, (p, data) in enumerate(zip(plans, received)):
        qkv = p.unpack_qkv(data)
        for i, tensor in enumerate(qkv):
            expected = (full + i * 100000)[
                :, rank * p.local_heads : (rank + 1) * p.local_heads
            ]
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
            assert tensor.is_contiguous()
        outputs.append(p.pack_output(qkv[0]))
    for p, data in zip(plans, exchange_all(outputs)):
        torch.testing.assert_close(
            p.unpack_output(data), p.local_slice(full), atol=0, rtol=0
        )


@pytest.mark.parametrize("lengths", [[1], [3, 2, 6], [4, 4, 4], [1, 8, 2, 6]])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_segmented_attention_padding_never_enters_softmax(lengths, world):
    torch.manual_seed(6)
    n, heads, dim = sum(lengths), 8, 6
    full = [torch.randn(n, heads, dim) for _ in range(3)]

    def attention(qkv):
        result, start = [], 0
        for length in lengths:
            q, k, v = [x[start : start + length].transpose(0, 1) for x in qkv]
            result.append(F.scaled_dot_product_attention(q, k, v).transpose(0, 1))
            start += length
        return torch.cat(result)

    reference = attention(full)
    plans = [Plan(n, heads, world, rank) for rank in range(world)]
    # Poison padding: if even one padded key leaks into softmax this fails.
    send = [p.pack_qkv(*(p.local_slice(x, float("nan")) for x in full)) for p in plans]
    out = [
        p.pack_output(attention(p.unpack_qkv(x)))
        for p, x in zip(plans, exchange_all(send))
    ]
    local = [p.unpack_output(x) for p, x in zip(plans, exchange_all(out))]
    torch.testing.assert_close(torch.cat(local)[:n], reference, atol=1e-6, rtol=1e-5)


def test_group_contracts_and_single_rank_bypass():
    p = Plan(3, 4, 1, 0)
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    q = torch.randn(3, 4, 2)
    actual = p.sequence_to_heads(q, q, q, group)
    torch.testing.assert_close(actual[0], q)
    torch.testing.assert_close(p.heads_to_sequence(q, group), q)
    torch.testing.assert_close(p.gather_sequence(q, group), q)
    with pytest.raises(ValueError, match="group"):
        p.gather_sequence(q, SimpleNamespace(world_size=2, rank_in_group=0))


@pytest.mark.parametrize(
    "args", [(0, 8, 4, 0), (3, 7, 4, 0), (3, 8, 4, 4), (3, 8, 0, 0)]
)
def test_invalid_plans_fail(args):
    with pytest.raises(ValueError):
        Plan(*args)


def test_gather_strips_only_global_tail_and_preserves_merge_order():
    full = torch.arange(28 * 3).reshape(28, 3)
    plans = [Plan(28, 16, 8, rank) for rank in range(8)]
    shards = [p.local_slice(full, -999) for p in plans]
    for p, local in zip(plans, shards):
        group = SimpleNamespace(
            world_size=8,
            rank_in_group=p.rank,
            all_gather_into_tensor=lambda output, input: output.copy_(
                torch.cat(shards)
            ),
        )
        actual = p.gather_sequence(local, group)
        torch.testing.assert_close(
            actual.view(-1, 12), full.view(-1, 12), atol=0, rtol=0
        )
