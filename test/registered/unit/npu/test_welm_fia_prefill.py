"""CPU numerical contracts for the opt-in eager WeLM FIA prefill path.

The paged FIA stand-in computes real causal/GQA attention without sinks. An
independent token/head reference adds an explicit zero-value sink. This checks
lengths, page selection, layouts, padding and sink math, not Ascend kernel support
or device numerical/performance characteristics.
"""

import ast
import copy
import importlib.util
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/welm_fia_prefill.py"
SPEC = importlib.util.spec_from_file_location("welm_fia_prefill_cpu_test", SOURCE)
FIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIA)


class CpuPagedFia:
    """FIA-shaped callable; reads only the supplied physical KV pages."""

    def __init__(self):
        self.calls = []

    def __call__(self, **args):
        self.calls.append(args)
        assert args["learnable_sink"] is None
        assert args["query"].is_contiguous()
        layout = args["input_layout"]
        q = args["query"]
        q_heads = args["num_query_heads"]
        kv_heads = args["num_key_value_heads"]
        assert q_heads % kv_heads == 0
        assert args["key"].shape[1] == args["block_size"]
        assert isinstance(args["actual_seq_qlen"], list)
        assert isinstance(args["actual_seq_kvlen"], list)
        if layout == "TND":
            cumulative = args["actual_seq_qlen"]
            lengths = [b - a for a, b in zip([0] + cumulative, cumulative)]
            assert sum(lengths) == q.shape[0]
            parts = q.split(lengths)
            assert args["atten_mask"].ndim == 2
        else:
            assert layout == "BSND" and q.shape[0] == 1
            lengths = args["actual_seq_qlen"]
            assert lengths == [q.shape[1]]
            parts = [q[0]]
            assert args["atten_mask"].ndim == 3
        assert all(length > 0 for length in lengths)
        assert len(parts) == len(args["actual_seq_kvlen"])
        outputs, lses = [], []
        for index, (part, kv_len) in enumerate(zip(parts, args["actual_seq_kvlen"])):
            blocks = math.ceil(kv_len / args["block_size"])
            physical = args["block_table"][index, :blocks].long()
            assert torch.all(physical >= 0)
            keys = args["key"][physical].reshape(-1, kv_heads, part.shape[-1])[:kv_len]
            values = args["value"][physical].reshape(
                -1, kv_heads, args["value"].shape[-1] // kv_heads
            )[:kv_len]
            keys = keys.repeat_interleave(q_heads // kv_heads, dim=1)
            values = values.repeat_interleave(q_heads // kv_heads, dim=1)
            scores = torch.einsum("qhd,khd->hqk", part.double(), keys.double())
            scores *= args["softmax_scale"]
            # FIA sparse_mode 3/4 aligns the last query with the last KV row.
            query_pos = torch.arange(kv_len - part.shape[0], kv_len)
            key_pos = torch.arange(kv_len)
            visible = key_pos[None, :] <= query_pos[:, None]
            if args["sparse_mode"] == 4:
                assert args["next_tokens"] == 0
                visible &= key_pos[None, :] >= (
                    query_pos[:, None] - args["pre_tokens"]
                )
            else:
                assert args["sparse_mode"] == 3
            scores = scores.masked_fill(~visible[None], -torch.inf)
            output = torch.einsum(
                "hqk,khd->qhd", scores.softmax(-1), values.double()
            ).to(q.dtype)
            outputs.append(output)
            lses.append(scores.logsumexp(-1).transpose(0, 1).unsqueeze(-1).float())
        if layout == "TND":
            output, lse = torch.cat(outputs), torch.cat(lses)
        else:
            output = outputs[0].unsqueeze(0)
            lse = lses[0].transpose(0, 1).unsqueeze(0)
        return output, lse if args["return_softmax_lse"] else None


def make_case(q_lengths, prefixes, *, window=-1, padding=0, mirror=False, value_dim=6):
    generator = torch.Generator().manual_seed(1030)
    layer = NS(
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        qk_head_dim=6,
        v_head_dim=value_dim,
        scaling=1 / math.sqrt(6),
        sliding_window_size=window,
    )
    kv_lengths = [q_len + prefix for q_len, prefix in zip(q_lengths, prefixes)]
    page_size = 4
    pages_per_request = [math.ceil(length / page_size) for length in kv_lengths]
    pages = sum(pages_per_request) + 3
    # Unused entries are deliberately poisonous. Correct lengths must exclude
    # both page tails and DP padding, rather than relying on zero-filled memory.
    key = torch.full((pages, page_size, 2, 6), torch.nan)
    value = torch.full((pages, page_size, 2, value_dim), torch.nan)
    table = torch.full((len(q_lengths), max(pages_per_request, default=0)), -777)
    allocation = torch.randperm(pages - 1, generator=generator) + 1
    cursor = 0
    dense_keys, dense_values = [], []
    for index, length in enumerate(kv_lengths):
        keys = torch.randn(length, 2, 6, generator=generator)
        values = torch.randn(length, 2, value_dim, generator=generator) + index * 5
        dense_keys.append(keys)
        dense_values.append(values)
        for logical in range(pages_per_request[index]):
            physical = allocation[cursor].item()
            cursor += 1
            table[index, logical] = physical
            begin = logical * page_size
            count = min(page_size, length - begin)
            key[physical, :count] = keys[begin : begin + count]
            value[physical, :count] = values[begin : begin + count]
    rows = len(q_lengths) if mirror else sum(q_lengths) + padding
    query = torch.randn(rows, 4, 6, generator=generator)
    if mirror:
        for index, length in enumerate(kv_lengths):
            if length == 0:
                query[index].fill_(torch.nan)
    elif padding:
        query[-padding:].fill_(torch.nan)
    return NS(
        query=query, layer=layer, key=key, value=value, table=table,
        q_lengths=q_lengths, kv_lengths=kv_lengths, page_size=page_size,
        dense_keys=dense_keys, dense_values=dense_values, mirror=mirror,
        mask=torch.ones(64, 64, dtype=torch.bool).triu(1),
    )


def direct_attention(case, sinks):
    """Reference on original dense KV; explicit zero-value sink per Q head."""
    result = torch.zeros(
        case.query.shape[0], 4, case.layer.v_head_dim, dtype=case.query.dtype
    )
    offset = 0
    for request, kv_len in enumerate(case.kv_lengths):
        q_len = int(kv_len > 0) if case.mirror else case.q_lengths[request]
        for row in range(q_len):
            output_row = request if case.mirror else offset + row
            position = kv_len - q_len + row
            window = case.layer.sliding_window_size
            first = (
                max(0, position - window)
                if window is not None and window >= 0 else 0
            )
            for head in range(4):
                kv_head = head // 2
                keys = case.dense_keys[request][first : position + 1, kv_head].double()
                values = case.dense_values[request][
                    first : position + 1, kv_head
                ].double()
                logits = keys @ case.query[output_row, head].double()
                logits *= case.layer.scaling
                if sinks is not None:
                    logits = torch.cat((logits, sinks[head : head + 1].double()))
                    values = torch.cat((values, torch.zeros(1, values.shape[1])))
                result[output_row, head] = logits.softmax(0) @ values
        offset += q_len
    return result.flatten(1)


def run_case(case, *, use_tnd, sinks):
    operator = CpuPagedFia()
    before = [value.clone() for value in (case.key, case.value, case.table)]
    output = FIA.forward_fia_prefill(
        case.query.flatten(1), case.key, case.value, layer=case.layer,
        page_size=case.page_size, block_table=case.table,
        q_lens=None if case.mirror else torch.tensor(case.q_lengths, dtype=torch.int64),
        kv_lens=torch.tensor(case.kv_lengths, dtype=torch.int64),
        atten_mask=case.mask, fia=operator, use_tnd=use_tnd, sinks=sinks,
        mirror_prefill=case.mirror,
    )
    torch.testing.assert_close(
        output, direct_attention(case, sinks), rtol=2e-6, atol=3e-6
    )
    assert torch.isfinite(output).all()
    for old, current in zip(before, (case.key, case.value, case.table)):
        torch.testing.assert_close(current, old, equal_nan=True)
    return output, operator.calls


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sink_lse_matches_explicit_zero_value_sink(layout, dtype):
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 3, 4, 7, generator=generator)
    values = torch.randn(2, 7, 4, 5, generator=generator)
    # Per-head logits exercise strong/weak sinks, not a shared scalar sink.
    sinks = torch.tensor([-50.0, -0.75, 2.25, 50.0])
    plain = torch.einsum("bshk,bkhd->bshd", logits.softmax(-1), values).to(dtype)
    with_sink = torch.cat((logits, sinks[None, None, :, None].expand(2, 3, -1, 1)), -1)
    zero_value = torch.cat((values, torch.zeros(2, 1, 4, 5)), 1)
    expected = torch.einsum("bshk,bkhd->bshd", with_sink.softmax(-1), zero_value)
    if layout == "TND":
        output = plain.reshape(6, 4, 5)
        lse = logits.logsumexp(-1).reshape(6, 4, 1)
        expected = expected.reshape_as(output)
    else:
        output = plain
        lse = logits.logsumexp(-1).transpose(1, 2).unsqueeze(-1)
    actual = FIA.apply_sink_lse(output, lse, sinks, layout)
    assert actual.dtype == dtype
    tolerance = 0.008 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual.float(), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("use_tnd", [True, False])
@pytest.mark.parametrize("window", [None, -1, 0, 3])
@pytest.mark.parametrize("with_sink", [False, True])
def test_mixed_prefill_prefixes_padding_and_q1(use_tnd, window, with_sink):
    case = make_case([4, 1, 3], [0, 9, 2], window=window, padding=3)
    sinks = torch.tensor([-2.0, 0.1, 1.5, 4.0]) if with_sink else None
    output, calls = run_case(case, use_tnd=use_tnd, sinks=sinks)
    assert torch.count_nonzero(output[-3:]) == 0
    assert len(calls) == (1 if use_tnd else 3)
    assert all(call["return_softmax_lse"] == with_sink for call in calls)


@pytest.mark.parametrize("use_tnd", [True, False])
@pytest.mark.parametrize("window", [-1, 2])
def test_zero_query_requests_are_filtered_without_moving_packed_rows(use_tnd, window):
    case = make_case([0, 3, 0, 1], [11, 2, 0, 8], window=window, padding=2)
    _, calls = run_case(case, use_tnd=use_tnd, sinks=torch.tensor([-3., 0., 1., 3.]))
    observed = [length for call in calls for length in call["actual_seq_kvlen"]]
    assert observed == [5, 9]
    assert len(calls) == (1 if use_tnd else 2)


@pytest.mark.parametrize("use_tnd", [True, False])
@pytest.mark.parametrize("window", [-1, 8])
def test_mirror_q1_long_kv_and_inactive_request_restore(use_tnd, window):
    case = make_case([1, 0, 1, 1], [256, 0, 16, 0], window=window, mirror=True)
    output, calls = run_case(
        case, use_tnd=use_tnd, sinks=torch.tensor([-2., 0., 1., 2.])
    )
    assert torch.count_nonzero(output[1]) == 0
    assert [length for call in calls for length in call["actual_seq_kvlen"]] == [
        257, 17, 1
    ]


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("lengths,prefixes", [([], []), ([0, 0], [0, 0])])
def test_empty_and_all_inactive_requests_never_call_fia(mirror, lengths, prefixes):
    case = make_case(lengths, prefixes, padding=0 if mirror else 3, mirror=mirror)
    output, calls = run_case(case, use_tnd=True, sinks=None)
    assert not calls
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("overrides,match", [
    ({"use_fia": False}, "ASCEND_USE_FIA"),
    ({"prefill_backend": "breakable"}, "eager prefill"),
    ({"prefill_backend": "full"}, "eager prefill"),
    ({"attn_cp_size": 2}, "attention CP"),
])
def test_config_rejects_unavailable_or_graph_and_cp_paths(overrides, match):
    kwargs = dict(use_fia=True, prefill_backend="disabled", attn_cp_size=1)
    FIA.validate_fia_prefill_config(**kwargs)
    with pytest.raises(ValueError, match=match):
        FIA.validate_fia_prefill_config(**(kwargs | overrides))


@pytest.mark.parametrize("field,value,match", [
    ("q_lens", [2], "authoritative CPU"),
    ("kv_lens", torch.tensor([[3]]), "one-dimensional"),
    ("q_lens", torch.tensor([-1]), "Invalid"),
    ("kv_lens", torch.tensor([2.5]), "Invalid"),
    ("kv_lens", torch.empty(1, device="meta"), "authoritative CPU"),
    ("q_lens", torch.tensor([4]), "Q length <= KV length"),
    ("q_lens", torch.tensor([3]), "Q rows/page tables"),
    ("q_lens", torch.tensor([1, 1]), "Q length <= KV length"),
    ("block_table", torch.zeros(2, 1, dtype=torch.int64), "Q rows/page tables"),
    ("sinks", torch.zeros(3), "sink count"),
])
def test_invalid_metadata_fails_before_operator(field, value, match):
    case = make_case([2], [1])
    operator = Mock(side_effect=AssertionError("Invalid input reached FIA"))
    kwargs = dict(
        layer=case.layer, page_size=4, block_table=case.table,
        q_lens=torch.tensor([2]), kv_lens=torch.tensor([3]),
        atten_mask=case.mask, fia=operator, use_tnd=True,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        FIA.forward_fia_prefill(case.query, case.key, case.value, **kwargs)
    operator.assert_not_called()


def test_sink_lse_rejects_unknown_layout():
    with pytest.raises(ValueError, match="layout"):
        FIA.apply_sink_lse(
            torch.ones(1, 4, 5), torch.zeros(1, 4, 1), torch.zeros(4), "BNSD"
        )


@pytest.mark.parametrize("use_tnd", [True, False])
def test_different_qk_value_dimensions_without_sink_and_sink_guard(use_tnd):
    case = make_case([3, 1], [2, 4], value_dim=5)
    run_case(case, use_tnd=use_tnd, sinks=None)
    operator = Mock(side_effect=AssertionError("Unsupported LSE reached operator"))
    with pytest.raises(ValueError, match="equal QK/V head dimensions"):
        FIA.forward_fia_prefill(
            case.query, case.key, case.value, layer=case.layer, page_size=4,
            block_table=case.table, q_lens=torch.tensor(case.q_lengths),
            kv_lens=torch.tensor(case.kv_lengths), atten_mask=case.mask,
            fia=operator, use_tnd=use_tnd, sinks=torch.zeros(4),
        )
    operator.assert_not_called()


BACKEND_SOURCE = SOURCE.with_name("ascend_backend.py")


def backend_method(name):
    tree = ast.parse(BACKEND_SOURCE.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    return copy.deepcopy(next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ))


def execute_backend_node(node, namespace):
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], node],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(module), str(BACKEND_SOURCE), "exec"),
        namespace,
    )
    return namespace[node.name]


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", [
    "EXTEND", "MIXED", "DECODE", "TARGET_VERIFY", "DRAFT_EXTEND"
])
def test_backend_dispatch_prefers_fia_for_enabled_prefill(mirror, enabled, mode):
    method = backend_method("_forward_extend_kv_mirror" if mirror else "forward_extend")

    def directly_returns_call(node, name):
        if not isinstance(node, ast.If) or len(node.body) != 1:
            return False
        statement = node.body[0]
        return (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == name
        )

    # Execute both adjacent production route predicates/returns in source order.
    # The unrelated KV writer and vendor fallbacks need not be mocked to test
    # which of these two prefill implementations receives a request.
    branches = [
        node for node in ast.walk(method)
        if directly_returns_call(node, "_forward_welm_fia_prefill")
        or directly_returns_call(node, "_forward_welm_flash_attention")
    ]
    assert len(branches) == 2
    branches.sort(key=lambda node: node.lineno)
    route = ast.parse(
        "def route(self, q, k_cache, v_cache, layer, forward_batch, sinks):\n"
        "    return 'legacy'\n"
    ).body[0]
    route.body = branches + route.body
    modes = NS(**{name: index for index, name in enumerate(
        ["EXTEND", "MIXED", "DECODE", "TARGET_VERIFY", "DRAFT_EXTEND"]
    )})
    fn = execute_backend_node(route, {"ForwardMode": modes})
    backend = NS(
        use_welm_fia_prefill=enabled, use_welm_flash_attn=True,
        _forward_welm_fia_prefill=Mock(return_value="fia"),
        _forward_welm_flash_attention=Mock(return_value="flash"),
    )
    selected = fn(backend, object(), object(), object(), object(),
                  NS(forward_mode=getattr(modes, mode)), None)
    expected = "fia" if enabled and mode in ("EXTEND", "MIXED") else "flash"
    assert selected == expected
    fia_method = backend._forward_welm_fia_prefill
    flash_method = backend._forward_welm_flash_attention
    called = fia_method if expected == "fia" else flash_method
    assert called.call_args.kwargs.get("mirror_prefill", False) == mirror
    other = flash_method if expected == "fia" else fia_method
    other.assert_not_called()


@pytest.mark.parametrize("swa_pool", [False, True])
@pytest.mark.parametrize("window", [-1, 7])
@pytest.mark.parametrize("mirror", [False, True])
def test_backend_wrapper_uses_physical_pool_kind_and_authoritative_cpu_lengths(
    monkeypatch, swa_pool, window, mirror
):
    helper = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules, "sglang.srt.hardware_backend.npu.attention.welm_fia_prefill",
        NS(forward_fia_prefill=helper),
    )
    operator = object()
    wrapper = execute_backend_node(
        backend_method("_forward_welm_fia_prefill"),
        {"torch_npu": NS(npu_fused_infer_attention_score_v2=operator)},
    )
    metadata = NS(
        block_tables=object(), block_tables_swa=object(),
        extend_seq_lens_cpu_int=torch.tensor([3, 1]),
        seq_lens_cpu_int=torch.tensor([15, 25]),
    )
    backend = NS(
        graph_mode=False, forward_metadata=metadata, page_size=128,
        _is_swa_layer=lambda layer: swa_pool, _can_use_tnd=lambda layer: True,
        fia_mask=object(),
    )
    output = wrapper(
        backend, object(), object(), object(), NS(sliding_window_size=window),
        object(), mirror_prefill=mirror,
    )
    assert output is helper.return_value
    args = helper.call_args.kwargs
    assert args["block_table"] is (
        metadata.block_tables_swa if swa_pool else metadata.block_tables
    )
    assert args["q_lens"] is metadata.extend_seq_lens_cpu_int
    assert args["kv_lens"] is metadata.seq_lens_cpu_int
    assert args["mirror_prefill"] == mirror
    assert args["fia"] is operator
    helper.reset_mock()
    backend.graph_mode = True
    with pytest.raises(RuntimeError, match="eager execution"):
        wrapper(backend, object(), object(), object(), object(), None)
    helper.assert_not_called()
