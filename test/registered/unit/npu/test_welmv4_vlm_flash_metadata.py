"""Real CPU tensor checks for VL lengths and the merged NPU Flash metadata ABI.

Vendor calls are recorded, not executed: device kernel/capture correctness is
covered by the separate manual Ascend tests.
"""

import ast
import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
BACKEND = SRT / "hardware_backend/npu/attention/ascend_backend.py"
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _execute(nodes, namespace):
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(BACKEND), "exec"), namespace)
    return namespace


def _backend(hybrid=True):
    def metadata(**kwargs):
        return SimpleNamespace(
            welm_flash_schedules={},
            welm_flash_mirror_q_lengths=None,
            **kwargs,
        )

    names = {
        "create_welm_prefill_graph_metadata",
        "prepare_welm_prefill_graph_metadata",
        "write_welm_prefill_graph_kv",
        "_forward_welm_flash_attention",
    }
    tree = ast.parse(BACKEND.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "AscendAttnBackend")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scatter = Mock()
    namespace = _execute(methods, {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_scatter_pa_kv_cache=scatter),
        "ForwardMetadata": metadata,
    })
    cache = torch.zeros(4 * 4096 + 128, 1, 8, dtype=torch.bfloat16)
    backend = SimpleNamespace(
        device="cpu", max_context_len=4096, page_size=64,
        is_hybrid_swa=hybrid, graph_mode=False,
        forward_metadata=object(), graph_metadata=object(),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(4 * 4096).reshape(4, 4096)
        ),
        full_to_swa_index_mapping=torch.arange(4 * 4096) + 128,
        token_to_kv_pool=SimpleNamespace(
            get_key_buffer=lambda _: cache, get_value_buffer=lambda _: cache,
        ),
        _is_swa_layer=lambda layer: hybrid and layer.sliding_window_size >= 0,
        _has_layerwise_sliding_window=lambda layer: layer.sliding_window_size >= 0,
        welm_flash_attn_mask=object(),
        _welm_flash_attn_metadata=Mock(return_value=object()),
        _welm_flash_attn=Mock(side_effect=lambda query, *_, **__: (query.clone(), None)),
        scatter=scatter,
    )
    for name in names:
        setattr(backend, name, MethodType(namespace[name], backend))
    return backend


def _batch(lengths, kv_lengths, rows):
    return SimpleNamespace(
        batch_size=len(lengths),
        seq_lens_cpu=torch.tensor(kv_lengths),
        seq_lens=torch.tensor(kv_lengths),
        extend_seq_lens=torch.tensor(lengths),
        req_pool_indices=torch.tensor(rows),
    )


def _layer(swa=False):
    return SimpleNamespace(
        layer_id=0, tp_q_head_num=2, tp_k_head_num=1, tp_v_head_num=1,
        qk_head_dim=8, v_head_dim=8, sliding_window_size=511 if swa else -1,
        scaling=8**-0.5,
    )


@pytest.mark.parametrize("arch", [
    "WeLMV4VLMForConditionalGeneration", "WeLMV4MoeForCausalLM",
    "WeLMV4MoeForCausalLMNextN", "OtherModel",
])
@pytest.mark.parametrize("enabled", [False, True])
def test_vl_attention_architecture_gate_preserves_flash_opt_in(arch, enabled):
    tree = ast.parse(BACKEND.read_text())
    fields = {"is_welm_v4", "use_welm_flash_attn"}
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(target, ast.Attribute) and target.attr in fields for target in n.targets
    )]
    backend = SimpleNamespace()
    _execute(nodes, {"self": backend, "architectures": [arch],
                     "get_bool_env_var": lambda *_: enabled})
    assert backend.is_welm_v4 == (arch != "OtherModel")
    assert backend.use_welm_flash_attn == (enabled and arch != "OtherModel")


@pytest.mark.parametrize("hybrid", [False, True])
def test_multi_image_request_token_lengths_update_without_stale_rows(hybrid):
    backend = _backend(hybrid)
    state = backend.create_welm_prefill_graph_metadata(4)
    eager, decode = backend.forward_metadata, backend.graph_metadata
    pointers = [x.data_ptr() for x in (
        state.block_tables, state.welm_flash_seqused_q, state.welm_flash_seqused_kv
    )]
    # VL images contribute token spans within a request, not metadata rows.
    # Two image-bearing prompts, then a one-token cached continuation, then
    # another longer batch reuse the same graph buffers.
    cases = [([2305, 769], [2370, 900], [2, 0]),
             ([1], [4096], [3]), ([1281, 513], [1400, 700], [0, 1])]
    for lengths, kv_lengths, rows in cases:
        backend.prepare_welm_prefill_graph_metadata(
            state, _batch(lengths, kv_lengths, rows)
        )
        pages = (max(kv_lengths) + 63) // 64
        expected = torch.zeros(4, 64, dtype=torch.int32)
        for i, row in enumerate(rows):
            expected[i, :pages] = row * 64 + torch.arange(pages)
        torch.testing.assert_close(state.block_tables, expected)
        if hybrid:
            expected[:len(rows), :pages] += 2
            torch.testing.assert_close(state.block_tables_swa, expected)
        else:
            assert state.block_tables_swa is None
        assert state.welm_flash_seqused_q.tolist() == lengths + [0] * (4 - len(rows))
        assert state.welm_flash_seqused_kv.tolist() == kv_lengths + [0] * (4 - len(rows))
        assert pointers == [x.data_ptr() for x in (
            state.block_tables, state.welm_flash_seqused_q, state.welm_flash_seqused_kv
        )]
        assert backend.forward_metadata is eager and backend.graph_metadata is decode


def test_capture_uses_reserved_page_and_capacity_error_precedes_writes():
    backend = _backend()
    state = backend.create_welm_prefill_graph_metadata(2)
    backend.req_to_token_pool = None
    backend.prepare_welm_prefill_graph_metadata(
        state, _batch([3, 7], [3, 7], [0, 1]), capture=True
    )
    assert not state.block_tables.any()
    before = state.welm_flash_seqused_q.clone()
    for bad in (_batch([1], [4097], [0]), _batch([1] * 3, [1] * 3, [0, 1, 2])):
        with pytest.raises(ValueError, match="capacity exceeded"):
            backend.prepare_welm_prefill_graph_metadata(state, bad)
        torch.testing.assert_close(state.welm_flash_seqused_q, before)


@pytest.mark.parametrize("swa", [False, True])
def test_graph_writer_preserves_negative_slots_and_uses_paged_bf16_layout(swa):
    backend = _backend()
    kv = torch.ones(5, 1, 8, dtype=torch.bfloat16)
    full = torch.tensor([64, 65, 66, -1, -1])
    sliding = torch.tensor([192, 193, 194, -1, -1])
    backend.write_welm_prefill_graph_kv(_layer(swa), kv, kv, full, sliding)
    args = backend.scatter.call_args
    assert args.kwargs == {"cache_mode": "Norm"}
    assert args.args[2].shape[1:] == (64, 1, 8)
    assert args.args[0].dtype == torch.bfloat16
    torch.testing.assert_close(args.args[4], (sliding if swa else full).to(torch.int32))


@pytest.mark.parametrize("graph", [False, True])
def test_flash_graph_uses_private_lengths_and_eager_restores_padded_output(graph):
    backend = _backend()
    state = backend.create_welm_prefill_graph_metadata(4)
    backend.prepare_welm_prefill_graph_metadata(state, _batch([3, 2], [70, 130], [0, 1]))
    state.welm_flash_cu_seqlens_q = torch.tensor([0, 3, 5, 5, 8], dtype=torch.int32)
    original = backend.forward_metadata
    if not graph:
        state.extend_seq_lens_cpu_int = torch.tensor([3, 2])
        state.welm_flash_max_seqlen_q = 3
        backend.forward_metadata = state
    query = torch.ones(8, 2, 8, dtype=torch.bfloat16)
    cache = backend.token_to_kv_pool.get_key_buffer(0)
    output = backend._forward_welm_flash_attention(
        query, cache, cache, _layer(), torch.ones(2, dtype=torch.bfloat16),
        **({"graph_metadata": state} if graph else {}),
    )
    args = backend._welm_flash_attn.call_args
    assert args.args[0].shape[0] == (8 if graph else 5)
    assert args.kwargs["max_seqlen_q"] == (-1 if graph else 3)
    assert args.kwargs["seqused_q"] is state.welm_flash_seqused_q
    assert args.kwargs["sinks"].dtype == torch.float32
    assert output.shape == (8, 16)
    if graph:
        assert backend.forward_metadata is original
    else:
        assert torch.all(output[:5] == 1) and not output[5:].any()
    assert not backend.graph_mode


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
