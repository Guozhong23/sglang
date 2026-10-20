"""CPU tests of production WeLM VL PD contracts, metadata and KV address paths.

Only serving imports and the physical transfer engine are isolated. NPU buffer
export, Ascend/Mooncake address planning, padding, metadata and OE run from the
production sources with real CPU tensors. These tests do not certify HCCL.
"""

import ast
import asyncio
import concurrent.futures
import copy
import dataclasses
import importlib.util
import json
import logging
import runpy
import sys
from array import array
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))["register_cpu_ci"](
    est_time=10, suite="base-a-test-cpu"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, *, owner=None, **scope):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner:
        nodes = next(node.body for node in nodes if getattr(node, "name", "") == owner)
    tree.body = ast.parse("from __future__ import annotations").body + [
        node for node in nodes if getattr(node, "name", "") in names
    ]
    exec(compile(tree, str(path), "exec"), scope)
    return NS(**scope)


PD = load(SRT / "disaggregation/welm_vlm.py", "welm_vlm_pd_test_helpers")
KVPoll = NS(Failed=0, Bootstrapping=1, WaitingForInput=2, Transferring=3, Success=4)
UTILS = definitions(
    SRT / "disaggregation/utils.py",
    {
        "MetadataBuffers",
        "_apply_metadata_gate",
        "_is_fake_transfer",
        "poll_and_all_reduce",
    },
    torch=torch,
    dist=torch.distributed,
    nullcontext=nullcontext,
    is_npu=lambda: False,
    envs=NS(SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NS(get=lambda: None)),
    PROMPT_SIGNATURE_SLICE=PD.PROMPT_SIGNATURE_SLICE,
    KVPoll=KVPoll,
    FAKE_BOOTSTRAP_HOST="2.2.2.2",
    _poll_with_failure_injection=lambda pollers: [p.poll() for p in pollers],
)
PAD = definitions(
    SRT / "managers/mm_utils.py",
    {"pad_input_tokens"},
    owner="MultiModalityDataPaddingPatternMultimodalTokens",
    torch=torch,
    defaultdict=defaultdict,
    Modality=NS(IMAGE="IMAGE", AUDIO="AUDIO", VIDEO="VIDEO"),
).pad_input_tokens


@pytest.fixture(autouse=True)
def dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.disaggregation.welm_vlm", PD)


def fixture(arch="WeLMMoonViTVLForConditionalGeneration"):
    name = "welm_moonvit_vl_config.json" if "Moon" in arch else "welmv4_vlm_config.json"
    raw = json.loads((ROOT / "test/registered/unit/models/fixtures" / name).read_text())
    return NS(architectures=[arch], to_dict=lambda: copy.deepcopy(raw))


export = definitions(
    SRT / "hardware_backend/npu/memory_pool_npu.py",
    {"get_contiguous_buf_infos"},
    owner="NPUMHATokenToKVPool",
).get_contiguous_buf_infos


class Pool:
    get_contiguous_buf_infos = export

    def __init__(self, layers, page_size=4, pages=16, fia=False):
        self.start_layer = 0
        self.layer_num = layers
        self.page_size = page_size
        self.head_num, self.head_dim, self.v_head_dim = 2, 4, 3
        self.use_fia = fia
        self.k_buffer = [
            torch.zeros(
                pages, page_size, self.head_num, self.head_dim, dtype=torch.bfloat16
            )
            for _ in range(layers)
        ]
        self.v_buffer = [
            torch.zeros(
                pages, page_size, self.head_num, self.v_head_dim, dtype=torch.bfloat16
            )
            for _ in range(layers)
        ]
        if fia:
            self.k_buffer = [
                t.view(-1, 1, self.head_num, self.head_dim) for t in self.k_buffer
            ]
            self.v_buffer = [
                t.view(-1, 1, self.head_num, self.v_head_dim) for t in self.v_buffer
            ]

    def get_key_buffer(self, i):
        return self.k_buffer[i]

    def get_value_buffer(self, i):
        return self.v_buffer[i]


def runtime(arch="WeLMMoonViTVLForConditionalGeneration", *, pages=16, fia=False):
    config = fixture(arch)
    text = config.to_dict()["text_config"]
    windows = text["sliding_window_size_layerwise"][: text["num_hidden_layers"]]
    full = [i for i, window in enumerate(windows) if window != 512]
    swa = [i for i, window in enumerate(windows) if window == 512]
    pool = NS(
        layers_mapping={
            **{gid: (i, False) for i, gid in enumerate(full)},
            **{gid: (i, True) for i, gid in enumerate(swa)},
        },
        full_kv_pool=Pool(len(full), pages=pages, fia=fia),
        swa_kv_pool=Pool(len(swa), pages=pages, fia=fia),
        page_size=4,
    )
    fp, fl, fi = pool.full_kv_pool.get_contiguous_buf_infos()
    sp, sl, si = pool.swa_kv_pool.get_contiguous_buf_infos()
    kv = NS(
        kv_data_ptrs=fp,
        kv_data_lens=fl,
        kv_item_lens=fi,
        state_data_ptrs=[sp],
        state_data_lens=[sl],
        state_item_lens=[si],
        state_types=["swa"],
        page_size=4,
        kv_cache_dtype_str="bfloat16",
        aux_item_lens=[64, 64, 64, 64, 512, 512, 64, 128, 4096, 64],
        prefill_start_layer=0,
    )
    scheduler = NS(
        model_config=NS(hf_config=config, dtype=torch.bfloat16, context_len=32768),
        token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
        ps=NS(attn_tp_size=4),
        server_args=NS(enable_kv_mirror=True, enable_over_encoding=True),
    )
    return scheduler, kv, pool


@pytest.mark.parametrize("arch", PD.WELM_VLM_ARCHITECTURES)
@pytest.mark.parametrize("fia", [False, True])
def test_contract_uses_real_buffer_layout_not_capacity_or_vision_parallelism(arch, fia):
    p, pkv, _ = runtime(arch, pages=16, fia=fia)
    d, dkv, _ = runtime(arch, pages=21, fia=fia)
    p.server_args.mm_enable_dp_encoder = True
    d.server_args.mm_enable_dp_encoder = False
    contract = PD.build_welm_vlm_pd_contract(p, pkv)
    peer = PD.build_welm_vlm_pd_contract(d, dkv)
    PD.validate_welm_vlm_pd_contract(contract, json.loads(json.dumps(peer)))
    assert sorted(contract["full_layers"] + contract["swa_layers"]) == list(range(48))
    assert len(contract["full_page_bytes"]) == 2 * len(contract["full_layers"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("attn_tp_size", 8),
        ("kv_dtype", "float16"),
        ("page_size", 128),
        ("kv_mirror", False),
        ("over_encoding", False),
        ("full_layers", [0]),
        ("swa_layers", [1]),
        ("full_page_bytes", [8]),
        ("swa_page_bytes", [16]),
        ("heads", [[2, 8, 8]]),
        ("config_digest", "different"),
        ("version", 99),
        ("aux_row_bytes", [128]),
    ],
)
def test_contract_rejects_incompatible_layout_before_address_exchange(key, value):
    scheduler, kv, _ = runtime()
    contract = PD.build_welm_vlm_pd_contract(scheduler, kv)
    peer = copy.deepcopy(contract)
    peer[key] = value
    with pytest.raises(ValueError, match=key):
        PD.validate_welm_vlm_pd_contract(contract, peer)


def test_contract_requires_both_workers_and_rejects_virtual_or_bad_strides():
    scheduler, kv, pool = runtime()
    contract = PD.build_welm_vlm_pd_contract(scheduler, kv)
    for p, d in ((None, contract), (contract, None)):
        with pytest.raises(ValueError, match="both workers"):
            PD.validate_welm_vlm_pd_contract(p, d)
    PD.validate_welm_vlm_pd_contract(None, None)
    kv.state_item_lens[0][0] += 1
    with pytest.raises(ValueError, match="contiguous"):
        PD.build_welm_vlm_pd_contract(scheduler, kv)
    pool.unified_buffer = object()
    with pytest.raises(ValueError, match="full/SWA"):
        PD.build_welm_vlm_pd_contract(scheduler, kv)


@pytest.mark.parametrize("change", ["dtype", "head-major", "strided", "address-order"])
def test_contract_rejects_unsupported_physical_buffer_layout(change):
    scheduler, kv, pool = runtime()
    sub = pool.full_kv_pool
    if change == "dtype":
        # Same byte width and declared KV dtype do not establish BF16 storage.
        sub.k_buffer[0] = sub.k_buffer[0].to(torch.float16)
    elif change == "head-major":
        sub.k_buffer[0] = sub.k_buffer[0].transpose(1, 2).contiguous()
    elif change == "strided":
        sub.k_buffer[0] = sub.k_buffer[0][::2]
    else:
        kv.kv_data_ptrs.reverse()
    with pytest.raises(ValueError, match="WeLM-VL PD"):
        PD.build_welm_vlm_pd_contract(scheduler, kv)


def test_config_digest_ignores_paths_but_preserves_mirror_oe_and_rope():
    config = fixture().to_dict()
    other = copy.deepcopy(config)
    other["_name_or_path"] = "/a/different/mount"
    other["text_config"]["_name_or_path"] = "/another/mount"
    other["transformers_version"] = "local-build"
    assert PD._digest(PD._semantic_config(config)) == PD._digest(
        PD._semantic_config(other)
    )
    other["text_config"]["kv_mirror_layers"] = [7]
    assert PD._digest(PD._semantic_config(config)) != PD._digest(
        PD._semantic_config(other)
    )


def request(count=2, duplicate=False):
    tokens, items = [11], []
    for i in range(count):
        grid = (1, 4 + 2 * (i % 2), 4)
        rows = int(np.prod(grid)) // 4
        start = len(tokens)
        tokens.extend([154752] * rows)
        items.append(
            NS(
                modality="IMAGE",
                hash=100 + (0 if duplicate else i),
                pad_value=1000100 + (0 if duplicate else i),
                image_grid_thw=torch.tensor([grid]),
                offsets=[(start, len(tokens) - 1)],
                is_image=lambda: True,
            )
        )
        tokens.append(12 + i)
    mm = NS(
        mm_items=items, im_token_id=154752, audio_token_id=None, video_token_id=None
    )
    mm.compute_mm_token_counts = lambda: (
        sum(b - a + 1 for it in mm.mm_items for a, b in it.offsets),
        0,
        0,
    )
    # Real production placeholder replacement, not a hand-written equivalent.
    expanded = PAD(None, tokens, mm)
    return NS(
        rid="pd-image",
        origin_input_ids=array("q", expanded),
        output_ids=array("q", [42]),
        multimodal_inputs=mm if count else None,
        finished=lambda: False,
        metadata_buffer_index=0,
        cached_tokens=7,
        cached_tokens_device=7,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        return_logprob=False,
        return_sampling_mask=False,
        hidden_states_tensor=None,
        bootstrap_room=123,
        bootstrap_host="127.0.0.1",
    )


def metadata(req):
    buf = UTILS.MetadataBuffers(1, 8, torch.bfloat16, max_sampling_mask_tokens=0)
    buf.set_buf(req)
    return buf


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 50])
@pytest.mark.parametrize("duplicate", [False, True])
def test_prompt_signature_and_metadata_round_trip(count, duplicate):
    p, d = request(count, duplicate), request(count, duplicate)
    cfg = NS(hf_config=fixture())
    PD.init_welm_vlm_pd_request(p, cfg)
    PD.init_welm_vlm_pd_request(d, cfg)
    assert p.welm_vlm_pd_signature == d.welm_vlm_pd_signature
    buf = metadata(p)
    assert tuple(buf.get_buf(0)[1][7:16].tolist()) == d.welm_vlm_pd_signature
    assert buf.cached_tokens[0, :4].tolist() == [7, 7, 0, 0]
    assert buf.cached_tokens[0, 4].item() == len(p.origin_input_ids) - 1 - count
    assert len(buf.get_buf_infos()[2]) == 10  # No new transfer buffer.
    del p.welm_vlm_pd_signature
    buf.set_buf(p)
    assert buf.cached_tokens[0, 7:16].tolist() == [0] * 9


@pytest.mark.parametrize("change", ["tokens", "order", "grid", "offset", "hash"])
def test_signature_catches_equal_length_semantic_mismatch(change):
    p, d = request(), request()
    if change == "tokens":
        d.origin_input_ids[-1] += 1
    elif change == "order":
        d.multimodal_inputs.mm_items.reverse()
    elif change == "grid":
        d.multimodal_inputs.mm_items[0].image_grid_thw = torch.tensor([[1, 2, 8]])
    elif change == "offset":
        d.multimodal_inputs.mm_items[0].offsets = [(2, 5)]
    else:
        d.multimodal_inputs.mm_items[0].hash += (
            2**30
        )  # Same modulo pad is insufficient.
    assert len(p.origin_input_ids) == len(d.origin_input_ids)
    assert PD.welm_vlm_prompt_signature(p) != PD.welm_vlm_prompt_signature(d)


@pytest.mark.parametrize(
    "state", ["valid", "mismatch", "old-peer", "not-ready", "fake"]
)
def test_metadata_validation_precedes_tp_consensus_and_decode_admission(
    monkeypatch, state
):
    p, d = request(), request()
    p.welm_vlm_pd_signature = PD.welm_vlm_prompt_signature(p)
    d.welm_vlm_pd_signature = PD.welm_vlm_prompt_signature(d)
    buf = metadata(p)
    if state == "mismatch":
        buf.cached_tokens[0, 9] += 1
    elif state == "old-peer":
        buf.cached_tokens[0, 7:16].zero_()
    elif state == "not-ready":
        buf.bootstrap_room.zero_()
    elif state == "fake":
        d.bootstrap_host = "2.2.2.2"
        buf.cached_tokens.zero_()
    receiver = NS(kv_mgr=NS(record_failure=Mock()), poll=lambda: KVPoll.Success)
    dr = NS(req=d, kv_receiver=receiver, metadata_buffer_index=0)
    observed = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, **kwargs: observed.append(tensor.tolist()),
    )
    result = UTILS.poll_and_all_reduce(
        [receiver],
        object(),
        decode_reqs=[dr],
        metadata_buffers=buf,
        server_args=NS(disaggregation_transfer_backend="ascend"),
    )
    expected = (
        KVPoll.Failed
        if state in ("mismatch", "old-peer")
        else (KVPoll.Transferring if state == "not-ready" else KVPoll.Success)
    )
    assert result == [expected] and observed == [[expected]]
    assert receiver.kv_mgr.record_failure.called == (expected == KVPoll.Failed)


GROUP = definitions(
    SRT / "disaggregation/common/utils.py", {"group_concurrent_contiguous"}, np=np
).group_concurrent_contiguous
TRANSFER_SCOPE = dict(
    concurrent=concurrent,
    group_concurrent_contiguous=GROUP,
    timed_component=lambda *a, **kw: lambda fn: fn,
    logger=logging.getLogger(__name__),
)
SEND = definitions(
    SRT / "disaggregation/ascend/conn.py",
    {"send_kvcache"},
    owner="AscendKVManager",
    **TRANSFER_SCOPE,
).send_kvcache
SEND_SWA = definitions(
    SRT / "disaggregation/mooncake/conn.py",
    {"_send_kvcache_generic"},
    owner="MooncakeKVManager",
    **TRANSFER_SCOPE,
)._send_kvcache_generic
MHA = definitions(
    SRT / "disaggregation/common/conn.py",
    {"get_mha_kv_ptrs_with_pp"},
    owner="CommonKVManager",
).get_mha_kv_ptrs_with_pp


@pytest.mark.parametrize("fia", [False, True])
@pytest.mark.parametrize("arch", PD.WELM_VLM_ARCHITECTURES)
def test_actual_npu_export_and_ascend_chunked_full_swa_transfer(fia, arch):
    p, pkv, pp = runtime(arch, fia=fia)
    d, dkv, dp = runtime(arch, pages=21, fia=fia)
    PD.validate_welm_vlm_pd_contract(
        PD.build_welm_vlm_pd_contract(p, pkv), PD.build_welm_vlm_pd_contract(d, dkv)
    )
    tensors = []
    for owner, pool in enumerate((pp, dp)):
        for sub in (pool.full_kv_pool, pool.swa_kv_pool):
            for i, tensor in enumerate(sub.k_buffer + sub.v_buffer):
                if owner == 0:
                    tensor.copy_(
                        torch.arange(tensor.numel()).reshape(tensor.shape) % 97 + i
                    )
                tensors.append(tensor)
    ranges = [(t.data_ptr(), t.view(torch.uint8).flatten()) for t in tensors]

    def region(ptr, length):
        for base, data in ranges:
            if base <= ptr and ptr + length <= base + data.numel():
                return data[ptr - base : ptr - base + length]
        raise AssertionError("Transfer escaped an allocated buffer")

    def engine(peer, src, dst, lengths):
        for a, b, size in zip(src, dst, lengths):
            region(b, size).copy_(region(a, size))
        return 0

    manager = NS(
        kv_args=pkv,
        pp_size=1,
        is_mla_backend=False,
        is_hybrid_mla_backend=False,
        enable_custom_mem_pool=False,
        engine=NS(batch_transfer_sync=engine),
    )
    manager.get_mha_kv_ptrs_with_pp = lambda *args: MHA(manager, *args)
    src, dst = np.array([2, 3, 7, 9, 10]), np.array([5, 8, 9, 13, 14])
    # Two prefill chunks; only the final SWA window is transferred.
    for part in (slice(0, 3), slice(3, 5)):
        assert SEND(manager, "D", src[part], dkv.kv_data_ptrs, dst[part], None) == 0
    assert (
        SEND_SWA(
            manager,
            "D",
            pkv.state_data_ptrs[0],
            dkv.state_data_ptrs[0],
            pkv.state_item_lens[0],
            src[-2:],
            dst[-2:],
            None,
        )
        == 0
    )
    for ps, ds, source, target in (
        (pp.full_kv_pool, dp.full_kv_pool, src, dst),
        (pp.swa_kv_pool, dp.swa_kv_pool, src[-2:], dst[-2:]),
    ):
        for a, b in zip(ps.k_buffer + ps.v_buffer, ds.k_buffer + ds.v_buffer):
            a, b = a.reshape(16, -1), b.reshape(21, -1)
            assert torch.equal(a[source], b[target])
            assert torch.count_nonzero(b[0]) == 0


def test_pd_pause_retract_rejected_before_any_scheduler_mutation():
    pause = definitions(
        SRT / "managers/tokenizer_manager.py",
        {"pause_generation"},
        owner="TokenizerManager",
    ).pause_generation
    manager = NS(
        server_args=NS(disaggregation_mode="decode"),
        model_config=NS(hf_config=fixture()),
    )
    with pytest.raises(ValueError, match="mode='abort'"):
        asyncio.run(pause(manager, NS(mode="retract")))
    assert not hasattr(manager, "is_pause")


def server_info_class():
    return definitions(
        SRT / "disaggregation/common/conn.py",
        {"PrefillServerInfo"},
        dataclasses=dataclasses,
        __name__=__name__,
    ).PrefillServerInfo


@pytest.mark.parametrize("peer_kind", ["same", "missing", "wrong"])
def test_real_bootstrap_validation_blocks_rank_mapping_on_mismatch(peer_kind):
    scheduler, kv, _ = runtime()
    kv.welm_vlm_contract = PD.build_welm_vlm_pd_contract(scheduler, kv)
    peer = copy.deepcopy(kv.welm_vlm_contract)
    if peer_kind == "missing":
        peer = None
    elif peer_kind == "wrong":
        peer["attn_tp_size"] = 8
    info = dict(
        attn_tp_size=4,
        attn_cp_size=1,
        dp_size=1,
        pp_size=1,
        page_size=4,
        kv_cache_dtype="bfloat16",
        follow_bootstrap_room=True,
    )
    if peer is not None:
        info["welm_vlm_contract"] = peer
    ensure = definitions(
        SRT / "disaggregation/common/conn.py",
        {"try_ensure_parallel_info"},
        owner="CommonKVManager",
        PrefillServerInfo=server_info_class(),
        requests=NS(get=lambda *a, **kw: NS(status_code=200, json=lambda: info)),
        logger=logging.getLogger(__name__),
    ).try_ensure_parallel_info
    manager = NS(
        prefill_info_table={},
        kv_args=kv,
        kv_cache_dtype_str="bfloat16",
        dcp_size=1,
        _resolve_rank_mapping=Mock(),
    )
    if peer_kind == "same":
        assert ensure(manager, "P:8995")
        manager._resolve_rank_mapping.assert_called_once()
        assert "P:8995" in manager.prefill_info_table
    else:
        with pytest.raises(ValueError, match="WeLM-VL PD"):
            ensure(manager, "P:8995")
        manager._resolve_rank_mapping.assert_not_called()
        assert not manager.prefill_info_table


@pytest.mark.parametrize("welm", [False, True])
def test_bootstrap_round_trip_and_all_prefill_rank_agreement(welm):
    scheduler, kv, _ = runtime()
    contract = PD.build_welm_vlm_pd_contract(scheduler, kv) if welm else None
    methods = definitions(
        SRT / "disaggregation/common/conn.py",
        {"_handle_route_put", "_handle_route_get"},
        owner="CommonKVBootstrapServer",
        web=NS(
            Response=lambda **kwargs: NS(**kwargs),
            json_response=lambda data, **kwargs: NS(data=data, **kwargs),
        ),
        PrefillServerInfo=server_info_class(),
        PrefillRankInfo=NS,
        logger=logging.getLogger(__name__),
        dataclasses=dataclasses,
    )
    server = NS(
        welm_vlm_contract=None,
        _welm_vlm_contract_registered=False,
        attn_tp_size=None,
        attn_cp_size=None,
        dp_size=None,
        pp_size=None,
        page_size=None,
        kv_cache_dtype=None,
        prefill_http_port=None,
        follow_bootstrap_room=None,
        enable_dsa_cache_layer_split=None,
        lock=asyncio.Lock(),
        prefill_port_table={},
        _registered_count=0,
        _is_ready=lambda: True,
    )
    data = dict(
        attn_tp_size=4,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
        attn_dp_size=1,
        attn_dp_rank=0,
        pp_size=1,
        pp_rank=0,
        system_dp_size=1,
        system_dp_rank=0,
        rank_ip="127.0.0.1",
        rank_port=8000,
        page_size=4,
        kv_cache_dtype="bfloat16",
    )
    if welm:
        data["welm_vlm_contract"] = contract

    async def exercise():
        async def body():
            return data

        for rank in range(4):
            data["attn_tp_rank"] = rank
            assert (
                await methods._handle_route_put(server, NS(json=body))
            ).status == 200
        response = await methods._handle_route_get(
            server,
            NS(
                query={
                    key: "-1"
                    for key in (
                        "prefill_dp_rank",
                        "prefill_cp_rank",
                        "target_tp_rank",
                        "target_pp_rank",
                    )
                }
            ),
        )
        assert response.status == 200
        assert response.data.get("welm_vlm_contract") == contract
        assert ("welm_vlm_contract" in response.data) == welm
        if welm:
            data["welm_vlm_contract"] = {**contract, "kv_mirror": False}
            assert (
                await methods._handle_route_put(server, NS(json=body))
            ).status == 400
            assert server._registered_count == 4

    asyncio.run(exercise())


def test_manual_comparison_requires_first_decode_and_matching_workload():
    tool = load(ROOT / "examples/runtime/welm_vl/compare_pd.py", "compare_welm_pd_test")
    response = dict(
        choices=[
            dict(
                message=dict(content="a b"),
                logprobs=dict(
                    content=[
                        dict(token="a", bytes=[97], logprob=-0.1),
                        dict(token="b", bytes=[98], logprob=-0.2),
                    ]
                ),
            )
        ],
        usage=dict(
            prompt_tokens=17,
            completion_tokens=2,
            total_tokens=19,
            prompt_tokens_details=dict(image_tokens=8),
        ),
    )
    assert tool.compare(response, response, 0)["passed"]
    other = copy.deepcopy(response)
    other["choices"][0]["logprobs"]["content"][1]["logprob"] -= 0.1
    assert not tool.compare(response, other, 0.05)["passed"]
    assert tool.compare(response, other, 0.2)["passed"]
    other["usage"]["prompt_tokens"] += 1
    assert not tool.compare(response, other, 0.2)["passed"]
    one_token = copy.deepcopy(response)
    one_token["choices"][0]["logprobs"]["content"].pop()
    assert not tool.compare(one_token, one_token, 0)[
        "observed_handoff_and_first_decode"
    ]
