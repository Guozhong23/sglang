"""CPU contracts for explicit batched vision encoding and DP cache coordination.

Run the production multimodal slicing and LRU implementation with real tensors.
Only serving imports, accelerator synchronization, and the encoder are isolated.
"""

import ast
import dataclasses
import logging
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_cache(cache_bytes):
    path = SRT / "mem_cache/multimodal_cache.py"
    tree = ast.parse(path.read_text())
    future = ast.parse("from __future__ import annotations").body[0]
    tree.body = [future] + [
        node for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module.startswith("sglang."))
    ]
    cache_module = types.ModuleType("_welm_mm_cache_contract_cache")
    sys.modules[cache_module.__name__] = cache_module
    exec(compile(tree, str(path), "exec"), cache_module.__dict__)
    cache = cache_module.MultiModalStaticCache(cache_bytes)

    names = {
        "get_embedding_chunk", "_get_precomputed_embedding", "_flatten_embedding_result",
        "_can_skip_pre_embed_feature_move", "_move_items_to_device",
        "_acknowledge_deferred_cuda_ipc_cache_hits", "_get_chunked_embedding_full",
        "PerImageRequestInfo", "_batch_encode_per_image_misses",
        "_get_chunked_embedding_by_item", "_assemble_per_image_chunk",
        "_get_chunked_prefill_embedding", "_get_multimodal_mask",
        "_adjust_embedding_length", "get_embedding_and_mask",
    }
    path = SRT / "managers/mm_utils.py"
    tree = ast.parse(path.read_text())
    tree.body = [future] + [node for node in tree.body if getattr(node, "name", None) in names]
    module = types.ModuleType("_welm_mm_cache_contract_utils")
    sys.modules[module.__name__] = module
    torch_proxy = SimpleNamespace(
        **{key: value for key, value in vars(torch).items() if key != "npu"},
        npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
    )
    module.__dict__.update(
        torch=torch_proxy, dataclass=dataclasses.dataclass, field=dataclasses.field,
        embedding_cache=cache, EmbeddingResult=cache_module.EmbeddingResult,
        MultiModalStaticCache=cache_module.MultiModalStaticCache,
        EVSEmbeddingResult=type("UnsupportedEVS", (), {}),
        _is_npu=True, _is_hip=False, logger=logging.getLogger(__name__),
        get_schedule=lambda: SimpleNamespace(chunked_prefill_size=4),
        get_parallel=lambda: SimpleNamespace(attn_tp_rank=0, tp_size=4),
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    module._move_items_to_device = Mock(wraps=module._move_items_to_device)
    return module, cache


def _image(key, start=0, rows=2):
    pixels = torch.arange(key, key + rows).float().unsqueeze(1).expand(-1, 4).clone()
    return SimpleNamespace(
        hash=key, pad_value=key, offsets=[(start, start + rows - 1)],
        feature=pixels, image_grid_thw=torch.tensor([[1, 2, rows * 2]]),
        precomputed_embeddings=None,
        acknowledge_deferred_cuda_ipc_feature=lambda count: None,
    )


def _encoder(calls, *, return_list=False):
    def encode(items):
        calls.append([item.hash for item in items])
        assert all(item.feature.device.type == "cpu" for item in items)
        outputs = [item.feature.clone() for item in items]
        return outputs if return_list else torch.cat(outputs)
    return encode


def _run(module, encode, requests, **kwargs):
    """A request is (images, cached prefix length, current chunk token ids)."""
    items, sizes, offsets, prefixes, lengths, ids = [], [0], [], [], [], []
    for images, prefix, tokens in requests:
        items.extend(images)
        sizes.append(len(items))
        offsets.append([offset for item in images for offset in item.offsets])
        prefixes.append(prefix)
        lengths.append(len(tokens))
        ids.extend(tokens)
    return module.get_embedding_and_mask(
        data_embedding_func=encode, embedding_items=items,
        placeholder_tensor=torch.tensor([item.pad_value for item in items]),
        input_ids=torch.tensor(ids), items_size=sizes, prefix_length=prefixes,
        extend_length=lengths, items_offset_list=offsets, **kwargs,
    )


@pytest.mark.parametrize("batched", [False, True])
def test_npu_cross_request_batching_is_explicit_and_keeps_default(batched):
    mm, _ = _load_cache(4096)
    first, second = _image(100), _image(200)
    calls = []
    output, mask, _ = _run(
        mm, _encoder(calls), [([first], 0, [100, 100]), ([second], 0, [200, 200])],
        batch_encode_per_image=batched,
    )
    assert calls == ([[100, 200]] if batched else [[100], [200]])
    assert mm._move_items_to_device.call_count == (1 if batched else 2)
    torch.testing.assert_close(output[:, 0], torch.tensor([100., 101., 200., 201.]))
    assert mask.all()


@pytest.mark.parametrize("cache_bytes", [0, 32, 4096])
@pytest.mark.parametrize("return_list", [False, True])
def test_batched_duplicates_chunks_and_eviction_retain_local_embeddings(
    cache_bytes, return_list
):
    mm, cache = _load_cache(cache_bytes)
    first, second, duplicate = _image(100, start=1, rows=3), _image(200), _image(100, rows=3)
    originals = [item.feature for item in (first, second, duplicate)]
    calls = []
    encode = _encoder(calls, return_list=return_list)
    output, mask, _ = _run(
        mm, encode,
        [([first], 2, [100, 100, 9]), ([second], 0, [200, 200]), ([duplicate], 0, [100])],
        batch_encode_per_image=True, defer_feature_move=True,
    )
    assert calls == [[100, 200]]  # A repeated image is encoded only once.
    torch.testing.assert_close(output[:, 0], torch.tensor([101., 102., 200., 201., 100.]))
    assert mask[:, 0].tolist() == [True, True, False, True, True, True]
    assert not mm._move_items_to_device.called
    for item, original in zip((first, second, duplicate), originals):
        assert item.feature is original
    cached_before = cache.has(first.hash)
    next_output, _, _ = _run(
        mm, encode, [([first], 3, [100])],
        batch_encode_per_image=True, defer_feature_move=True,
    )
    assert len(calls) == 1 + int(not cached_before)
    assert next_output[0, 0].item() == 102


def test_synchronized_global_miss_reencodes_local_hit_before_collectives():
    mm, cache = _load_cache(4096)
    first, second = _image(100), _image(200)
    # Distinguish a stale local hit from the encoder's new broadcast result.
    cache.set(100, mm.EmbeddingResult(embedding=torch.full((2, 4), -100.)))
    sync_calls, calls = [], []

    def synchronize(items, cached_mask):
        sync_calls.append(([item.hash for item in items], cached_mask))
        return [True, True]  # Another rank missed image 100.

    output, _, _ = _run(
        mm, _encoder(calls), [([first], 0, [100, 100]), ([second], 0, [200, 200])],
        batch_encode_per_image=True, defer_feature_move=True,
        synchronize_cache_misses=synchronize,
    )
    assert sync_calls == [([100, 200], [True, False])]
    assert calls == [[100, 200]]
    torch.testing.assert_close(output[:, 0], torch.tensor([100., 101., 200., 201.]))
    torch.testing.assert_close(cache.get_single(100).embedding, first.feature)
    assert not mm._move_items_to_device.called
    hit_output, _, _ = _run(
        mm, _encoder(calls), [([first], 0, [100, 100])],
        batch_encode_per_image=True, defer_feature_move=True,
        synchronize_cache_misses=lambda items, cached: [False],
    )
    torch.testing.assert_close(hit_output, first.feature)
    assert calls == [[100, 200]]


def test_all_local_hits_still_synchronize_and_all_global_hits_skip_encoder():
    mm, cache = _load_cache(4096)
    first = _image(100)
    cache.set(100, mm.EmbeddingResult(embedding=first.feature.clone()))
    synchronize = Mock(return_value=[False])
    encode = Mock(side_effect=AssertionError("all-rank cache hit must not encode"))
    output, _, _ = _run(
        mm, encode, [([first], 0, [100, 100]), ([first], 0, [100])],
        batch_encode_per_image=True, defer_feature_move=True,
        synchronize_cache_misses=synchronize,
    )
    synchronize.assert_called_once_with([first], [True])
    assert not encode.called
    torch.testing.assert_close(output[:, 0], torch.tensor([100., 101., 100.]))


@pytest.mark.parametrize("batched", [False, True])
def test_encoder_failure_does_not_mutate_deferred_cpu_pixels(batched):
    mm, cache = _load_cache(4096)
    image = _image(100)
    original = image.feature
    with pytest.raises(RuntimeError, match="injected"):
        _run(
            mm, Mock(side_effect=RuntimeError("injected")), [([image], 0, [100, 100])],
            batch_encode_per_image=batched, defer_feature_move=True,
        )
    assert image.feature is original
    assert not mm._move_items_to_device.called
    assert len(cache) == 0


@pytest.mark.parametrize(
    "bad_mask, error", [([], "mask length"), ([False], "local cache miss")]
)
def test_invalid_synchronization_mask_fails_before_encoder(bad_mask, error):
    mm, _ = _load_cache(4096)
    image = _image(100)
    encode = Mock(side_effect=AssertionError("invalid mask must fail before encoding"))
    with pytest.raises((ValueError, RuntimeError), match=error):
        _run(
            mm, encode, [([image], 0, [100, 100])], batch_encode_per_image=True,
            synchronize_cache_misses=lambda items, cached: bad_mask,
        )
    assert not encode.called


def test_cache_coordination_cannot_use_per_request_fallback():
    mm, _ = _load_cache(4096)
    image = _image(100)
    encode, synchronize = Mock(), Mock()
    with pytest.raises(ValueError, match="requires batched per-image"):
        _run(
            mm, encode, [([image], 0, [100, 100])],
            synchronize_cache_misses=synchronize,
        )
    assert not encode.called
    assert not synchronize.called


def test_empty_overlap_still_coordinates_before_skipping_encoder():
    mm, _ = _load_cache(4096)
    image = _image(100)
    encode = Mock(side_effect=AssertionError("no overlapping image"))
    synchronize = Mock(return_value=[])
    output, mask, ids = _run(
        mm, encode, [([image], 3, [9])],
        batch_encode_per_image=True, defer_feature_move=True,
        synchronize_cache_misses=synchronize,
    )
    synchronize.assert_called_once_with([], [])
    assert output is None and mask is None
    assert ids.tolist() == [9]
    assert not encode.called
