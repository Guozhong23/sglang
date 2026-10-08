"""Exercise the real token-table lifecycle on CPU without serving dependencies.

Only SGLang imports and the device scatter kernel are isolated. The manager,
chunk handling, request history reconstruction and sampling updates execute
from their production source with real torch tensors.
"""

import ast
import importlib.util
import runpy
import sys
import types
import unittest
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[5] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _scatter_reference(**kwargs):
    offset = 0
    for row, start, length in zip(
        kwargs["row_indices"].tolist(),
        kwargs["column_starts"].tolist(),
        kwargs["req_lens"].tolist(),
    ):
        kwargs["ne_token_table"][row, start : start + length] = kwargs["tokens"][
            offset : offset + length
        ]
        offset += length


def _load_manager():
    root = Path(__file__).resolve().parents[5] / "python/sglang/srt"
    helper_path = root / "models/welmv4_vlm_utils.py"
    helper_spec = importlib.util.spec_from_file_location(
        "welm_vlm_history_utils", helper_path
    )
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)

    path = root / "model_executor/model_runner_components/ngram_embedding_manager.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("sglang.")
        )
    ]
    module = types.ModuleType("_welm_vlm_ngram_history_test_manager")
    module.__dict__.update(
        is_npu=lambda: False,
        ForwardMode=SimpleNamespace(EXTEND="extend", DECODE="decode", MIXED="mixed"),
        normalize_welmv4_image_tokens=helpers.normalize_welmv4_image_tokens,
        update_token_table=_scatter_reference,
    )
    # dataclasses resolves postponed annotations through sys.modules.
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


manager_module = _load_manager()


def _request(tokens, *, row=1, pads=(), prefix=0, length=None):
    return SimpleNamespace(
        rid=f"request-{row}",
        req_pool_idx=row,
        origin_input_ids=array("q", tokens),
        output_ids=array("q"),
        prefix_indices=list(range(prefix)),
        extend_range=SimpleNamespace(
            length=len(tokens) - prefix if length is None else length
        ),
        multimodal_inputs=SimpleNamespace(
            mm_items=[
                SimpleNamespace(pad_value=pad, is_image=lambda: True) for pad in pads
            ]
        ),
        ngram_token_table_needs_init=False,
    )


class TestWeLMVLMNgramHistory(unittest.TestCase):
    def setUp(self):
        self.table = torch.full((4, 16), -1, dtype=torch.int32)
        self.manager = manager_module.NgramEmbeddingManager(
            enabled=True, table=self.table, n=3, k=4, image_token_id=154752
        )

    def _batch(self, *reqs, mode="extend", decoding_reqs=()):
        return SimpleNamespace(
            reqs=list(reqs),
            req_pool_indices=torch.tensor([req.req_pool_idx for req in reqs]),
            forward_mode=mode,
            decoding_reqs=list(decoding_reqs),
        )

    def test_chunked_prefill_and_decode_keep_logical_history(self):
        original = [11, 999001, 999001, 12, 999002, 999002, 13]
        req = _request(original, pads=[999001, 999002], length=3)
        batch = self._batch(req)
        self.manager.prepare_for_forward(batch, chunked_req=req)
        self.assertEqual(self.table[1, :3].tolist(), [11, 154752, 154752])
        self.assertEqual(batch.ne_skip_token_table_update.tolist(), [True])

        req.prefix_indices = [0, 1, 2]
        req.extend_range.length = 4
        self.manager.prepare_for_forward(batch, chunked_req=None)
        self.assertEqual(
            self.table[1, :7].tolist(), [11, 154752, 154752, 12, 154752, 154752, 13]
        )
        self.assertIsNone(batch.ne_skip_token_table_update)

        info = SimpleNamespace(
            token_table=self.table,
            skip_token_table_update=None,
            out_column_starts=torch.empty(1, dtype=torch.int32),
            out_req_lens=torch.empty(1, dtype=torch.int32),
        )
        self.manager.update_after_decode(
            torch.tensor([71]),
            SimpleNamespace(
                ngram_embedding_info=info,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=torch.tensor([7]),
                batch_size=1,
            ),
        )
        self.assertEqual(self.table[1, 5:8].tolist(), [154752, 13, 71])
        self.assertEqual(list(req.origin_input_ids), original)
        self.assertTrue(torch.all(self.table[0] == -1))

    def test_prefix_cache_rebuild_normalizes_preceding_image_tokens(self):
        req = _request([11, 999001, 999001, 12, 13], pads=[999001], prefix=3)
        self.manager.prepare_for_forward(self._batch(req), chunked_req=None)
        # The cached prefix is not in this new table row. Only its two-token
        # n-gram boundary plus the uncached suffix needs reconstruction.
        self.assertEqual(self.table[1, :5].tolist(), [-1, 154752, 154752, 12, 13])

    def test_mixed_batch_does_not_normalize_another_requests_padding(self):
        first = _request([11, 999001, 999002], row=1, pads=[999001])
        second = _request([21, 999002, 22], row=2, pads=[999002])
        self.manager.prepare_for_forward(self._batch(first, second), chunked_req=None)
        self.assertEqual(self.table[1, :3].tolist(), [11, 154752, 999002])
        self.assertEqual(self.table[2, :3].tolist(), [21, 154752, 22])

    def test_mixed_chunk_normalizes_prefill_without_overwriting_decode_history(self):
        manager = replace(self.manager, welm_mixed_chunk=True)
        original = [11, 999001, 999001, 12, 999002, 999002, 13]
        image_req = _request(original, row=1, pads=[999001, 999002], prefix=3, length=2)
        text_req = _request([31, 32], row=3)
        decode_req = _request([21, 999003, 22], row=2, pads=[999003])
        # Under overlap the newest decode history is already on device while
        # the scheduler's CPU output_ids are one token behind.
        decode_req.output_ids = array("q", [81])
        self.table[2, :5] = torch.tensor([21, 154752, 22, 81, 82])
        device_history = self.table[2].clone()
        batch = self._batch(
            image_req, text_req, decode_req, mode="mixed", decoding_reqs=[decode_req]
        )
        manager.prepare_for_forward(batch, chunked_req=image_req)
        self.assertEqual(self.table[1, :5].tolist(), [-1, 154752, 154752, 12, 154752])
        self.assertEqual(self.table[3, :2].tolist(), [31, 32])
        self.assertTrue(torch.equal(self.table[2], device_history))
        self.assertEqual(
            batch.ne_skip_token_table_update.tolist(), [True, False, False]
        )
        self.assertEqual(list(image_req.origin_input_ids), original)
        self.assertEqual(list(decode_req.origin_input_ids), [21, 999003, 22])

        manager.update_after_decode(
            torch.tensor([901, 902, 903]),
            SimpleNamespace(
                ngram_embedding_info=SimpleNamespace(
                    token_table=self.table,
                    skip_token_table_update=batch.ne_skip_token_table_update,
                ),
                req_pool_indices=batch.req_pool_indices,
                seq_lens=torch.tensor([5, 2, 5]),
                batch_size=3,
            ),
        )
        self.assertEqual(self.table[1, 5].item(), -1)
        self.assertEqual(self.table[3, :3].tolist(), [31, 32, 902])
        self.assertEqual(self.table[2, :6].tolist(), [21, 154752, 22, 81, 82, 903])

        image_req.prefix_indices = list(range(5))
        image_req.extend_range.length = 2
        batch = self._batch(
            image_req, decode_req, mode="mixed", decoding_reqs=[decode_req]
        )
        manager.prepare_for_forward(batch, chunked_req=None)
        self.assertIsNone(batch.ne_skip_token_table_update)
        self.assertEqual(
            self.table[1, :7].tolist(), [-1, 154752, 154752, 12, 154752, 154752, 13]
        )
        self.assertEqual(self.table[2, :6].tolist(), [21, 154752, 22, 81, 82, 903])
        self.assertTrue(torch.all(self.table[0] == -1))

    def test_mixed_chunk_capability_covers_text_and_vl_only_on_enabled_npu(self):
        for architecture in (
            "WeLMV4MoeForCausalLM",
            "WeLMV4VLMForConditionalGeneration",
            "OtherForCausalLM",
        ):
            for npu, mixed in ((True, True), (False, True), (True, False)):
                with self.subTest(architecture=architecture, npu=npu, mixed=mixed):
                    # No model allocation is needed to exercise the capability
                    # gate and checkpoint image-token identity in the factory.
                    config = SimpleNamespace(
                        use_ngram_embedding=False,
                        hf_config=SimpleNamespace(
                            architectures=[architecture], image_token_id=154752
                        ),
                    )
                    with patch.object(manager_module, "is_npu", return_value=npu):
                        manager = manager_module.NgramEmbeddingManager.from_model(
                            model=None,
                            model_config=config,
                            req_to_token_pool=None,
                            server_args=SimpleNamespace(enable_mixed_chunk=mixed),
                            max_running_requests=4,
                            device="cpu",
                        )
                    self.assertEqual(
                        manager.welm_mixed_chunk,
                        npu and mixed and architecture != "OtherForCausalLM",
                    )
                    self.assertEqual(
                        manager.image_token_id,
                        154752
                        if architecture == "WeLMV4VLMForConditionalGeneration"
                        else None,
                    )

    def test_disaggregated_history_rebuild_preserves_output_tokens(self):
        req = _request([11, 999001, 999001], pads=[999001])
        req.output_ids = array("q", [31, 32])
        req.ngram_token_table_needs_init = True
        self.manager.prepare_for_forward(
            self._batch(req, mode="decode"), chunked_req=None
        )
        self.assertEqual(self.table[1, :5].tolist(), [11, 154752, 154752, 31, 32])
        self.assertFalse(req.ngram_token_table_needs_init)
        self.assertEqual(list(req.origin_input_ids), [11, 999001, 999001])

    def test_text_manager_keeps_current_placeholder_behavior(self):
        manager = manager_module.NgramEmbeddingManager(
            enabled=True, table=self.table, n=3, k=4
        )
        req = _request([11, 999001, 12], pads=[999001])
        manager.prepare_for_forward(self._batch(req), chunked_req=None)
        self.assertEqual(self.table[1, :3].tolist(), [11, 999001, 12])

    def test_shared_history_retains_and_validates_image_token_identity(self):
        mixed_manager = replace(self.manager, welm_mixed_chunk=True)
        shared = mixed_manager.share_table_from(mixed_manager)
        self.assertTrue(shared.welm_mixed_chunk)
        self.assertIs(shared.table, self.table)
        self.assertEqual(shared.image_token_id, 154752)
        text_manager = manager_module.NgramEmbeddingManager(
            enabled=True, table=self.table, n=3, k=4
        )
        with self.assertRaisesRegex(RuntimeError, "image token"):
            text_manager.share_table_from(self.manager)


if __name__ == "__main__":
    unittest.main()
