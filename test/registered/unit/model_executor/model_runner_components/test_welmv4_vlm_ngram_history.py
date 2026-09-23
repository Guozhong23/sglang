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
from pathlib import Path
from types import SimpleNamespace

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
        ForwardMode=SimpleNamespace(EXTEND="extend", DECODE="decode"),
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

    def _batch(self, *reqs, mode="extend"):
        return SimpleNamespace(
            reqs=list(reqs),
            req_pool_indices=torch.tensor([req.req_pool_idx for req in reqs]),
            forward_mode=mode,
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
        shared = self.manager.share_table_from(self.manager)
        self.assertIs(shared.table, self.table)
        self.assertEqual(shared.image_token_id, 154752)
        text_manager = manager_module.NgramEmbeddingManager(
            enabled=True, table=self.table, n=3, k=4
        )
        with self.assertRaisesRegex(RuntimeError, "image token"):
            text_manager.share_table_from(self.manager)


if __name__ == "__main__":
    unittest.main()
