"""CPU-only tests of the VL checkpoint/target execution boundary.

Load the device-independent helper and production forward methods directly,
so this test needs torch but no serving stack or NPU dependencies.
"""

import ast
import importlib.util
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

# Load the CPU-only CI marker without importing sglang's serving dependencies.
register_cpu_ci = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/models/welmv4_vlm_utils.py"
    )
    spec = importlib.util.spec_from_file_location("welmv4_vlm_utils_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helpers = _load_helpers()


def _config():
    return SimpleNamespace(
        _welmv4_vlm_target_only=True,
        num_hidden_layers=48,
        num_nextn_predict_layers=3,
        kv_mirror_layers=[48, 49, 50] + list(range(47, 32, -1)),
        kv_mirror_imitated_layers=[0, 0, 0] + list(range(1, 16)),
        kv_mirror_repeat=[3] + [1] * 15,
    )


class TestWeLMVLMTargetConfig(unittest.TestCase):
    def test_target_view_preserves_checkpoint_and_all_active_pairs(self):
        config = _config()
        runtime = helpers.welmv4_vlm_target_config(config)
        self.assertIsNot(runtime, config)
        self.assertEqual(runtime.kv_mirror_layers, list(range(47, 32, -1)))
        self.assertEqual(runtime.kv_mirror_imitated_layers, list(range(1, 16)))
        self.assertEqual(runtime.kv_mirror_repeat, [1] * 15)
        self.assertEqual(config.kv_mirror_layers[:3], [48, 49, 50])
        self.assertEqual(config.kv_mirror_imitated_layers[:3], [0, 0, 0])
        self.assertEqual(config.kv_mirror_repeat[0], 3)
        self.assertEqual(runtime.num_nextn_predict_layers, 3)
        # Every executed consumer still has exactly its original source.
        expected = dict(zip(config.kv_mirror_layers, config.kv_mirror_imitated_layers))
        actual = dict(zip(runtime.kv_mirror_layers, runtime.kv_mirror_imitated_layers))
        self.assertEqual(actual, {k: v for k, v in expected.items() if k < 48})

    def test_text_config_is_unchanged_even_with_speculative_decoding(self):
        config = _config()
        del config._welmv4_vlm_target_only
        self.assertIs(helpers.welmv4_vlm_target_config(config, "EAGLE"), config)

    def test_vl_speculative_decoding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            helpers.welmv4_vlm_target_config(_config(), "EAGLE")

    def test_active_source_fanout_is_rejected(self):
        config = _config()
        config.kv_mirror_imitated_layers[-1] = 1
        with self.assertRaisesRegex(ValueError, "fan-out"):
            helpers.welmv4_vlm_target_config(config)

    def test_invalid_inactive_metadata_is_not_silently_discarded(self):
        for field, value, message in (
            ("kv_mirror_layers", [51], "out of range"),
            ("kv_mirror_layers", [48, 48], "unique"),
            ("kv_mirror_imitated_layers", [48], "source is invalid"),
        ):
            config = _config()
            config.kv_mirror_layers = [48] * len(value)
            config.kv_mirror_imitated_layers = [0] * len(value)
            setattr(config, field, value)
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, message):
                    helpers.welmv4_vlm_target_config(config)


class TestWeLMVLMLogicalHistory(unittest.TestCase):
    def _images(self, *pads):
        return SimpleNamespace(
            mm_items=[
                SimpleNamespace(pad_value=pad, is_image=lambda: True) for pad in pads
            ]
        )

    def test_multiple_images_and_chunk_boundary_history(self):
        tokens = [101, 999001, 999001, 102, 999002, 999002, 103]
        logical = helpers.normalize_welmv4_image_tokens(
            tokens, self._images(999001, 999002), 154752
        )
        self.assertEqual(logical, [101, 154752, 154752, 102, 154752, 154752, 103])
        # A 3-gram at the next chunk starts with the two logical predecessors.
        self.assertEqual(logical[2:5], [154752, 102, 154752])
        self.assertEqual(tokens, [101, 999001, 999001, 102, 999002, 999002, 103])

    def test_normalization_is_scoped_to_request_and_image_modality(self):
        mm = self._images(999001, None)
        mm.mm_items.append(SimpleNamespace(pad_value=999002, is_image=lambda: False))
        self.assertEqual(
            helpers.normalize_welmv4_image_tokens([999001, 999002, 999003], mm, 154752),
            [154752, 999002, 999003],
        )

    def test_text_models_and_requests_do_not_change_ids(self):
        tokens = [101, 999001, 102]
        self.assertIs(
            helpers.normalize_welmv4_image_tokens(tokens, self._images(999001), None),
            tokens,
        )
        self.assertIs(
            helpers.normalize_welmv4_image_tokens(tokens, None, 154752), tokens
        )


def _load_forward(class_name):
    """Run production forward control flow with its NPU layers substituted."""
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/models/welmv4.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    forward = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    extracted = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], forward],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "_is_npu": False,
        "KVMirrorManager": SimpleNamespace(activations_dict_kv={}),
    }
    exec(compile(extracted, str(path), "exec"), namespace)
    return namespace["forward"]


class TestWeLMVLMOverEncodingFusion(unittest.TestCase):
    def test_pre_fused_visual_embeddings_skip_oe_exactly_once(self):
        forward = _load_forward("Qwen2MoeModel")
        fuse = Mock(side_effect=lambda ids, batch, values: values + 10)
        model = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
            embed_tokens=lambda ids: torch.ones((ids.numel(), 2)),
            oe_grams=[2, 2, 3, 3],
            _compute_oe_embedding=fuse,
            scale_seq_times=0,
            start_layer=0,
            end_layer=0,
            layers_to_capture=[],
            norm=lambda values: (values, None),
            _restore_npu_prefill_deepep_output_layout=lambda values, aux, batch: (
                values,
                aux,
            ),
        )
        batch = SimpleNamespace(
            ngram_embedding_info=object(),
            can_run_tbo=False,
            capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        )
        ids = torch.tensor([154752, 7])
        positions = torch.tensor([0, 1])
        text_result = forward(model, ids, positions, batch)
        torch.testing.assert_close(text_result, torch.full((2, 2), 11.0))
        fuse.assert_called_once()

        # The wrapper has already fused text OE and replaced the image row.
        visual_embeddings = torch.tensor([[123.0, 456.0], [11.0, 11.0]])
        fuse.reset_mock()
        visual_result = forward(
            model,
            ids,
            positions,
            batch,
            input_embeds=visual_embeddings,
            skip_oe_fusion=True,
        )
        torch.testing.assert_close(visual_result, visual_embeddings)
        fuse.assert_not_called()

    def test_top_level_forward_propagates_skip_flag_with_false_default(self):
        forward = _load_forward("WeLMV4MoeForCausalLM")
        hidden = torch.tensor([[1.0, 2.0]])
        backbone = Mock(return_value=hidden)
        model = SimpleNamespace(
            model=backbone, pp_group=SimpleNamespace(is_last_rank=False)
        )
        batch = SimpleNamespace()
        ids, positions = torch.tensor([7]), torch.tensor([0])
        for skip in (False, True):
            with self.subTest(skip=skip):
                backbone.reset_mock()
                kwargs = {"skip_oe_fusion": True} if skip else {}
                result = forward(model, ids, positions, batch, **kwargs)
                self.assertIs(result, hidden)
                self.assertEqual(backbone.call_args.kwargs["skip_oe_fusion"], skip)


if __name__ == "__main__":
    unittest.main()
