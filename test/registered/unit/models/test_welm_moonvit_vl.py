"""CPU arithmetic and checkpoint contracts; no NPU or 80B allocation required."""

import ast
import copy
import math
import os
import re
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import PytorchGELUTanh

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=10, suite="base-a-test-cpu")
MODEL = ROOT / "python/sglang/srt/models/welm_moonvit_vl.py"
KIMI = ROOT / "python/sglang/srt/models/kimi_vl_moonvit.py"
PROCESSOR = ROOT / "python/sglang/srt/multimodal/processors/welm_moonvit_vl.py"


def definitions(path, names, **scope):
    tree = ast.parse(path.read_text())
    tree.body = ast.parse("from __future__ import annotations").body + [
        n for n in tree.body if getattr(n, "name", None) in names
    ]
    scope = dict(torch=torch, nn=nn, F=F, math=math, copy=copy, re=re, os=os, **scope)
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


FUN = definitions(
    MODEL,
    [
        "moonvit_rope_coefficients",
        "moonvit_apply_rope",
        "convert_moonvit_weights",
        "checked_moonvit_language_weights",
    ],
)
EXPAND = definitions(
    PROCESSOR, ["expand_moonvit_image_tokens"]
).expand_moonvit_image_tokens


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("grids", [[(2, 4)], [(2, 2), (4, 2)]])
def test_rope_matches_checkpoint_complex_xy_adjacent_pairs(dtype, grids):
    torch.manual_seed(41)
    positions = [(x, y) for h, w in grids for y in range(h) for x in range(w)]
    frequency = torch.tensor([10000 ** (-4 * i / 72) for i in range(18)])
    phases = torch.tensor(positions).float()[:, None, :] * frequency[None, :, None]
    cis = torch.polar(torch.ones_like(phases), phases).flatten(1).unsqueeze(1)
    q, k = (torch.randn(len(positions), 4, 72).to(dtype) for _ in range(2))
    reference = [
        torch.view_as_real(torch.view_as_complex(x.float().reshape(-1, 4, 36, 2)) * cis)
        .flatten(-2)
        .to(dtype)
        for x in (q, k)
    ]
    coefficients = FUN.moonvit_rope_coefficients(grids, 72, "cpu")
    actual = FUN.moonvit_apply_rope(q, k, coefficients)
    for a, b in zip(actual, reference):
        torch.testing.assert_close(
            a,
            b,
            atol=2e-6 if dtype == torch.float32 else 0.016,
            rtol=1e-5 if dtype == torch.float32 else 0.008,
        )


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "kind,projection,shape",
    [
        ("gate", "gate_proj", (3, 8)),
        ("up", "up_proj", (3, 8)),
        ("down", "down_proj", (8, 3)),
    ],
)
def test_stacked_expert_weights_preserve_all_experts_and_orientation(
    transposed, kind, projection, shape
):
    config = SimpleNamespace(hidden_size=8, moe_intermediate_size=3, num_experts=4)
    weights = torch.arange(4 * 24).reshape(4, *shape)
    stored = weights.transpose(1, 2).contiguous() if transposed else weights
    converted = list(
        FUN.convert_moonvit_weights(
            [(f"language_model.model.layers.0.mlp.stacked_{kind}_weights", stored)],
            config,
        )
    )
    assert len(converted) == 4
    for i, (name, tensor) in enumerate(converted):
        assert name == f"model.layers.0.mlp.experts.{i}.{projection}.weight"
        torch.testing.assert_close(tensor, weights[i])


def test_weight_prefixes_and_oe_projection_are_not_silently_dropped():
    names = {
        "language_model.model.oe_up_proj.weight": "model.oe_gate_up_proj.weight",
        "language_model.lm_head.weight": "lm_head.weight",
        "visual.encoder.blocks.0.wqkv.weight": "vision_encoder.encoder.blocks.0.attn.qkv_proj.weight",
        "visual.encoder.blocks.0.wo.bias": "vision_encoder.encoder.blocks.0.attn.proj.bias",
        "visual.merger.linear_1.weight": "vision_projector.linear_1.weight",
    }
    weight = torch.ones(1)
    assert dict(FUN.convert_moonvit_weights([(n, weight) for n in names], None)) == {
        n: weight for n in names.values()
    }
    with pytest.raises(ValueError, match="prefix"):
        list(FUN.convert_moonvit_weights([("unknown.weight", weight)], None))


def test_weight_coverage_requires_every_qkv_part_and_expert():
    parameters = {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.experts.w13_weight",
        "model.oe_gate_up_proj.weight",
    }
    names = [f"model.layers.0.self_attn.{p}_proj.weight" for p in "qkv"]
    names += [
        f"model.layers.0.mlp.experts.{e}.{p}_proj.weight"
        for e in range(4)
        for p in ("gate", "up")
    ]
    names += ["model.oe_gate_up_proj.weight"]
    weights = [(n, torch.zeros(1)) for n in names]
    assert list(FUN.checked_moonvit_language_weights(weights, parameters, 4)) == weights
    for i in range(len(weights)):
        with pytest.raises(ValueError, match="Missing MoonViT language"):
            list(
                FUN.checked_moonvit_language_weights(
                    weights[:i] + weights[i + 1 :], parameters, 4
                )
            )
    with pytest.raises(ValueError, match="duplicate"):
        list(FUN.checked_moonvit_language_weights(weights + weights[:1], parameters, 4))


def test_placeholder_expansion_preserves_native_prompt_ids():
    assert EXPAND([1, 77, 2, 77, 3], 77, [[1, 2, 4], [1, 4, 4]], (2, 2)) == [
        1,
        77,
        77,
        2,
        77,
        77,
        77,
        77,
        3,
    ]
    assert EXPAND([1, 2], 77, [], (2, 2)) == [1, 2]
    with pytest.raises(ValueError, match="placeholder"):
        EXPAND([77, 77], 77, [[1, 2, 2]], (2, 2))
    with pytest.raises(ValueError, match="grid"):
        EXPAND([77], 77, [[2, 2, 2]], (2, 2))


class TupleLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return super().forward(x), None


class CPUVisionAttention(nn.Module):
    """Replace communication/Ascend attention only; exercise real encoder blocks."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__()
        self.heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.customized_position_embedding_applier = kwargs[
            "customized_position_embedding_applier"
        ]

    def forward(self, x, cu_seqlens, position_embeddings, **kwargs):
        q, k, v = self.qkv_proj(x).reshape(len(x), 3, self.heads, -1).unbind(1)
        q, k = self.customized_position_embedding_applier(q, k, position_embeddings)
        result = []
        for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            result.append(
                F.scaled_dot_product_attention(
                    q[a:b].transpose(0, 1),
                    k[a:b].transpose(0, 1),
                    v[a:b].transpose(0, 1),
                )
                .transpose(0, 1)
                .flatten(1)
            )
        return self.proj(torch.cat(result))


def tiny_vision():
    kimi = definitions(
        KIMI,
        [
            "Learnable2DInterpPosEmb",
            "MoonVisionPatchEmbed",
            "MLP2",
            "MoonVitEncoderLayer",
            "patch_merger",
        ],
        Conv2dLayer=nn.Conv2d,
        ColumnParallelLinear=TupleLinear,
        RowParallelLinear=TupleLinear,
        ModelSlimConfig=type("UnusedQuantization", (), {}),
        VisionAttention=CPUVisionAttention,
        apply_rope=None,
        Sequence=(list, tuple),
        get_parallel=lambda: SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
        add_prefix=lambda name, prefix: f"{prefix}.{name}",
        _MAX_INFERENCE_POS_EMB_CACHE_ENTRIES=256,
    )
    model = definitions(
        MODEL,
        [
            "moonvit_rope_coefficients",
            "moonvit_apply_rope",
            "WeLMMoonViTEncoder",
            "WeLMMoonViTProjector",
        ],
        MoonVisionPatchEmbed=kimi.MoonVisionPatchEmbed,
        MoonVitEncoderLayer=kimi.MoonVitEncoderLayer,
        patch_merger=kimi.patch_merger,
        ColumnParallelLinear=TupleLinear,
        RowParallelLinear=TupleLinear,
        get_parallel=lambda: SimpleNamespace(attn_tp_size=1),
        PytorchGELUTanh=PytorchGELUTanh,
        add_prefix=lambda name, prefix: f"{prefix}.{name}",
    )
    cfg = SimpleNamespace(
        patch_size=2,
        merge_kernel_size=[2, 2],
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=24,
        num_hidden_layers=2,
        init_pos_emb_height=4,
        init_pos_emb_width=4,
        out_hidden_size=8,
    )
    return model.WeLMMoonViTEncoder(cfg).eval(), model.WeLMMoonViTProjector(cfg).eval()


def reference_image(encoder, projector, pixels, h, w):
    # Independent checkpoint-style per-image path, using complex RoPE and
    # explicit row-major 2x2 merging (no production merge/rope helper).
    patch = encoder.patch_embed
    hidden = F.conv2d(pixels, patch.proj.weight, patch.proj.bias, stride=2).flatten(1)
    pos = (
        F.interpolate(
            patch.pos_emb.weight.permute(2, 0, 1)[None], size=(h, w), mode="bicubic"
        )[0]
        .permute(1, 2, 0)
        .reshape(h * w, -1)
    )
    hidden = hidden + pos
    freq = torch.tensor([10000 ** (-4 * i / 8) for i in range(2)])
    xy = torch.tensor([(x, y) for y in range(h) for x in range(w)])
    phase = (xy[:, None].float() * freq[None, :, None]).flatten(1)
    cis = torch.polar(torch.ones_like(phase), phase)[:, None]
    for block in encoder.encoder.blocks:
        normalized = F.layer_norm(
            hidden, (16,), block.norm0.weight, block.norm0.bias, 1e-5
        )
        q, k, v = (
            F.linear(normalized, block.attn.qkv_proj.weight, block.attn.qkv_proj.bias)
            .reshape(h * w, 3, 2, 8)
            .unbind(1)
        )
        q, k = [
            torch.view_as_real(
                torch.view_as_complex(x.reshape(h * w, 2, 4, 2)) * cis
            ).flatten(-2)
            for x in (q, k)
        ]
        attn = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            )
            .transpose(0, 1)
            .reshape(h * w, 16)
        )
        hidden = hidden + F.linear(attn, block.attn.proj.weight, block.attn.proj.bias)
        normalized = F.layer_norm(
            hidden, (16,), block.norm1.weight, block.norm1.bias, 1e-5
        )
        mlp = F.linear(normalized, block.mlp.fc0.weight, block.mlp.fc0.bias)
        mlp = PytorchGELUTanh()(mlp)
        hidden = hidden + F.linear(mlp, block.mlp.fc1.weight, block.mlp.fc1.bias)
    norm = encoder.encoder.final_layernorm
    hidden = F.layer_norm(hidden, (16,), norm.weight, norm.bias, 1e-5).reshape(h, w, 16)
    merged = torch.stack(
        [
            hidden[y : y + 2, x : x + 2].reshape(4, 16)
            for y in range(0, h, 2)
            for x in range(0, w, 2)
        ]
    )
    norm = projector.pre_norm
    merged = F.layer_norm(merged, (16,), norm.weight, norm.bias, 1e-5).flatten(1)
    out = F.gelu(F.linear(merged, projector.linear_1.weight, projector.linear_1.bias))
    return F.linear(out, projector.linear_2.weight, projector.linear_2.bias)


def test_two_image_encoder_projector_matches_independent_checkpoint_math():
    torch.manual_seed(42)
    encoder, projector = tiny_vision()
    pixels = torch.randn(16, 3, 2, 2)
    with torch.no_grad():
        actual = projector(encoder(pixels, torch.tensor([[1, 2, 4], [1, 4, 2]])))
        expected = torch.cat(
            [
                reference_image(encoder, projector, pixels[:8], 2, 4),
                reference_image(encoder, projector, pixels[8:], 4, 2),
            ]
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual.shape == (4, 8)
    with pytest.raises(ValueError, match="pixels"):
        encoder(pixels[:4], torch.tensor([[1, 2, 4]]))
    with pytest.raises(ValueError, match="t=1"):
        encoder(pixels, torch.tensor([[2, 2, 4]]))


def test_moonvit_processor_expands_ids_without_decoding_or_combined_processor():
    scope = definitions(
        PROCESSOR,
        ["expand_moonvit_image_tokens", "WeLMMoonViTVLImageProcessor"],
        WeLMV4VLMImageProcessor=object,
        WeLMMoonViTVLForConditionalGeneration=object,
    )
    cls = scope.WeLMMoonViTVLImageProcessor
    instance = cls()
    pixels = torch.randn(12, 3, 14, 14)

    class ImageProcessor:
        merge_kernel_size = [2, 2]

        def __call__(self, images, return_tensors):
            assert images == ["first", "second"]
            return {
                "pixel_values": pixels,
                "image_grid_hws": torch.tensor([[2, 4], [2, 2]]),
            }

    instance._processor = SimpleNamespace(image_processor=ImageProcessor())
    instance.image_config = {}
    instance.image_token_id = 77
    instance._load_images = lambda images: images
    out = instance._process_token_ids([1, 77, 2, 77, 3], ["first", "second"])
    assert out["input_ids"].tolist() == [[1, 77, 77, 2, 77, 3]]
    assert out["image_grid_thw"].tolist() == [[1, 2, 4], [1, 2, 2]]
    assert out["pixel_values"] is pixels
    assert instance._process_token_ids([1, 2], [])["input_ids"].tolist() == [[1, 2]]


def test_moonvit_architecture_reaches_vl_ep_validator():
    import runpy
    from unittest.mock import Mock

    startup = runpy.run_path(
        str(Path(__file__).with_name("test_welmv4_vlm_startup.py"))
    )
    validate = startup["_definitions"](
        ROOT / "python/sglang/srt/server_args.py",
        {"_validate_welm_dp_attention_capabilities"},
        {
            "is_npu": lambda: True,
            "parse_connector_type": lambda path: "local",
            "ConnectorType": SimpleNamespace(INSTANCE="instance"),
        },
    )["_validate_welm_dp_attention_capabilities"]
    args = SimpleNamespace(
        model_path="checkpoint",
        get_model_config=lambda: SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["WeLMMoonViTVLForConditionalGeneration"]
            )
        ),
        _validate_welm_vlm_parallel_config=Mock(),
    )
    validate(args)
    args._validate_welm_vlm_parallel_config.assert_called_once_with()


@pytest.mark.parametrize("rank", range(4))
def test_checkpoint_packed_vision_qkv_uses_real_tp4_loader(rank):
    from types import MethodType

    path = ROOT / "python/sglang/srt/layers/linear.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "QKVParallelLinear"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "weight_loader"
    )
    scope = {"torch": torch, "_is_cpu": False}
    module = ast.Module(
        body=ast.parse("from __future__ import annotations").body + [method],
        type_ignores=[],
    )
    exec(compile(module, str(path), "exec"), scope)
    layer = SimpleNamespace(
        tp_size=4,
        tp_rank=rank,
        num_heads=4,
        num_kv_heads=4,
        total_num_heads=16,
        total_num_kv_heads=16,
        head_size=4,
        v_head_size=4,
        num_kv_head_replicas=1,
        use_presharded_weights=False,
    )
    layer.weight_loader = MethodType(scope["weight_loader"], layer)
    packed = torch.arange(192 * 8).reshape(192, 8).float()
    param = nn.Parameter(torch.zeros(48, 8), requires_grad=False)
    param.output_dim = 0
    layer.weight_loader(param, packed)
    expected = torch.cat(
        [part[rank * 16 : (rank + 1) * 16] for part in packed.chunk(3)]
    )
    torch.testing.assert_close(param, expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend,ep", [("none", "1"), ("deepep", "4")])
def test_launcher_preserves_tp4_and_disables_unvalidated_profiles(
    tmp_path, backend, ep
):
    import json
    import subprocess

    stub = tmp_path / "fake-python"
    output = tmp_path / "calls.jsonl"
    stub.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['MOONVIT_TEST_OUTPUT'], 'a') as f:
    f.write(json.dumps({'args': sys.argv[1:], 'env': {k: os.environ.get(k) for k in ['WELM_NPU_USE_MEGAMOE', 'SGLANG_VLM_VISION_USE_ULYSSES', 'SGLANG_VLM_VISION_USE_DP']}}) + '\\n')
""")
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PYTHON_BIN=str(stub),
        MODEL_PATH=str(tmp_path / "weights with spaces"),
        MOE_A2A_BACKEND=backend,
        TP_SIZE="4",
        MOONVIT_TEST_OUTPUT=str(output),
        WELM_NPU_USE_MEGAMOE="1",
        SGLANG_VLM_VISION_USE_ULYSSES="1",
        SGLANG_VLM_VISION_USE_DP="1",
    )
    subprocess.run(
        ["bash", str(ROOT / "examples/runtime/welm_vl/run_moonvit_950pr.sh")],
        env=env,
        check=True,
    )
    preflight, launch = [json.loads(line) for line in output.read_text().splitlines()]
    assert preflight["args"][-1] == "--runtime"
    args = launch["args"]
    assert args[args.index("--tp-size") + 1] == "4"
    assert args[args.index("--ep-size") + 1] == ep
    assert args[args.index("--model-path") + 1] == str(tmp_path / "weights with spaces")
    assert "--disable-prefill-cuda-graph" in args and "--disable-cuda-graph" in args
    assert "--enable-over-encoding" in args and "--enable-kv-mirror" in args
    assert set(launch["env"].values()) == {"0"}
