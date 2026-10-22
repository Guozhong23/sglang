"""Startup lifecycle and real image-DP warmup on CPU; no serving/NPU imports."""

import ast
import copy
import dataclasses
import enum
import importlib.util
import json
import math
import runpy
import sys
import weakref
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=15, suite="base-a-test-cpu")
COMPONENT = (
    ROOT
    / "python/sglang/srt/model_executor/model_runner_components/vision_encoder_warmup.py"
)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, class_name=None, **scope):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(
            node.body for node in nodes if getattr(node, "name", "") == class_name
        )
    tree.body = ast.parse("from __future__ import annotations").body + [
        node for node in nodes if getattr(node, "name", "") in names
    ]
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


WARM = load(COMPONENT, "moonvit_warmup_cpu_test")
DP = load(
    ROOT / "python/sglang/srt/layers/welmv4_vision_dp.py", "moonvit_warmup_dp_test"
)
LoadFormat = definitions(
    ROOT / "python/sglang/srt/configs/load_config.py", {"LoadFormat"}, enum=enum
).LoadFormat
ITEMS = definitions(
    ROOT / "python/sglang/srt/managers/schedule_batch.py",
    {"Modality", "MultimodalInputFormat", "MultimodalDataItem"},
    dataclasses=dataclasses,
    Enum=enum.Enum,
    auto=enum.auto,
)
HOOKS = definitions(
    ROOT / "python/sglang/srt/models/welm_moonvit_vl.py",
    {"_build_vision_warmup_items", "warmup_vision_encoder"},
    class_name="WeLMMoonViTVLForConditionalGeneration",
    torch=torch,
    math=math,
)


class DefaultLoader:
    pass


@pytest.fixture(autouse=True)
def serving_dependencies(monkeypatch):
    # Only heavyweight serving imports are replaced. Planning, encoding,
    # gathering, the model hook and the actual item dataclass execute unchanged.
    modules = {
        "sglang.srt.layers.welmv4_vision_dp": DP,
        "sglang.srt.configs.load_config": SimpleNamespace(LoadFormat=LoadFormat),
        "sglang.srt.model_loader.loader": SimpleNamespace(
            DefaultModelLoader=DefaultLoader
        ),
        "sglang.srt.managers.schedule_batch": ITEMS,
        "sglang.srt.model_executor.model_runner_components.vision_encoder_warmup": WARM,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    flag = SimpleNamespace(get=lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.environ",
        SimpleNamespace(envs=SimpleNamespace(SGLANG_VLM_VISION_DP_WARMUP=flag)),
    )
    return flag


def group(world=4, rank=0):
    return SimpleNamespace(
        world_size=world,
        rank_in_group=rank,
        ranks=list(range(world)),
        cpu_group=object(),
    )


def spec():
    return DP.VisionDPInputSpec((2, 2), (3, 14, 14), 8, True, 512)


def duplicate_status(monkeypatch, world=4, mutate=None):
    seen = []

    def gather(peers, status, group):
        seen.append(copy.deepcopy(status))
        for rank in range(world):
            peers[rank] = dict(copy.deepcopy(status), rank=rank)
        if mutate:
            mutate(peers)

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    return seen


class Device:
    def __init__(self):
        self.events = []

    def synchronize(self):
        self.events.append("sync")

    def stream(self, stream):
        self.events.append("stream")
        return nullcontext()

    def reset_peak_memory_stats(self):
        self.events.append("reset_peak")

    def memory_allocated(self):
        return 100

    memory_reserved = max_memory_allocated = max_memory_reserved = memory_allocated


def runner():
    device = SimpleNamespace(type="npu")
    model = SimpleNamespace(
        _vision_dp_group=group(),
        vision_encoder=SimpleNamespace(
            vision_dp_input_spec=spec(),
            dtype=torch.bfloat16,
            device=device,
            vision_dp_max_patches=0,
        ),
        vision_projector=SimpleNamespace(
            linear_2=SimpleNamespace(
                weight=SimpleNamespace(
                    shape=(8, 16), dtype=torch.bfloat16, device=device
                )
            )
        ),
        modules=lambda: [SimpleNamespace(training=False)],
        warmup_vision_encoder=Mock(),
    )
    return SimpleNamespace(
        model=model,
        device="npu",
        dtype=torch.bfloat16,
        is_draft_worker=False,
        server_args=SimpleNamespace(
            is_ep_scale_joiner=False,
            mm_enable_dp_encoder=True,
            enable_dp_attention=False,
            nnodes=1,
            skip_server_warmup=False,
            checkpoint_engine_wait_weights_before_ready=False,
        ),
        ps=SimpleNamespace(
            pp_size=1, dp_size=1, attn_cp_size=1, dcp_size=1, tp_size=4, attn_tp_size=4
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["WeLMMoonViTVLForConditionalGeneration"]
            )
        ),
        load_config=SimpleNamespace(load_format=LoadFormat.AUTO),
        loader=DefaultLoader(),
        tp_group=group(),
        attention_tp_group=group(),
        forward_stream=object(),
    )


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize(
    "profile,images,patches,rows",
    [
        (WARM.MOONVIT_PROFILES[0], 1, 256, 64),
        (WARM.MOONVIT_PROFILES[1], 12, 8192, 2048),
    ],
)
def test_profiles_use_production_plan_with_equal_nonempty_owners(
    world, profile, images, patches, rows
):
    plan = WARM.checked_profile_plan(profile, spec(), group(world))
    assert len(plan.grids) == images * world
    assert [len(indices) for indices in plan.assignments] == [images] * world
    assert [
        sum(plan.patch_counts[i] for i in indices) for indices in plan.assignments
    ] == [patches] * world
    assert plan.rows_per_rank == (rows,) * world


@pytest.mark.parametrize(
    "profile",
    [
        WARM.VisionWarmupProfile("bad", ((1, 3, 4),), 12),
        WARM.VisionWarmupProfile("bad", ((2, 2, 2),), 8),
        WARM.VisionWarmupProfile("bad", ((1, 512, 2),), 1024),
        WARM.VisionWarmupProfile("bad", ((1, 2, 2),), 8),
    ],
)
def test_bad_profiles_fail_before_encoding(profile):
    with pytest.raises(ValueError):
        WARM.checked_profile_plan(profile, spec(), group())


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("is_draft_worker", True),
        ("device", "cuda"),
        ("dtype", torch.float32),
        ("server_args.is_ep_scale_joiner", True),
        ("server_args.disaggregation_mode", "decode"),
        ("server_args.nnodes", 2),
        ("server_args.mm_enable_dp_encoder", False),
        ("server_args.skip_server_warmup", True),
        ("server_args.checkpoint_engine_wait_weights_before_ready", True),
        ("ps.pp_size", 2),
        ("ps.dp_size", 2),
        ("ps.attn_cp_size", 2),
        ("ps.dcp_size", 2),
        ("load_config.load_format", LoadFormat.DUMMY),
        ("load_config.load_format", LoadFormat.REMOTE),
        ("load_config.load_format", LoadFormat.IPC_CACHE),
        ("model_config.hf_config.architectures", ["WeLMV4VLMForConditionalGeneration"]),
    ],
)
def test_unsupported_workers_do_not_run_device_warmup(monkeypatch, attribute, value):
    obj = runner()
    owner = obj
    *parts, name = attribute.split(".")
    for part in parts:
        owner = getattr(owner, part)
    setattr(owner, name, value)
    duplicate_status(monkeypatch)
    WARM.maybe_warmup_vision_encoder(obj)
    obj.model.warmup_vision_encoder.assert_not_called()
    assert not getattr(obj, "_vision_encoder_warmup_done", False)


def test_disabled_flag_still_coordinates_and_global_skip_takes_precedence(
    monkeypatch, serving_dependencies, caplog
):
    serving_dependencies.get = lambda: False
    statuses = duplicate_status(monkeypatch)
    obj = runner()
    WARM.maybe_warmup_vision_encoder(obj)
    assert statuses[0]["manifest"]["enabled"] is False
    obj.server_args.skip_server_warmup = True
    WARM.maybe_warmup_vision_encoder(obj)
    assert statuses[-1]["manifest"]["skip_reason"] == "skip_server_warmup"
    obj.model.warmup_vision_encoder.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [("enabled", False), ("done", True), ("max_patches", 2048), ("profiles", [])],
)
def test_divergent_manifest_never_enters_device_work(monkeypatch, field, value):
    def change(peers):
        peers[1]["manifest"][field] = value
        peers[1]["digest"] = "different"

    statuses = duplicate_status(monkeypatch, mutate=change)
    obj = runner()
    with pytest.raises(ValueError, match="manifests differ"):
        WARM.maybe_warmup_vision_encoder(obj)
    assert len(statuses) == 1
    obj.model.warmup_vision_encoder.assert_not_called()


@pytest.mark.parametrize(
    "load_format", [LoadFormat.AUTO, LoadFormat.PT, LoadFormat.SAFETENSORS]
)
def test_completion_requires_stream_and_device_sync_and_is_idempotent(
    monkeypatch, load_format
):
    obj, device = runner(), Device()
    obj.server_args.skip_tokenizer_init = True
    obj.load_config.load_format = load_format
    statuses = duplicate_status(monkeypatch)
    monkeypatch.setattr(torch, "get_device_module", lambda _: device)

    def hook(*args):
        assert torch.is_inference_mode_enabled()
        assert device.events == ["sync", "reset_peak", "stream"]
        assert not getattr(obj, "_vision_encoder_warmup_done", False)
        device.events.append("hook")

    obj.model.warmup_vision_encoder.side_effect = hook
    WARM.maybe_warmup_vision_encoder(obj)
    assert device.events[-2:] == ["hook", "sync"]
    assert obj._vision_encoder_warmup_done
    WARM.maybe_warmup_vision_encoder(obj)
    obj.model.warmup_vision_encoder.assert_called_once()
    assert len(statuses) == 2 and statuses[-1]["manifest"]["done"]


@pytest.mark.parametrize("failure", ["hook", "final_sync"])
def test_failure_propagates_without_success_flag_or_exception_collectives(
    monkeypatch, failure
):
    obj, device = runner(), Device()
    statuses = duplicate_status(monkeypatch)
    monkeypatch.setattr(torch, "get_device_module", lambda _: device)
    if failure == "hook":
        obj.model.warmup_vision_encoder.side_effect = RuntimeError("injected")
    else:
        device.synchronize = Mock(side_effect=[None, RuntimeError("injected")])
    with pytest.raises(RuntimeError, match="injected"):
        WARM.maybe_warmup_vision_encoder(obj)
    assert not getattr(obj, "_vision_encoder_warmup_done", False)
    assert len(statuses) == 1


class Encoder(torch.nn.Module):
    dtype = torch.bfloat16
    device = torch.device("cpu")
    vision_dp_input_spec = spec()

    def __init__(self, cap=0):
        super().__init__()
        self.vision_dp_max_patches = cap
        self.calls = 0

    def forward(self, pixels, grids):
        self.calls += 1
        # A tiny deterministic encoder: real helper owns batching and device gather.
        return pixels[:, 0, 0, 0].reshape(-1, 4).mean(-1, keepdim=True)


class Projector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.expand(-1, 8).contiguous()


def cpu_model(cap=0, rank=0):
    obj = SimpleNamespace(
        vision_encoder=Encoder(cap),
        vision_projector=Projector(),
        _vision_dp_group=group(rank=rank),
    )
    obj._build_vision_warmup_items = lambda grids: HOOKS._build_vision_warmup_items(
        obj, grids
    )
    return obj


@pytest.mark.parametrize("cap,batch_count", [(0, 1), (1000, 11)])
def test_real_model_hook_runs_every_round_preserves_cap_rng_and_releases_outputs(
    monkeypatch, cap, batch_count
):
    torch.set_num_threads(1)
    obj, device = cpu_model(cap), Device()
    statuses = duplicate_status(monkeypatch)
    gathered = []

    def gather(output, padded):
        gathered.append(weakref.ref(output))
        output.copy_(padded.repeat(4, 1))

    obj._vision_dp_group.all_gather_into_tensor = gather
    rng = torch.get_rng_state().clone()
    items = obj._build_vision_warmup_items(WARM.MOONVIT_PROFILES[0].global_grids(4))
    assert all(
        isinstance(item, ITEMS.MultimodalDataItem)
        and item.hash is None
        and item.pad_value is None
        for item in items
    )
    assert items[0].feature is items[1].feature
    with torch.inference_mode():
        HOOKS.warmup_vision_encoder(
            obj, WARM.MOONVIT_PROFILES, device, "launch", "digest"
        )
    assert torch.equal(torch.get_rng_state(), rng)
    assert obj.vision_encoder.calls == obj.vision_projector.calls == 2 + 2 * batch_count
    assert len(gathered) == 4 and all(ref() is None for ref in gathered)
    assert device.events == ["sync"] * 4
    assert [(row["profile"], row["iteration"]) for row in statuses] == [
        ("S", 1),
        ("S", 2),
        ("B", 1),
        ("B", 2),
    ]
    for row in statuses[2:]:
        assert sum(batch["patches"] for batch in row["microbatches"]) == 8192
        assert len(row["microbatches"]) == batch_count
        if cap:
            assert (
                row["microbatches"][0]["patches"] == 1008
            )  # Soft cap never splits an image.
            assert any(batch["images"] == 2 for batch in row["microbatches"])
            assert all(
                batch["patches"] <= cap or batch["images"] == 1
                for batch in row["microbatches"]
            )
    assert (
        not obj.vision_encoder._forward_pre_hooks
        and not obj.vision_projector._forward_hooks
    )


@pytest.mark.parametrize("failure", ["encoder", "gather", "sync", "output", "summary"])
def test_round_failure_does_not_attempt_failure_collective(monkeypatch, failure):
    obj, device = cpu_model(), Device()
    statuses = duplicate_status(monkeypatch)
    obj._vision_dp_group.all_gather_into_tensor = lambda output, padded: output.copy_(
        padded.repeat(4, 1)
    )
    if failure == "encoder":
        obj.vision_encoder.forward = Mock(side_effect=RuntimeError("injected"))
    elif failure == "gather":
        obj._vision_dp_group.all_gather_into_tensor = Mock(
            side_effect=RuntimeError("injected")
        )
    elif failure == "sync":
        device.synchronize = Mock(side_effect=RuntimeError("injected"))
    elif failure == "output":
        monkeypatch.setattr(DP.VisionDPPlan, "gather", lambda *args: [])
    else:
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            Mock(side_effect=RuntimeError("injected")),
        )
    with torch.inference_mode(), pytest.raises((ValueError, RuntimeError)):
        HOOKS.warmup_vision_encoder(
            obj, WARM.MOONVIT_PROFILES, device, "launch", "digest"
        )
    assert not statuses
    assert (
        not obj.vision_encoder._forward_pre_hooks
        and not obj.vision_projector._forward_hooks
    )


def test_runner_hook_occurs_after_postload_before_kv_allocation():
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if getattr(node, "name", "") == "ModelRunner")
    init = next(node for node in cls.body if getattr(node, "name", "") == "initialize")
    calls = [
        node.value.func.attr
        for node in init.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
    ]
    assert calls[-2:] == ["configure_kv_cache_dtype", "maybe_warmup_vision_encoder"]
    assert calls.index("maybe_apply_post_load_model_transforms") < calls.index(
        "maybe_warmup_vision_encoder"
    )
    assert "alloc_memory_pool" not in calls


@pytest.mark.parametrize(
    "kind", ["no_hook", "no_group", "training", "projector", "negative_cap"]
)
def test_broken_supported_model_fails_instead_of_reporting_warmup_success(
    monkeypatch, kind
):
    obj = runner()
    duplicate_status(monkeypatch)
    if kind == "no_hook":
        del obj.model.warmup_vision_encoder
    elif kind == "no_group":
        obj.model._vision_dp_group = None
    elif kind == "training":
        obj.model.modules = lambda: [SimpleNamespace(training=True)]
    elif kind == "projector":
        obj.model.vision_projector.linear_2.weight.dtype = torch.float32
    else:
        obj.model.vision_encoder.vision_dp_max_patches = -1
    with pytest.raises(ValueError):
        WARM.maybe_warmup_vision_encoder(obj)
    assert not getattr(obj, "_vision_encoder_warmup_done", False)


@pytest.mark.parametrize("kind", ["single_rank", "other_spec", "resolved_loader"])
def test_unvalidated_configuration_is_explicitly_skipped(monkeypatch, kind):
    obj = runner()
    duplicate_status(monkeypatch)
    if kind == "single_rank":
        obj.model._vision_dp_group = group(1)
    elif kind == "other_spec":
        obj.model.vision_encoder.vision_dp_input_spec = dataclasses.replace(
            spec(), merge=(2, 4)
        )
    else:
        obj.loader = object()
    WARM.maybe_warmup_vision_encoder(obj)
    obj.model.warmup_vision_encoder.assert_not_called()


@pytest.mark.parametrize("missing", [None, "round", "complete", "memory", "ready"])
def test_manual_log_checker_requires_complete_coverage_and_ready(missing):
    checker = runpy.run_path(
        str(ROOT / "test/manual/ascend/check_moonvit_dp_warmup.py")
    )["check_log"]
    lines = []
    for rank in range(4):
        for profile, images, patches, rows in [
            ("S", 1, 256, 64),
            ("B", 12, 8192, 2048),
        ]:
            for iteration in [1, 2]:
                if (
                    missing == "round"
                    and rank == 2
                    and profile == "B"
                    and iteration == 2
                ):
                    continue
                record = dict(
                    launch="L",
                    manifest="D",
                    rank=rank,
                    profile=profile,
                    iteration=iteration,
                    success=True,
                    images=images,
                    patches=patches,
                    output_rows=rows,
                    microbatches=[dict(images=images, patches=patches)],
                )
                lines.append("Vision DP warmup round " + json.dumps(record))
        if not (missing == "complete" and rank == 2):
            lines.append(
                f"Vision DP warmup complete launch=L rank={rank} manifest=D elapsed_ms=1.0"
            )
        for stage in (
            "before_hook",
            "after_hook_released",
            "after_kv_allocation",
            "after_graph_initialization",
        ):
            if not (
                missing == "memory" and rank == 2 and stage == "after_kv_allocation"
            ):
                lines.append(
                    "Vision DP warmup memory "
                    + json.dumps(dict(launch="L", rank=rank, stage=stage))
                )
    if missing != "ready":
        lines.append("The server is fired up and ready to roll!")
    if missing:
        with pytest.raises(ValueError):
            checker("\n".join(lines), 4)
    else:
        result = checker("\n".join(lines), 4)
        assert result["startup_log_coverage_passed"]
        assert not result["performance_and_accuracy_validated"]


def _distributed_warmup_worker(rank, world, rendezvous, output_dir, mismatch):
    import torch.distributed as dist

    torch.set_num_threads(1)
    sys.modules["sglang.srt.layers.welmv4_vision_dp"] = DP
    sys.modules["sglang.srt.managers.schedule_batch"] = ITEMS
    sys.modules[
        "sglang.srt.model_executor.model_runner_components.vision_encoder_warmup"
    ] = WARM
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )
    obj = cpu_model(rank=rank)
    obj._vision_dp_group.cpu_group = dist.group.WORLD

    def gloo_device_gather(output, local):
        # Gloo writes on its worker thread, outside inference mode. Stage its
        # CPU result in a normal tensor while exercising the real DP payload.
        with torch.inference_mode(False):
            staging = torch.empty(output.shape, dtype=output.dtype)
            dist.all_gather_into_tensor(staging, local)
        output.copy_(staging)

    obj._vision_dp_group.all_gather_into_tensor = gloo_device_gather
    result = {}
    try:
        try:
            launch_id, digest = WARM.agree_warmup_manifest(
                obj._vision_dp_group,
                {
                    "enabled": not (mismatch and rank == 1),
                    "version": WARM.PROFILE_VERSION,
                },
            )
        except ValueError as exc:
            assert mismatch and "manifests differ" in str(exc)
            result["mismatch_detected"] = True
        else:
            assert not mismatch
            original_builder = obj._build_vision_warmup_items

            def distinct_items(grids):
                items = original_builder(grids)
                for index, item in enumerate(items):
                    item.feature = torch.full_like(item.feature, index / 128)
                return items

            obj._build_vision_warmup_items = distinct_items
            original_encode = DP.encode_vision_dp
            references = []

            def check_order(*args):
                outputs = original_encode(*args)
                for index, output in enumerate(outputs):
                    assert torch.all(output == index / 128)
                    references.append(weakref.ref(output))
                return outputs

            DP.encode_vision_dp = check_order
            device = Device()
            with torch.inference_mode():
                HOOKS.warmup_vision_encoder(
                    obj, WARM.MOONVIT_PROFILES, device, launch_id, digest
                )
            assert all(ref() is None for ref in references)
            result.update(
                encoder_calls=obj.vision_encoder.calls,
                projector_calls=obj.vision_projector.calls,
                completed_syncs=len(device.events),
            )
    finally:
        dist.destroy_process_group()
    Path(output_dir, f"{rank}.json").write_text(json.dumps(result))


@pytest.mark.parametrize("mismatch", [False, True])
def test_real_four_process_warmup_and_manifest_agreement(tmp_path, mismatch):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_warmup_worker,
        args=(4, str(tmp_path / "rendezvous"), str(tmp_path), mismatch),
        nprocs=4,
        join=True,
    )
    for rank in range(4):
        result = json.loads((tmp_path / f"{rank}.json").read_text())
        if mismatch:
            assert result == {"mismatch_detected": True}
        else:
            assert result == {
                "encoder_calls": 4,
                "projector_calls": 4,
                "completed_syncs": 4,
            }


@pytest.mark.parametrize("ranks", [[0, 1], [2, 3]])
def test_attention_dp_warmup_manifest_stays_in_request_group(monkeypatch, ranks):
    current = runner()
    current.server_args.enable_dp_attention = True
    current.ps.dp_size = 2
    current.ps.attn_tp_size = 2
    attention_group = group(world=2)
    attention_group.ranks = ranks
    current.model._vision_dp_group = attention_group
    current.attention_tp_group = attention_group
    device = Device()
    monkeypatch.setattr(torch, "get_device_module", lambda _: device)
    observed = []

    def gather(peers, status, *, group):
        assert group is attention_group.cpu_group
        assert group is not current.tp_group.cpu_group
        assert len(peers) == 2
        observed.append(status)
        peers[:] = [dict(copy.deepcopy(status), rank=rank) for rank in range(2)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    WARM.maybe_warmup_vision_encoder(current)
    assert current._vision_encoder_warmup_done
    current.model.warmup_vision_encoder.assert_called_once()
    manifest = observed[0]["manifest"]
    assert manifest["ranks"] == ranks and manifest["world_size"] == 2
    assert len(manifest["grids"][0]) == 2


def test_attention_dp_warmup_rejects_outer_world_group(monkeypatch):
    current = runner()
    current.server_args.enable_dp_attention = True
    current.ps.dp_size = current.ps.attn_tp_size = 2
    current.attention_tp_group = group(world=2)
    gather = Mock(side_effect=AssertionError("no collective on wrong group"))
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(ValueError, match="attention-TP group"):
        WARM.maybe_warmup_vision_encoder(current)
    gather.assert_not_called()
