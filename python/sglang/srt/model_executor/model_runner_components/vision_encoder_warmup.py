# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Startup-only, cache-free image-DP warmup. No request-path collectives.

Failures propagate to worker initialization. In particular, never add a
collective in an exception handler: peers may already be in device AllGather.
CPU-group timeouts do not bound total startup time; deployments need an outer
startup deadline. The original HTTP warmup remains responsible for end-to-end
serving readiness.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass

import torch

logger = logging.getLogger(__name__)
PROFILE_VERSION = "moonvit-dp-v1"


@dataclass(frozen=True)
class VisionWarmupProfile:
    name: str
    rank_grids: tuple[tuple[int, int, int], ...]
    patches_per_rank: int
    rounds: int = 2

    def global_grids(self, world_size):
        return tuple(grid for grid in self.rank_grids for _ in range(world_size))


MOONVIT_PROFILES = (
    VisionWarmupProfile("S", ((1, 16, 16),), 256),
    VisionWarmupProfile(
        "B",
        tuple(
            (1, h, w)
            for h, w in ((36, 28), (36, 24), (26, 28), (22, 28), (20, 28), (16, 20))
            for _ in range(2)
        ),
        8192,
    ),
)


def checked_profile_plan(profile, spec, group):
    from sglang.srt.layers.welmv4_vision_dp import VisionDPPlan

    plan = VisionDPPlan(
        profile.global_grids(group.world_size),
        spec.merge,
        group.world_size,
        group.rank_in_group,
    )
    if (
        profile.rounds != 2
        or sum(map(math.prod, profile.rank_grids)) != profile.patches_per_rank
    ):
        raise ValueError(f"Vision warmup {profile.name}: invalid rounds/patch budget")
    for t, h, w in plan.grids:
        if t != 1 or (spec.grid_limit is not None and max(h, w) >= spec.grid_limit):
            raise ValueError(
                f"Vision warmup {profile.name}: invalid image grid {(t, h, w)}"
            )
    for rank, indices in enumerate(plan.assignments):
        if (
            not indices
            or Counter(plan.grids[i] for i in indices) != Counter(profile.rank_grids)
            or sum(plan.patch_counts[i] for i in indices) != profile.patches_per_rank
            or plan.rows_per_rank[rank]
            != profile.patches_per_rank // math.prod(spec.merge)
        ):
            raise ValueError(
                f"Vision warmup {profile.name}: owner/budget mismatch on rank {rank}"
            )
    return plan


def _gather_status(group, status):
    peers = [None] * group.world_size
    torch.distributed.all_gather_object(peers, status, group=group.cpu_group)
    if [peer["rank"] for peer in peers] != list(range(group.world_size)):
        raise ValueError("Vision warmup status does not cover the vision group")
    return peers


def agree_warmup_manifest(group, manifest):
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    peers = _gather_status(
        group,
        dict(
            rank=group.rank_in_group,
            manifest=manifest,
            digest=digest,
            pid=os.getpid(),
            time_ns=time.time_ns(),
        ),
    )
    if any(peer["digest"] != digest for peer in peers):
        differing = sorted(
            key
            for key in manifest
            if any(peer["manifest"].get(key) != manifest[key] for peer in peers)
        )
        raise ValueError(f"Vision warmup manifests differ across ranks: {differing}")
    return f"{peers[0]['pid']}-{peers[0]['time_ns']}", digest


def _skip_reason(runner, enabled):
    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.model_loader.loader import DefaultModelLoader

    if runner.server_args.skip_server_warmup:
        return "skip_server_warmup"
    if not enabled:
        return "SGLANG_VLM_VISION_DP_WARMUP=0"
    if runner.server_args.checkpoint_engine_wait_weights_before_ready:
        return "weights_not_ready"
    if runner.load_config.load_format not in (
        LoadFormat.AUTO,
        LoadFormat.PT,
        LoadFormat.SAFETENSORS,
    ):
        return "unsupported_load_format"
    if type(runner.loader) is not DefaultModelLoader:
        return "unsupported_resolved_loader"
    return None


def log_warmup_memory(runner, stage):
    device = torch.get_device_module(runner.device)
    record = dict(
        stage=stage,
        pid=os.getpid(),
        device=str(runner.model.vision_encoder.device),
        rank=runner.model._vision_dp_group.rank_in_group,
        allocated_bytes=device.memory_allocated(),
        reserved_bytes=device.memory_reserved(),
        peak_allocated_bytes=device.max_memory_allocated(),
        peak_reserved_bytes=device.max_memory_reserved(),
        launch=getattr(runner, "_vision_encoder_warmup_launch", None),
    )
    if hasattr(runner, "max_total_num_tokens"):
        record["kv_tokens"] = runner.max_total_num_tokens
        record["effective_kv_tokens"] = runner.effective_max_total_num_tokens
    logger.info("Vision DP warmup memory %s", json.dumps(record, sort_keys=True))


def maybe_warmup_vision_encoder(runner):
    from sglang.srt.environ import envs

    args, model, ps = runner.server_args, runner.model, runner.ps
    architectures = runner.model_config.hf_config.architectures or []
    if "WeLMMoonViTVLForConditionalGeneration" not in architectures:
        return
    # Joiners must not contact existing workers that are serving, not starting.
    reason = None
    if runner.is_draft_worker:
        reason = "draft_worker"
    elif args.is_ep_scale_joiner:
        reason = "elastic_joiner"
    elif runner.device != "npu" or runner.dtype != torch.bfloat16:
        reason = "requires_npu_bf16"
    elif not args.mm_enable_dp_encoder:
        reason = "vision_tp"
    elif (
        args.nnodes != 1
        or ps.pp_size != 1
        or ps.dp_size != 1
        or ps.attn_cp_size != 1
        or ps.dcp_size != 1
    ):
        reason = "unsupported_topology"
    if reason:
        logger.info("Vision DP warmup skipped: %s", reason)
        return

    group = getattr(model, "_vision_dp_group", None)
    if group is None:
        raise ValueError("MoonViT image DP warmup requires its vision group")
    if group.world_size == 1:
        logger.info("Vision DP warmup skipped: single_rank")
        return
    if (
        group.world_size != ps.tp_size
        or group.world_size != ps.attn_tp_size
        or list(group.ranks) != list(runner.tp_group.ranks)
    ):
        raise ValueError(
            "Vision warmup requires the complete single-node text TP group"
        )
    encoder, projector = model.vision_encoder, model.vision_projector
    spec = encoder.vision_dp_input_spec
    enabled = envs.SGLANG_VLM_VISION_DP_WARMUP.get()
    reason = _skip_reason(runner, enabled)
    if reason is None and (
        spec.merge != (2, 2)
        or spec.pixel_tail_shape != (3, 14, 14)
        or not spec.image_only
        or spec.grid_limit != 512
    ):
        reason = "unsupported_profile_spec"
    plans = (
        [checked_profile_plan(profile, spec, group) for profile in MOONVIT_PROFILES]
        if reason is None
        else []
    )
    manifest = dict(
        version=PROFILE_VERSION,
        enabled=enabled,
        skip_reason=reason,
        done=getattr(runner, "_vision_encoder_warmup_done", False),
        model=f"{type(model).__module__}.{type(model).__qualname__}",
        dtype=str(runner.dtype),
        encoder_dtype=str(encoder.dtype),
        world_size=group.world_size,
        ranks=list(group.ranks),
        spec=asdict(spec),
        max_patches=encoder.vision_dp_max_patches,
        profiles=[asdict(profile) for profile in MOONVIT_PROFILES],
        grids=[plan.grids for plan in plans],
        owners=[plan.assignments for plan in plans],
        load_format=str(runner.load_config.load_format),
        hook=callable(getattr(model, "warmup_vision_encoder", None)),
    )
    launch_id, digest = agree_warmup_manifest(group, manifest)
    if group.rank_in_group == 0 and not manifest["done"]:
        logger.info(
            "Vision DP warmup manifest %s",
            json.dumps(
                dict(launch=launch_id, digest=digest, **manifest), sort_keys=True
            ),
        )
    if reason:
        logger.info("Vision DP warmup skipped: %s (manifest=%s)", reason, digest)
        return
    if manifest["done"]:
        return
    if not manifest["hook"]:
        raise ValueError("MoonViT image DP model has no warmup hook")
    if any(module.training for module in model.modules()):
        raise ValueError("Vision warmup requires an already-eval model")
    if (
        encoder.dtype != torch.bfloat16
        or encoder.device.type != "npu"
        or spec.output_hidden_size <= 0
        or encoder.vision_dp_max_patches < 0
    ):
        raise ValueError("Vision warmup has invalid dtype/device/width/patch cap")
    projection = projector.linear_2.weight
    if (
        projection.shape[0] != spec.output_hidden_size
        or projection.dtype != encoder.dtype
        or projection.device != encoder.device
    ):
        raise ValueError(
            "Vision warmup projector width/dtype/device does not match encoder"
        )
    device = torch.get_device_module(runner.device)
    runner._vision_encoder_warmup_launch = launch_id
    device.synchronize()  # Weight loading may have used another stream.
    log_warmup_memory(runner, "before_hook")
    device.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode(), device.stream(runner.forward_stream):
        model.warmup_vision_encoder(MOONVIT_PROFILES, device, launch_id, digest)
    device.synchronize()
    # Hook-local tensors are now released. Peak counters cover the whole hook.
    log_warmup_memory(runner, "after_hook_released")
    runner._vision_encoder_warmup_done = True
    logger.info(
        "Vision DP warmup complete launch=%s rank=%d manifest=%s elapsed_ms=%.3f",
        launch_id,
        group.rank_in_group,
        digest,
        (time.perf_counter() - start) * 1000,
    )


def run_vision_dp_warmup(
    encoder, projector, group, profiles, build_items, device, launch_id, digest
):
    from sglang.srt.layers.welmv4_vision_dp import encode_vision_dp

    spec = encoder.vision_dp_input_spec
    for profile in profiles:
        plan = checked_profile_plan(profile, spec, group)
        items = build_items(plan.grids)
        if len(items) != len(plan.grids):
            raise ValueError("Vision warmup item count differs from its plan")
        for item, grid, patches in zip(items, plan.grids, plan.patch_counts):
            spec.validate(item, grid, patches)
            if (
                item.feature.dtype != encoder.dtype
                or item.feature.device.type != "cpu"
                or item.image_grid_thw.dtype != torch.int64
                or item.image_grid_thw.device.type != "cpu"
                or item.image_grid_thw.tolist() != [list(grid)]
            ):
                raise ValueError(
                    "Vision warmup item dtype/device/grid differs from plan"
                )
        for iteration in range(1, profile.rounds + 1):
            calls, projection_rows = [], []

            def observe_encoder(module, inputs):
                pixels, grids = inputs
                calls.append(
                    dict(images=len(grids), patches=len(pixels), grids=grids.tolist())
                )

            def observe_projector(module, inputs, output):
                projection_rows.append(int(output.shape[0]))

            start = time.perf_counter()
            with (
                encoder.register_forward_pre_hook(observe_encoder),
                projector.register_forward_hook(observe_projector),
            ):
                with torch.profiler.record_function(
                    f"vision_dp_warmup/{profile.name}/{iteration}"
                ):
                    outputs = encode_vision_dp(items, encoder, projector, group)
                call_ms = (time.perf_counter() - start) * 1000
                device.synchronize()
            complete_ms = (time.perf_counter() - start) * 1000
            owned_grids = [list(plan.grids[i]) for i in plan.assignments[plan.rank]]
            if [grid for call in calls for grid in call["grids"]] != owned_grids or [
                call["patches"] // math.prod(spec.merge) for call in calls
            ] != projection_rows:
                raise ValueError(
                    "Vision warmup did not execute the expected encoder/projector batches"
                )
            if len(outputs) != len(items) or any(
                tuple(output.shape) != (rows, spec.output_hidden_size)
                or output.dtype != encoder.dtype
                or output.device != encoder.device
                for output, rows in zip(outputs, plan.token_counts)
            ):
                raise ValueError(
                    "Vision warmup gathered outputs differ from the image plan"
                )
            del outputs
            status = dict(
                launch=launch_id,
                manifest=digest,
                profile=profile.name,
                iteration=iteration,
                rank=group.rank_in_group,
                pid=os.getpid(),
                device=str(encoder.device),
                images=len(owned_grids),
                patches=profile.patches_per_rank,
                output_rows=plan.rows_per_rank[plan.rank],
                microbatches=calls,
                call_ms=call_ms,
                device_complete_ms=complete_ms,
                success=True,
            )
            peers = _gather_status(
                group, status
            )  # Success path only, after device completion.
            for rank, peer in enumerate(peers):
                if (
                    any(
                        peer[key] != status[key]
                        for key in (
                            "launch",
                            "manifest",
                            "profile",
                            "iteration",
                            "success",
                        )
                    )
                    or peer["images"] != len(plan.assignments[rank])
                    or peer["patches"] != profile.patches_per_rank
                    or peer["output_rows"] != plan.rows_per_rank[rank]
                ):
                    raise ValueError(
                        "Vision warmup completion summary differs across ranks"
                    )
            status["profile_total_ms"] = (time.perf_counter() - start) * 1000
            logger.info("Vision DP warmup round %s", json.dumps(status, sort_keys=True))
            if group.rank_in_group == 0:
                logger.info(
                    "Vision DP warmup coverage %s", json.dumps(peers, sort_keys=True)
                )
        del items
