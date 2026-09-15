#!/usr/bin/env python3
"""Compare two backend-neutral WeLM MoE stage-dump directories."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable, Optional

import torch

_STAGES = (
    "moe_input",
    "router_logits",
    "topk_ids",
    "topk_weights",
    "routed_output",
    "shared_output",
    "moe_output",
)


def _load(root: Path) -> dict[tuple[int, int, int], tuple[dict[str, Any], Path]]:
    payloads: dict[tuple[int, int, int], tuple[dict[str, Any], Path]] = {}
    paths = sorted(root.rglob("layer_*_call_*.pt"))
    if not paths:
        raise ValueError(f"no WeLM MoE stage dumps found below {root}")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        key = (
            int(payload["layer_id"]),
            int(payload["call_index"]),
            int(payload["rank"]),
        )
        if key in payloads:
            raise ValueError(
                f"duplicate layer/call/rank {key} below {root}; pass one tag "
                "directory rather than a parent containing multiple runs"
            )
        payloads[key] = (payload, path)
    return payloads


def _base_row(
    key: tuple[int, int, int],
    stage: str,
    gold_payload: dict[str, Any],
    test_payload: dict[str, Any],
    gold_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    return {
        "layer": key[0],
        "call": key[1],
        "rank": key[2],
        "stage": stage,
        "gold_tag": gold_payload.get("tag", ""),
        "test_tag": test_payload.get("tag", ""),
        "gold_backend": gold_payload.get("backend", ""),
        "test_backend": test_payload.get("backend", ""),
        "status": "ok",
        "shape": "",
        "gold_dtype": "",
        "test_dtype": "",
        "gold_rms": "",
        "test_rms": "",
        "rms_ratio": "",
        "max_abs": "",
        "mean_abs": "",
        "p50_abs": "",
        "p90_abs": "",
        "p99_abs": "",
        "rmse": "",
        "nrmse": "",
        "cosine": "",
        "exact_match_rate": "",
        "topk_set_match_rate": "",
        "gold_file": str(gold_path),
        "test_file": str(test_path),
    }


def _compare_stage(
    key: tuple[int, int, int],
    stage: str,
    gold_payload: dict[str, Any],
    test_payload: dict[str, Any],
    gold_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    row = _base_row(key, stage, gold_payload, test_payload, gold_path, test_path)
    gold = gold_payload.get(stage)
    test = test_payload.get(stage)
    if gold is None or test is None:
        row["status"] = "both_none" if gold is None and test is None else "missing"
        return row
    if not isinstance(gold, torch.Tensor) or not isinstance(test, torch.Tensor):
        row["status"] = "not_tensor"
        return row

    row["shape"] = str(tuple(gold.shape))
    row["gold_dtype"] = str(gold.dtype)
    row["test_dtype"] = str(test.dtype)
    if tuple(gold.shape) != tuple(test.shape):
        row["status"] = f"shape_mismatch:{tuple(test.shape)}"
        return row

    exact = gold == test
    row["exact_match_rate"] = exact.float().mean().item() if exact.numel() else 1.0
    if stage == "topk_ids" and gold.ndim == 2:
        gold_sets = torch.sort(gold.to(torch.int64), dim=-1).values
        test_sets = torch.sort(test.to(torch.int64), dim=-1).values
        row["topk_set_match_rate"] = (
            (gold_sets == test_sets).all(dim=-1).float().mean().item()
            if gold.shape[0]
            else 1.0
        )

    if not (gold.is_floating_point() or test.is_floating_point()):
        return row

    gold_f = gold.float().reshape(-1)
    test_f = test.float().reshape(-1)
    finite = torch.isfinite(gold_f) & torch.isfinite(test_f)
    if not bool(finite.all()):
        row["status"] = (
            f"nonfinite:{int(finite.sum().item())}/{finite.numel()}"
        )
    gold_f = torch.where(finite, gold_f, torch.zeros_like(gold_f))
    test_f = torch.where(finite, test_f, torch.zeros_like(test_f))
    difference = test_f - gold_f
    absolute = difference.abs()
    gold_rms = torch.square(gold_f).mean().sqrt().item() if gold_f.numel() else 0.0
    test_rms = torch.square(test_f).mean().sqrt().item() if test_f.numel() else 0.0
    rmse = torch.square(difference).mean().sqrt().item() if difference.numel() else 0.0
    epsilon = torch.finfo(torch.float32).eps
    denominator = max(gold_rms, epsilon)
    norm_product = (
        torch.linalg.vector_norm(gold_f).item()
        * torch.linalg.vector_norm(test_f).item()
    )
    cosine = (
        torch.dot(gold_f, test_f).item() / norm_product
        if norm_product > epsilon
        else (1.0 if rmse == 0.0 else 0.0)
    )
    if absolute.numel():
        p50, p90, p99 = torch.quantile(
            absolute, torch.tensor([0.5, 0.9, 0.99])
        ).tolist()
    else:
        p50 = p90 = p99 = 0.0
    row.update(
        {
            "gold_rms": gold_rms,
            "test_rms": test_rms,
            "rms_ratio": test_rms / denominator,
            "max_abs": absolute.max().item() if absolute.numel() else 0.0,
            "mean_abs": absolute.mean().item() if absolute.numel() else 0.0,
            "p50_abs": p50,
            "p90_abs": p90,
            "p99_abs": p99,
            "rmse": rmse,
            "nrmse": rmse / denominator,
            "cosine": cosine,
        }
    )
    return row


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else str(value)
    return str(value)


def _first_problem(
    rows: list[dict[str, Any]],
    stage: str,
    field: str,
    predicate: Callable[[float], bool],
) -> Optional[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["stage"] == stage
        and isinstance(row.get(field), (float, int))
        and predicate(float(row[field]))
    ]
    return (
        min(candidates, key=lambda row: (row["call"], row["layer"], row["rank"]))
        if candidates
        else None
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare paired DeepEP/reference and MegaMoE WeLM stage dumps."
    )
    parser.add_argument("gold_dir", type=Path, help="gold tag dump directory")
    parser.add_argument("test_dir", type=Path, help="test tag dump directory")
    parser.add_argument("--csv", type=Path, help="optional detailed CSV output")
    args = parser.parse_args()

    gold = _load(args.gold_dir)
    test = _load(args.test_dir)
    common = sorted(set(gold) & set(test), key=lambda key: (key[1], key[0], key[2]))
    if not common:
        parser.error("the two dump directories have no common layer/call/rank keys")
    missing_test = sorted(set(gold) - set(test))
    missing_gold = sorted(set(test) - set(gold))
    if missing_test or missing_gold:
        print(
            f"WARNING: unmatched keys missing_test={missing_test} "
            f"missing_gold={missing_gold}"
        )

    rows: list[dict[str, Any]] = []
    for key in common:
        gold_payload, gold_path = gold[key]
        test_payload, test_path = test[key]
        for stage in _STAGES:
            rows.append(
                _compare_stage(
                    key,
                    stage,
                    gold_payload,
                    test_payload,
                    gold_path,
                    test_path,
                )
            )

    columns = list(rows[0])
    print(",".join(columns))
    for row in rows:
        print(",".join(_format(row[column]) for column in columns))

    numeric = [
        row
        for row in rows
        if isinstance(row.get("nrmse"), (float, int)) and row["status"] == "ok"
    ]
    if numeric:
        worst = max(numeric, key=lambda row: float(row["nrmse"]))
        print(
            "\nWorst NRMSE: "
            f"layer={worst['layer']} call={worst['call']} rank={worst['rank']} "
            f"stage={worst['stage']} nrmse={worst['nrmse']:.9g} "
            f"cosine={worst['cosine']:.10f}"
        )
    first_input = _first_problem(rows, "moe_input", "nrmse", lambda value: value > 0)
    first_route = _first_problem(
        rows, "topk_ids", "exact_match_rate", lambda value: value < 1
    )
    if first_input:
        print(
            "First non-identical MoE input: "
            f"call={first_input['call']} layer={first_input['layer']} "
            f"rank={first_input['rank']} nrmse={first_input['nrmse']:.9g}"
        )
    if first_route:
        print(
            "First TopK id mismatch: "
            f"call={first_route['call']} layer={first_route['layer']} "
            f"rank={first_route['rank']} "
            f"exact_match_rate={first_route['exact_match_rate']:.9g} "
            f"set_match_rate={first_route['topk_set_match_rate']}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved to: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
