#!/usr/bin/env python3
"""Summarize MegaMoE fused/reference tensor dumps produced by SGLang."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch


def _metrics(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    actual = payload["actual_output"].float().reshape(-1)
    reference = payload["reference_output"].float().reshape(-1)
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch in {path}: actual={tuple(actual.shape)} "
            f"reference={tuple(reference.shape)}"
        )

    difference = actual - reference
    actual_rms = torch.square(actual).mean().sqrt().item()
    reference_rms = torch.square(reference).mean().sqrt().item()
    rmse = torch.square(difference).mean().sqrt().item()
    denominator = max(reference_rms, torch.finfo(torch.float32).eps)
    cosine_denominator = max(
        torch.linalg.vector_norm(actual).item()
        * torch.linalg.vector_norm(reference).item(),
        torch.finfo(torch.float32).eps,
    )
    cosine = torch.dot(actual, reference).item() / cosine_denominator
    return {
        "layer": int(payload["layer_id"]),
        "call": int(payload["call_index"]),
        "rank": int(payload["ep_rank"]),
        "local_rows": int(payload["actual_output"].shape[0]),
        "actual_rms": actual_rms,
        "reference_rms": reference_rms,
        "rms_ratio": actual_rms / denominator,
        "max_abs": difference.abs().max().item() if difference.numel() else 0.0,
        "mean_abs": difference.abs().mean().item() if difference.numel() else 0.0,
        "rmse": rmse,
        "nrmse": rmse / denominator,
        "cosine": cosine,
        "file": str(path),
    }


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.8g}"
        return str(value)
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare MegaMoE fused outputs with local-EP reference outputs."
    )
    parser.add_argument("dump_dir", type=Path, help="MegaMoE dump root directory")
    parser.add_argument("--csv", type=Path, help="optional summary CSV path")
    args = parser.parse_args()

    paths = sorted(args.dump_dir.glob("rank_*/layer_*_call_*.pt"))
    if not paths:
        parser.error(f"no dump files found below {args.dump_dir}")

    rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(_metrics(payload, path))
    rows.sort(key=lambda row: (row["call"], row["layer"], row["rank"]))

    columns = [
        "layer",
        "call",
        "rank",
        "local_rows",
        "actual_rms",
        "reference_rms",
        "rms_ratio",
        "max_abs",
        "mean_abs",
        "rmse",
        "nrmse",
        "cosine",
        "file",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(_format(row[column]) for column in columns))

    worst = max(rows, key=lambda row: row["nrmse"])
    print(
        "\nWorst NRMSE: "
        f"layer={worst['layer']} call={worst['call']} rank={worst['rank']} "
        f"nrmse={worst['nrmse']:.8g} max_abs={worst['max_abs']:.8g} "
        f"cosine={worst['cosine']:.10f}"
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
