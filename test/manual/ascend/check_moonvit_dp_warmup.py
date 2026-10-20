"""Check startup coverage logs; this does not certify HCCL timing or accuracy.

Run from the checkout root:
  python test/manual/ascend/check_moonvit_dp_warmup.py server.log --world-size 4
Use separate logs for each full worker restart. See the workspace's
09_markdown/WeLM_MoonViT_DP_warmup_implementation_20261020.md for A/B acceptance.
"""

import argparse
import json
import re
from pathlib import Path


def check_log(text, world_size, launch=None, expect_disabled=False):
    rounds, memories, complete = [], [], []
    for line in text.splitlines():
        for marker, output in (
            ("Vision DP warmup round ", rounds),
            ("Vision DP warmup memory ", memories),
        ):
            if marker in line:
                output.append(json.loads(line.split(marker, 1)[1]))
        match = re.search(
            r"Vision DP warmup complete launch=(\S+) rank=(\d+) manifest=(\S+) elapsed_ms=([\d.]+)",
            line,
        )
        if match:
            complete.append(
                dict(
                    launch=match[1],
                    rank=int(match[2]),
                    manifest=match[3],
                    elapsed_ms=float(match[4]),
                )
            )
    if "The server is fired up and ready to roll!" not in text:
        raise ValueError(
            "Server ready log is missing; startup coverage is not serving readiness"
        )
    if expect_disabled:
        if (
            rounds
            or complete
            or "Vision DP warmup skipped: SGLANG_VLM_VISION_DP_WARMUP=0" not in text
        ):
            raise ValueError("Expected the new stage disabled, without warmup rounds")
        return dict(new_warmup_disabled=True, ready_log_observed=True)
    launches = {record["launch"] for record in rounds}
    if launch is None:
        if len(launches) != 1:
            raise ValueError(
                f"Expected one launch, got {launches}; use a fresh log or --launch"
            )
        launch = launches.pop()
    rounds = [record for record in rounds if record["launch"] == launch]
    memories = [record for record in memories if record["launch"] == launch]
    complete = [record for record in complete if record["launch"] == launch]
    expected = [("S", 1), ("S", 2), ("B", 1), ("B", 2)]
    if sorted(record["rank"] for record in complete) != list(range(world_size)):
        raise ValueError("Not every rank completed the plan exactly once")
    digests = {record["manifest"] for record in rounds + complete}
    if len(digests) != 1:
        raise ValueError("Ranks reported different manifests")
    if {record["rank"] for record in rounds} != set(range(world_size)):
        raise ValueError("Round rank set does not match the expected vision group")
    summaries = []
    for rank in range(world_size):
        local = [record for record in rounds if record["rank"] == rank]
        if [(record["profile"], record["iteration"]) for record in local] != expected:
            raise ValueError(f"Rank {rank} did not execute S1/S2/B1/B2 exactly once")
        for record in local:
            images, patches, rows = (
                (1, 256, 64) if record["profile"] == "S" else (12, 8192, 2048)
            )
            if (
                not record["success"]
                or record["images"] != images
                or record["patches"] != patches
                or record["output_rows"] != rows
                or sum(batch["images"] for batch in record["microbatches"]) != images
                or sum(batch["patches"] for batch in record["microbatches"]) != patches
            ):
                raise ValueError(f"Rank {rank} has incomplete encoder coverage")
        memory = [record for record in memories if record["rank"] == rank]
        stages = {record["stage"] for record in memory}
        if (
            not {
                "before_hook",
                "after_hook_released",
                "after_kv_allocation",
                "after_graph_initialization",
            }
            <= stages
        ):
            raise ValueError(f"Rank {rank} is missing memory checkpoints")
        summaries.append(dict(rank=rank, rounds=local, memory=memory))
    return dict(
        launch=launch,
        manifest=digests.pop(),
        world_size=world_size,
        startup_log_coverage_passed=True,
        ready_log_observed=True,
        performance_and_accuracy_validated=False,
        ranks=summaries,
        complete=complete,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--launch")
    parser.add_argument("--expect-disabled", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_log(
        args.log.read_text(errors="replace"),
        args.world_size,
        args.launch,
        args.expect_disabled,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
