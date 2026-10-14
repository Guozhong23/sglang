r"""Opt-in BF16 MegaMoE shared-expert device acceptance test.

Run with the matched shared-expert wheel and CANN custom operator environment::

    torchrun --standalone --nproc-per-node=4 \
        test/manual/ascend/test_welmv4_megamoe_shared_npu.py

Every worker participates in a real HCCL EP group. The test compares routed-only
MegaMoE plus an independent BF16 shared FFN with fused eager and native NPU graph
replay. It changes inputs, routing, and padding within each captured token bucket.
Use --shared-experts 2 for the documented multiple-shared-expert case, and
--rows 32,64,128,256,384,512,640,768,1024,1280,1536,1792,2048,2560,3072,4128
for the TP4 local-row capacities of the example serving graph buckets.

This is an operator acceptance test, not a model/BreakableGraph benchmark. It
does not install packages, start a server, or run during unittest discovery.
Buckets run sequentially; simultaneous graphs with interleaved bucket replay
must also be checked in the model's end-to-end acceptance run.
"""

import argparse
import inspect
import os
import sys
from datetime import timedelta


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", default="128,520", help="Per-rank physical rows")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--routed-intermediate", type=int, default=512)
    parser.add_argument("--shared-intermediate", type=int, default=512)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--shared-experts", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--replays", type=int, default=8, help="At least four replays")
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--eager-only", action="store_true", help="Diagnostic only")
    args = parser.parse_args()
    try:
        args.rows = sorted({int(value) for value in args.rows.split(",")})
    except ValueError:
        parser.error("--rows must be comma-separated positive integers")
    if not args.rows or min(args.rows) <= 0:
        parser.error("--rows must be positive")
    dimensions = (args.hidden, args.routed_intermediate, args.shared_intermediate)
    if any(size <= 0 or size % 16 for size in dimensions):
        parser.error(
            "BF16 hidden/intermediate dimensions must be positive multiples of 16"
        )
    if args.experts_per_rank <= 0 or args.topk <= 0 or args.replays < 4:
        parser.error(
            "experts/topk must be positive and --replays must be at least four"
        )
    if args.atol < 0 or args.rtol < 0 or args.timeout_seconds <= 0:
        parser.error("tolerances must be nonnegative and timeout must be positive")
    return args


def main():
    args = parse_args()  # --help works without importing torch or compiling the wheel.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2 or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Use torchrun with at least two EP workers on Ascend 950")
    num_experts = world_size * args.experts_per_rank
    if num_experts > 1024 or args.topk > num_experts:
        raise ValueError("Require topk <= total routed experts <= 1024")

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("This acceptance test requires an Ascend NPU environment")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout_seconds))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    group._get_backend(device).get_hccl_comm_name(rank, init_comm=True)

    from npu_ops_transformer.ops.comm_context import comm_context_op_builder
    from npu_ops_transformer.ops.mega_moe import (
        get_symm_buffer_for_mega_moe,
        mega_moe,
    )

    shared_names = {"l1_weights_shared", "l2_weights_shared"}
    if not shared_names.issubset(inspect.signature(mega_moe).parameters):
        raise RuntimeError("Installed MegaMoE wrapper lacks the shared-expert ABI")
    if rank == 0:
        print(f"MegaMoE wrapper: {inspect.getsourcefile(mega_moe)}", flush=True)
        print(f"EP={world_size}, settings={vars(args)}", flush=True)
    comm_context_op_builder.load()
    dist.barrier(group=group)
    torch.npu.synchronize()
    symm_buffer = get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=max(args.rows),
        num_topk=args.topk,
        hidden=args.hidden,
        intermediate_hidden=0,
        max_recv_token_num=0,
        dispatch_quant_mode=0,
        dispatch_quant_out_dtype=28,
        combine_quant_mode=0,
        comm_alg="",
    )

    def weight(shape, seed, fan_in):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return (torch.randn(shape, generator=generator) / fan_in**0.5).to(
            device=device, dtype=torch.bfloat16
        )

    w1 = weight(
        (args.experts_per_rank, 2 * args.routed_intermediate, args.hidden),
        1100 + rank,
        args.hidden,
    )
    w2 = weight(
        (args.experts_per_rank, args.hidden, args.routed_intermediate),
        2100 + rank,
        args.routed_intermediate,
    )
    # Complete shared FFNs are identical on all EP ranks, with no TP slicing.
    shared_w1 = weight(
        (args.shared_experts, 2 * args.shared_intermediate, args.hidden),
        3100,
        args.hidden,
    )
    shared_w2 = weight(
        (args.shared_experts, args.hidden, args.shared_intermediate),
        4100,
        args.shared_intermediate,
    )

    def check(output, reference, counts, expected_counts, valid, label):
        # Move completed results to CPU before checking; never do this in capture.
        actual = output.cpu()
        expected = reference.cpu()
        error = None
        try:
            assert actual.shape == expected.shape
            assert actual.dtype == torch.bfloat16
            assert torch.isfinite(actual).all()
            assert torch.isfinite(expected).all()
            assert not actual[~valid.cpu()].count_nonzero()
            assert counts.shape == (args.experts_per_rank,)
            assert counts.dtype == torch.int32
            torch.testing.assert_close(
                counts.cpu(), expected_counts.cpu(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual, expected, rtol=args.rtol, atol=args.atol
            )
        except AssertionError as exc:
            error = str(exc)
        # All ranks agree on a numerical failure before issuing another MegaMoE.
        failed = torch.tensor(
            int(error is not None), device=device, dtype=torch.int32
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        if failed.item():
            if error is not None:
                print(f"rank={rank}, {label}: {error}", file=sys.stderr, flush=True)
            raise AssertionError(f"{label} failed on at least one EP rank")
        max_error = (actual.float() - expected.float()).abs().max().to(device)
        dist.all_reduce(max_error, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(
                f"PASS {label}: global_max_abs_error={max_error.item():.6g}",
                flush=True,
            )

    def run_bucket(rows):
        x = torch.empty((rows, args.hidden), device=device, dtype=torch.bfloat16)
        ids = torch.empty((rows, args.topk), device=device, dtype=torch.int32)
        probabilities = torch.empty_like(ids, dtype=torch.bfloat16)
        valid = torch.empty(rows, device=device, dtype=torch.bool)
        wrapper_calls = 0

        def update_inputs(step):
            generator = torch.Generator(device="cpu").manual_seed(
                5100 + rank * 10000 + rows * 31 + step
            )
            input_cpu = torch.randn((rows, args.hidden), generator=generator)
            # Consecutive IDs modulo total experts are distinct for each token.
            starts = torch.randint(num_experts, (rows, 1), generator=generator)
            ids_cpu = (starts + torch.arange(args.topk)) % num_experts
            weights_cpu = torch.rand((rows, args.topk), generator=generator)
            weights_cpu /= weights_cpu.sum(dim=-1, keepdim=True)
            mask = torch.ones(rows, dtype=torch.bool)
            pattern = step % 4
            if pattern == 1:
                mask[max(1, rows // 2) :] = False
            elif pattern == 2:
                mask[::3] = False
            elif pattern == 3:
                # Physical rows remain equal across ranks, including an empty peer.
                if rank == (step // 4) % world_size:
                    mask[:] = False
                else:
                    mask[-1:] = False
            input_cpu[~mask] = 0
            weights_cpu[~mask] = 0
            x.copy_(input_cpu)
            ids.copy_(ids_cpu)
            probabilities.copy_(weights_cpu)
            valid.copy_(mask)
            # Padding uses legal IDs and zero weights; it still gets dispatched.
            histogram = torch.bincount(ids_cpu.flatten(), minlength=num_experts)
            histogram = histogram.to(device=device, dtype=torch.int32)
            dist.all_reduce(histogram, group=group)
            begin = rank * args.experts_per_rank
            return histogram[begin : begin + args.experts_per_rank].clone()

        def call(fuse_shared):
            nonlocal wrapper_calls
            wrapper_calls += 1
            output, counts = mega_moe(
                x=x,
                topk_ids=ids,
                topk_weights=probabilities,
                l1_weights=[w1],
                l2_weights=[w2],
                sym_buffer=symm_buffer,
                scales=None,
                l1_weights_sf=None,
                l2_weights_sf=None,
                x_active_mask=None,
                weight1_type=28,
                weight2_type=28,
                timer=None,
                l1_weights_shared=[shared_w1] if fuse_shared else None,
                l2_weights_shared=[shared_w2] if fuse_shared else None,
                l1_weights_shared_sf=None,
                l2_weights_shared_sf=None,
            )
            output.masked_fill_(~valid[:, None], 0)
            return output, counts

        def reference():
            routed, counts = call(False)
            result = routed.clone()
            for index in range(args.shared_experts):
                gate_up = F.linear(x, shared_w1[index])
                activated = torch_npu.npu_swiglu(gate_up, dim=-1)
                result.add_(F.linear(activated, shared_w2[index]))
            result.masked_fill_(~valid[:, None], 0)
            return result, counts.clone()

        for step in range(4):
            expected_counts = update_inputs(step)
            expected, routed_counts = reference()
            output, counts = call(True)
            check(
                output,
                expected,
                counts,
                expected_counts,
                valid,
                f"eager rows={rows} case={step}",
            )
            check(
                expected,
                expected,
                routed_counts,
                expected_counts,
                valid,
                f"routed counts rows={rows} case={step}",
            )

        if args.eager_only:
            return
        update_inputs(0)
        for _ in range(2):
            call(True)
        torch.npu.synchronize()
        dist.barrier(group=group)
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        calls_before_capture = wrapper_calls
        # Match the native capture mode used by SGLang's BreakableCUDAGraph.
        with torch.npu.graph(
            graph, stream=capture_stream, auto_dispatch_capture=True
        ):
            captured_output, captured_counts = call(True)
        assert wrapper_calls == calls_before_capture + 1
        torch.npu.synchronize()
        for step in range(args.replays):
            expected_counts = update_inputs(100 + step)
            expected, _ = reference()
            calls_before_replay = wrapper_calls
            graph.replay()
            torch.npu.synchronize()
            assert wrapper_calls == calls_before_replay, (
                "Graph replay unexpectedly re-entered the Python MegaMoE wrapper"
            )
            check(
                captured_output,
                expected,
                captured_counts,
                expected_counts,
                valid,
                f"graph rows={rows} replay={step}",
            )
        # Graph objects and their tensors die before the shared window is freed.
        torch.npu.synchronize()

    try:
        with torch.inference_mode():
            for rows in args.rows:
                run_bucket(rows)
    except BaseException:
        print(
            f"rank={rank}: failed; skipping collective window cleanup "
            "on the exception path",
            file=sys.stderr,
            flush=True,
        )
        raise
    # Only the normal success path may wait for all peers and free the window.
    torch.npu.synchronize()
    dist.barrier(group=group)
    torch.npu.synchronize()
    symm_buffer.destroy()
    dist.destroy_process_group()
    if rank == 0:
        scope = (
            "EAGER ONLY (graph acceptance NOT RUN)"
            if args.eager_only
            else "EAGER + NPU GRAPH"
        )
        print(f"PASS BF16 MegaMoE shared experts: {scope}", flush=True)


if __name__ == "__main__":
    main()
