"""950PR BF16 prefill A/B: native FlashAttn versus the local FIA adapter.

Run from the repository root in the server's Ascend environment:
    python test/manual/ascend/bench_welm_fia_prefill.py --window -1
    python test/manual/ascend/bench_welm_fia_prefill.py --window 511
Repeat with --no-sink to isolate sink correction overhead. This is a synthetic
single-request operator benchmark, not a model accuracy or throughput test.
"""

import argparse
import importlib.util
import math
import time
from pathlib import Path
from types import SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--chunk-size", type=int, default=16512)
    parser.add_argument("--prefix-lengths", default="0,16512")
    parser.add_argument("--q-heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=-1, help="Left history; -1=full")
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--fia-layout", choices=("TND", "BSND"), default="TND")
    parser.add_argument("--softmax-scale", type=float, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        args.prefix_lengths = [int(value) for value in args.prefix_lengths.split(",")]
    except ValueError:
        parser.error("--prefix-lengths must be comma-separated integers")
    positive = (
        args.chunk_size, args.q_heads, args.kv_heads,
        args.head_dim, args.page_size, args.repeat,
    )
    if any(value <= 0 for value in positive):
        parser.error("Sizes, head counts and --repeat must be positive")
    if any(value < 0 for value in args.prefix_lengths) or args.warmup < 0:
        parser.error("Prefix lengths and --warmup must be nonnegative")
    if args.q_heads % args.kv_heads or args.window < -1:
        parser.error("Q heads must be divisible by KV heads; window must be >= -1")
    if args.softmax_scale is not None and args.softmax_scale <= 0:
        parser.error("--softmax-scale must be positive")
    if args.device.isdecimal():
        args.device = f"npu:{args.device}"
    if not args.device.startswith("npu:") or not args.device[4:].isdecimal():
        parser.error("--device must be a nonnegative index or npu:<index>")
    return args


def load_adapter():
    path = Path(__file__).resolve().parents[3] / (
        "python/sglang/srt/hardware_backend/npu/attention/welm_fia_prefill.py"
    )
    spec = importlib.util.spec_from_file_location("welm_fia_benchmark_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.forward_fia_prefill


def error_metrics(torch, label, actual, expected):
    actual, expected = actual.float().reshape(-1), expected.float().reshape(-1)
    nonfinite = int((~torch.isfinite(actual)).sum())
    delta = actual - expected
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    relative = delta.norm() / expected.norm().clamp_min(1e-30)
    print(
        f"  {label}: max_abs={delta.abs().max():.6g} "
        f"mean_abs={delta.abs().mean():.6g} rel_l2={relative:.6g} "
        f"cosine={cosine:.9f} nonfinite={nonfinite}",
        flush=True,
    )


def reference_rows(torch, args, prefix):
    # Prioritize causal/window/page boundaries, then spread remaining samples.
    rows = {0, args.chunk_size - 1}
    for boundary in (args.page_size, args.window - prefix, args.window + 1 - prefix):
        rows.update(
            i for i in (boundary - 1, boundary, boundary + 1)
            if 0 <= i < args.chunk_size
        )
    for value in torch.linspace(0, args.chunk_size - 1, 16).long().tolist():
        if len(rows) >= min(16, args.chunk_size):
            break
        rows.add(value)
    return sorted(rows)


def reference(torch, query, key, value, sinks, rows, prefix, args, scale):
    # FP32 CPU reference using the BF16-rounded inputs, with an explicit sink
    # logit and zero-valued sink. Only selected Q rows materialize score matrices.
    q = query[rows].cpu().float()
    length = prefix + args.chunk_size
    heads = torch.arange(args.q_heads) // (args.q_heads // args.kv_heads)
    k = key.reshape(-1, args.kv_heads, args.head_dim)[:length].cpu().float()
    v = value.reshape(-1, args.kv_heads, args.head_dim)[:length].cpu().float()
    scores = torch.einsum("qhd,khd->qhk", q, k[:, heads]) * scale
    positions = torch.tensor(rows)[:, None] + prefix
    kv_positions = torch.arange(length)[None, :]
    valid = kv_positions <= positions
    if args.window >= 0:
        valid &= kv_positions >= positions - args.window
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    if sinks is not None:
        sink = sinks.float().cpu()[None, :, None].expand(len(rows), -1, 1)
        scores = torch.cat((scores, sink), dim=-1)
    probabilities = scores.softmax(dim=-1)[..., :length]
    return torch.einsum("qhk,khd->qhd", probabilities, v[:, heads])


def main():
    args = parse_args()
    # Keep --help usable on a host without Ascend/PyTorch dependencies.
    import torch
    import torch_npu
    import cann_ops_transformer as flash

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    adapter = load_adapter()
    scale = args.softmax_scale or args.head_dim**-0.5
    layer = SimpleNamespace(
        tp_q_head_num=args.q_heads, tp_k_head_num=args.kv_heads,
        tp_v_head_num=args.kv_heads, qk_head_dim=args.head_dim,
        v_head_dim=args.head_dim, sliding_window_size=args.window, scaling=scale,
    )
    mask = torch.ones((2048, 2048), dtype=torch.bool).triu_(1).to(args.device)
    flash_mask = mask.to(torch.int8)
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} args={args}")
    print("Times: synchronized wall mean with Python launch overhead; FIA includes")
    print("its adapter and sink LSE correction. Flash schedule is generated beforehand")
    print("and excluded (reusable across matching layers within one network forward).")
    print("Errors are measurements, not an automatic accuracy pass/fail threshold.")
    with torch.inference_mode():
        for prefix in args.prefix_lengths:
            length = prefix + args.chunk_size
            blocks = (length + args.page_size - 1) // args.page_size
            query = torch.randn(
                args.chunk_size, args.q_heads, args.head_dim,
                dtype=torch.bfloat16, device=args.device,
            )
            shape = (blocks, args.page_size, args.kv_heads, args.head_dim)
            key = torch.randn(shape, dtype=query.dtype, device=args.device)
            value = torch.randn_like(key)
            table = torch.arange(blocks, device=args.device, dtype=torch.int32)[None]
            # Vary head sink logits around log(K) so the zero-value sink makes
            # a visible contribution, including for long contexts.
            sinks = (
                None if args.no_sink else torch.linspace(
                    math.log(length) - 2, math.log(length) + 2,
                    args.q_heads, dtype=torch.float32, device=args.device,
                )
            )
            q_cpu, kv_cpu = torch.tensor([args.chunk_size]), torch.tensor([length])
            common = dict(
                cu_seqlens_q=torch.tensor(
                    [0, args.chunk_size], dtype=torch.int32, device=args.device
                ),
                cu_seqlens_kv=None,
                seqused_q=q_cpu.to(device=args.device, dtype=torch.int32),
                seqused_kv=kv_cpu.to(device=args.device, dtype=torch.int32),
                max_seqlen_q=args.chunk_size, max_seqlen_kv=-1,
                mask_mode=4 if args.window >= 0 else 3,
                win_left=args.window, win_right=0 if args.window >= 0 else -1,
                layout_q="TND", layout_kv="PA_BBND", layout_out="TND",
            )
            schedule = flash.flash_attn_metadata(
                args.q_heads, args.kv_heads, args.head_dim, **common
            )

            def run_flash():
                return flash.flash_attn(
                    query, key, value, block_table=table, sinks=sinks,
                    attn_mask=flash_mask, metadata=schedule, softmax_scale=scale,
                    return_softmax_lse=False, **common,
                )[0]

            def run_fia():
                return adapter(
                    query, key, value, layer=layer, page_size=args.page_size,
                    block_table=table, q_lens=q_cpu, kv_lens=kv_cpu,
                    atten_mask=mask, fia=torch_npu.npu_fused_infer_attention_score_v2,
                    use_tnd=args.fia_layout == "TND", sinks=sinks,
                ).reshape_as(query)

            print(
                f"\nQ={args.chunk_size} KV={length} "
                f"prefix={prefix} window={args.window}"
            )
            outputs = {}
            for name, function in (("Flash", run_flash), ("FIA", run_fia)):
                for _ in range(args.warmup):
                    function()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(args.repeat):
                    output = function()
                torch.npu.synchronize()
                elapsed = (time.perf_counter() - start) * 1000 / args.repeat
                print(f"  {name}: {elapsed:.4f} ms/call", flush=True)
                outputs[name] = output.float().cpu()
            error_metrics(torch, "FIA vs Flash", outputs["FIA"], outputs["Flash"])
            rows = reference_rows(torch, args, prefix)
            expected = reference(
                torch, query, key, value, sinks, rows, prefix, args, scale
            )
            print(f"  FP32 reference Q rows: {rows}")
            for name in outputs:
                error_metrics(torch, f"{name} vs FP32", outputs[name][rows], expected)


if __name__ == "__main__":
    main()
