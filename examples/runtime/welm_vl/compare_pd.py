#!/usr/bin/env python3
"""Capture combined/PD chat results with an identical request for CPU-side comparison.

Run once against combined serving, then against the PD router with --reference.
This permits comparison on a single 8-NPU host without running both deployments
at once. Real accuracy tolerances and A-A variation still need model validation.
"""

import argparse
import base64
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path


def compare(reference, actual, max_logprob_diff):
    a, b = reference["choices"][0], actual["choices"][0]
    at, bt = (choice.get("logprobs") or {} for choice in (a, b))
    at, bt = at.get("content") or [], bt.get("content") or []
    coverage = len(at) >= 2 and len(bt) >= 2
    token_identity = lambda token: token.get("bytes") or token.get("token")
    same_tokens = (
        bool(at)
        and len(at) == len(bt)
        and all(token_identity(x) == token_identity(y) for x, y in zip(at, bt))
    )
    diffs = [
        abs(x["logprob"] - y["logprob"])
        for x, y in zip(at, bt)
        if token_identity(x) == token_identity(y)
    ]
    messages_equal = all(
        a["message"].get(key) == b["message"].get(key)
        for key in ("content", "reasoning_content", "tool_calls")
    )
    usage_equal = all(
        reference["usage"].get(key) == actual["usage"].get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    image_tokens = lambda response: (
        response["usage"].get("prompt_tokens_details") or {}
    ).get("image_tokens", 0)
    usage_equal &= image_tokens(reference) == image_tokens(actual)
    max_diff = max(diffs) if diffs else None
    return dict(
        passed=bool(
            coverage
            and same_tokens
            and messages_equal
            and usage_equal
            and max_diff is not None
            and max_diff <= max_logprob_diff
        ),
        observed_handoff_and_first_decode=coverage,
        handoff_token_equal=bool(
            at and bt and token_identity(at[0]) == token_identity(bt[0])
        ),
        first_decode_token_equal=bool(
            coverage and token_identity(at[1]) == token_identity(bt[1])
        ),
        all_tokens_equal=same_tokens,
        messages_equal=messages_equal,
        usage_equal=usage_equal,
        max_matching_token_logprob_diff=max_diff,
        max_logprob_diff_threshold=max_logprob_diff,
        baseline_image_tokens=image_tokens(reference),
        actual_image_tokens=image_tokens(actual),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Worker or router base URL")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--request-json",
        type=Path,
        help="Existing chat request, including 50-image workloads",
    )
    source.add_argument("--image", type=Path, nargs="+")
    parser.add_argument("--model", default="welm-vl")
    parser.add_argument("--prompt", default="请用中文分别描述这些图片中的主要内容。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-logprob-diff", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_logprob_diff < 0:
        parser.error("--max-logprob-diff must be non-negative")
    if args.request_json:
        body = json.loads(args.request_json.read_text())
    else:
        parts = [{"type": "text", "text": args.prompt}]
        for path in args.image:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode()
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                }
            )
        body = dict(
            model=args.model,
            messages=[dict(role="user", content=parts)],
            max_tokens=args.max_tokens,
        )
    body.update(stream=False, n=1, temperature=0, logprobs=True, top_logprobs=1)
    body.setdefault("max_tokens", args.max_tokens)
    encoded = json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference and reference["request_sha256"] != digest:
        parser.error(
            "Reference and current request differ; use identical image bytes, prompt and options"
        )
    request = urllib.request.Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        encoded,
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error
    record = dict(
        url=args.url,
        request_sha256=digest,
        wall_ms=(time.perf_counter() - start) * 1000,
        response=result,
    )
    if reference:
        record["comparison"] = compare(
            reference["response"], result, args.max_logprob_diff
        )
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            record.get("comparison")
            or dict(
                captured=str(args.output),
                request_sha256=digest,
                usage=result.get("usage"),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if reference and not record["comparison"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
