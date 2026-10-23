import triton
import triton.language as tl


@triton.jit
def pack_kernel(
    Out,
    In,
    Counts,
    size: tl.constexpr,
    input_rows: tl.constexpr,
    width: tl.constexpr,
    dp_rank: tl.constexpr,
    offset: tl.constexpr,
    contribute: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // width - offset
    live = tl.load(Counts + dp_rank)
    valid = (idx < size) & (row >= 0) & (row < input_rows) & (row < live) & contribute
    value = tl.load(In + row * width + idx % width, mask=valid, other=0)
    tl.store(Out + idx, value, mask=idx < size)
