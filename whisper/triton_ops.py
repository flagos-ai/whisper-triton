import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise RuntimeError(
        "triton import failed; install the Triton distribution supplied for "
        "the current accelerator and PyTorch environment"
    ) from exc


@triton.jit
def dtw_kernel(
    cost, trace, x, x_stride, cost_stride, trace_stride, N, M, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    for k in range(1, N + M + 1):  # k = i + j
        tl.debug_barrier()

        p0 = cost + (k - 1) * cost_stride
        p1 = cost + k * cost_stride
        p2 = cost + k * cost_stride + 1

        c0 = tl.load(p0 + offsets, mask=mask)
        c1 = tl.load(p1 + offsets, mask=mask)
        c2 = tl.load(p2 + offsets, mask=mask)

        x_row = tl.load(x + (k - 1) * x_stride + offsets, mask=mask, other=0)
        cost_row = x_row + tl.minimum(tl.minimum(c0, c1), c2)

        cost_ptr = cost + (k + 1) * cost_stride + 1
        tl.store(cost_ptr + offsets, cost_row, mask=mask)

        trace_ptr = trace + (k + 1) * trace_stride + 1
        tl.store(trace_ptr + offsets, 2, mask=mask & (c2 <= c0) & (c2 <= c1))
        tl.store(trace_ptr + offsets, 1, mask=mask & (c1 <= c0) & (c1 <= c2))
        tl.store(trace_ptr + offsets, 0, mask=mask & (c0 <= c1) & (c0 <= c2))


@triton.jit
def median_kernel(
    y,
    x,
    x_stride,
    y_stride,
    FILTER_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute a sliding-window median along the last tensor dimension."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < y_stride

    x_ptr = x + row_idx * x_stride
    y_ptr = y + row_idx * y_stride
    middle = FILTER_WIDTH // 2

    # FILTER_WIDTH is a compile-time constant, so both loops are fully unrolled.
    # A value is the median when no more than `middle` values are smaller than it
    # and more than `middle` values are less than or equal to it.  Using ranks
    # avoids runtime source rewriting and also handles duplicate values.
    median = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    for candidate_idx in tl.static_range(0, FILTER_WIDTH):
        candidate = tl.load(x_ptr + offsets + candidate_idx, mask=mask, other=0.0)
        num_smaller = tl.zeros((BLOCK_SIZE,), tl.int32)
        num_not_larger = tl.zeros((BLOCK_SIZE,), tl.int32)

        for value_idx in tl.static_range(0, FILTER_WIDTH):
            value = tl.load(x_ptr + offsets + value_idx, mask=mask, other=0.0)
            num_smaller += tl.where(value < candidate, 1, 0)
            num_not_larger += tl.where(value <= candidate, 1, 0)

        is_median = (num_smaller <= middle) & (num_not_larger > middle)
        median = tl.where(is_median, candidate, median)

    tl.store(y_ptr + offsets, median, mask=mask)


@triton.jit
def std_mean_kernel(std, mean, x, x_stride, M, BLOCK_SIZE: tl.constexpr):
    """Compute the biased standard deviation and mean of the last dimension."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < M

    x_ptr = x + row_idx * x_stride
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_val = tl.sum(values, axis=0) / M
    centered = tl.where(mask, values - mean_val, 0.0)
    var_val = tl.sum(centered * centered, axis=0) / M

    tl.store(std + row_idx, tl.sqrt(var_val).to(std.dtype.element_ty))
    tl.store(mean + row_idx, mean_val.to(mean.dtype.element_ty))


def std_mean_cuda(x: torch.Tensor):
    """Compute the biased std and mean along dim=-2 of x, keeping the dimension

    Equivalent to ``torch.std_mean(x, dim=-2, keepdim=True, unbiased=False)``,
    which some vendor PyTorch builds do not implement.
    """
    x = x.transpose(-1, -2).contiguous()
    row_count = x.numel() // x.shape[-1]
    M = x.shape[-1]
    std = x.new_empty((*x.shape[:-1], 1))
    mean = x.new_empty((*x.shape[:-1], 1))

    BLOCK_SIZE = 1 << (M - 1).bit_length()
    std_mean_kernel[(row_count,)](
        std,
        mean,
        x,
        x.shape[-1],
        M,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return std.transpose(-1, -2), mean.transpose(-1, -2)


def median_filter_cuda(x: torch.Tensor, filter_width: int):
    """Apply a median filter of given width along the last dimension of x"""
    x = x.contiguous()
    output_width = x.shape[-1] - filter_width + 1
    row_count = x.numel() // x.shape[-1]
    y = torch.empty((*x.shape[:-1], output_width), dtype=x.dtype, device=x.device)

    BLOCK_SIZE = 1 << (output_width - 1).bit_length()
    median_kernel[(row_count,)](
        y,
        x,
        x.shape[-1],
        output_width,
        FILTER_WIDTH=filter_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y
