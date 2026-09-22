import numpy as np
import pytest
import scipy.ndimage
import torch
import torch.nn.functional as F

from whisper.timing import dtw, dtw_cpu, dtw_cuda, median_filter, std_mean

sizes = [
    (10, 20),
    (32, 16),
    (123, 1500),
    (234, 189),
]
shapes = [
    (10,),
    (1, 15),
    (4, 5, 345),
    (6, 12, 240, 512),
]
std_mean_shapes = [
    (3, 11, 1500),
    (5, 9, 700),
]
filter_widths = [1, 3, 5, 7, 13]


def reflect_pad_last_dimension(x: torch.Tensor, pad_width: int):
    if pad_width == 0:
        return x

    input_shape = x.shape
    flattened = x.reshape(-1, input_shape[-1])
    padded = F.pad(flattened, (pad_width, pad_width), mode="reflect")
    return padded.reshape(*input_shape[:-1], padded.shape[-1])


@pytest.mark.parametrize("N, M", sizes)
def test_dtw(N: int, M: int):
    steps = np.concatenate([np.zeros(N - 1), np.ones(M - 1)])
    np.random.shuffle(steps)
    x = np.random.random((N, M)).astype(np.float32)

    i, j, k = 0, 0, 0
    trace = []
    while True:
        x[i, j] -= 1
        trace.append((i, j))

        if k == len(steps):
            break

        if k + 1 < len(steps) and steps[k] != steps[k + 1]:
            i += 1
            j += 1
            k += 2
            continue

        if steps[k] == 0:
            i += 1
        if steps[k] == 1:
            j += 1
        k += 1

    trace = np.array(trace).T
    dtw_trace = dtw_cpu(x)

    assert np.allclose(trace, dtw_trace)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("N, M", sizes)
def test_dtw_accelerator_equivalence(N: int, M: int, accelerator_device: torch.device):
    x_numpy = np.random.randn(N, M).astype(np.float32)
    x_accelerator = torch.from_numpy(x_numpy).to(accelerator_device)

    trace_cpu = dtw_cpu(x_numpy)
    trace_accelerator = dtw_cuda(x_accelerator)

    assert np.allclose(trace_cpu, trace_accelerator)


@pytest.mark.parametrize("shape", shapes)
def test_median_filter(shape):
    x = torch.randn(*shape)

    for filter_width in filter_widths:
        filtered = median_filter(x, filter_width)

        # use NumPy reflect padding because SciPy's edge behavior is different
        pad_width = filter_width // 2
        padded_x = np.pad(
            x, [(0, 0)] * (x.ndim - 1) + [(pad_width, pad_width)], mode="reflect"
        )
        scipy_filtered = scipy.ndimage.median_filter(
            padded_x, [1] * (x.ndim - 1) + [filter_width]
        )
        if pad_width > 0:
            scipy_filtered = scipy_filtered[..., pad_width:-pad_width]

        assert np.allclose(filtered, scipy_filtered)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("shape", shapes)
def test_median_filter_accelerator_equivalence(shape, accelerator_device: torch.device):
    x = torch.randn(*shape)

    for filter_width in filter_widths:
        filtered_cpu = median_filter(x, filter_width)
        filtered_accelerator = median_filter(
            x.to(accelerator_device), filter_width
        ).cpu()

        assert np.allclose(filtered_cpu, filtered_accelerator)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("shape", shapes)
def test_median_filter_accelerator_kernel_equivalence(
    shape, accelerator_device: torch.device
):
    """Call the Triton kernel directly so fallback cannot mask a failure."""
    from whisper.triton_ops import median_filter_cuda

    x = torch.randn(*shape)
    for filter_width in filter_widths:
        expected = median_filter(x, filter_width)
        pad_width = filter_width // 2
        padded = reflect_pad_last_dimension(x, pad_width)
        actual = median_filter_cuda(padded.to(accelerator_device), filter_width).cpu()
        assert np.allclose(expected, actual)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
def test_median_filter_accelerator_kernel_duplicate_values(
    accelerator_device: torch.device,
):
    """Repeated values must still have a well-defined median rank."""
    from whisper.triton_ops import median_filter_cuda

    x = torch.tensor([[3.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 0.0]])
    for filter_width in filter_widths:
        expected = median_filter(x, filter_width)
        pad_width = filter_width // 2
        padded = reflect_pad_last_dimension(x, pad_width)
        actual = median_filter_cuda(padded.to(accelerator_device), filter_width).cpu()
        assert torch.equal(expected, actual)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
def test_timing_falls_back_to_cpu(monkeypatch, accelerator_device: torch.device):
    import whisper.timing as timing
    import whisper.triton_ops as triton_ops

    def compile_failure(*args, **kwargs):
        raise RuntimeError("simulated compilation failure")

    monkeypatch.setattr(triton_ops, "median_filter_cuda", compile_failure)
    monkeypatch.setattr(timing, "dtw_cuda", compile_failure)

    median_input = torch.randn(4, 5, 32, device=accelerator_device)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        median_result = median_filter(median_input, 7)
    assert median_result.device.type == "cpu"

    dtw_input = torch.randn(10, 20, device=accelerator_device)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        dtw_result = dtw(dtw_input)
    assert np.allclose(dtw_result, dtw_cpu(dtw_input.cpu().numpy()))


@pytest.mark.parametrize("shape", std_mean_shapes)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_std_mean(shape, dtype):
    x = torch.randn(*shape, dtype=dtype)

    actual_std, actual_mean = std_mean(x, dim=-2)
    expected_std, expected_mean = torch.std_mean(
        x.double(), dim=-2, keepdim=True, unbiased=False
    )

    assert actual_std.shape == expected_std.shape
    assert actual_mean.shape == expected_mean.shape
    assert torch.allclose(actual_std.double(), expected_std, atol=1e-3)
    assert torch.allclose(actual_mean.double(), expected_mean, atol=1e-3)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("shape", std_mean_shapes)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_std_mean_accelerator_equivalence(shape, dtype, accelerator_device):
    x = torch.randn(*shape, dtype=dtype)
    expected_std, expected_mean = torch.std_mean(
        x.double(), dim=-2, keepdim=True, unbiased=False
    )

    actual_std, actual_mean = std_mean(x.to(accelerator_device), dim=-2)

    # compare against the double reference; fp16 kernels may differ from the
    # CPU implementation in the last mantissa bit, which is within fp16
    # resolution
    assert torch.allclose(actual_std.cpu().double(), expected_std, atol=1e-3)
    assert torch.allclose(
        actual_mean.cpu().double(), expected_mean, atol=1e-3, rtol=1e-3
    )


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("shape", std_mean_shapes)
def test_std_mean_accelerator_kernel_equivalence(
    shape, accelerator_device: torch.device
):
    """Call the Triton kernel directly so fallback cannot mask a failure."""
    from whisper.triton_ops import std_mean_cuda

    for dtype in (torch.float32, torch.float16):
        x = torch.randn(*shape, dtype=dtype)
        x64 = x.double()
        expected = torch.std_mean(x64, dim=-2, keepdim=True, unbiased=False)
        actual_std, actual_mean = std_mean_cuda(x.to(accelerator_device))
        assert torch.allclose(
            actual_std.cpu().double(), expected[0], atol=1e-3, rtol=1e-3
        )
        assert torch.allclose(
            actual_mean.cpu().double(), expected[1], atol=1e-3, rtol=1e-3
        )


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
def test_std_mean_falls_back_to_cpu(monkeypatch, accelerator_device):
    import whisper.triton_ops as triton_ops

    def compile_failure(*args, **kwargs):
        raise RuntimeError("simulated compilation failure")

    monkeypatch.setattr(triton_ops, "std_mean_cuda", compile_failure)

    x = torch.randn(3, 11, 700, device=accelerator_device)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        actual_std, actual_mean = std_mean(x, dim=-2)
    expected_std, expected_mean = torch.std_mean(
        x.cpu().double(), dim=-2, keepdim=True, unbiased=False
    )
    assert actual_std.device.type == "cpu"
    assert np.allclose(actual_std.double(), expected_std)
    assert np.allclose(actual_mean.double(), expected_mean)
