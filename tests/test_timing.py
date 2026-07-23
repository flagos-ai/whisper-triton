import numpy as np
import pytest
import scipy.ndimage
import torch
import torch.nn.functional as F

from whisper.timing import dtw, dtw_cpu, dtw_cuda, median_filter

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

    for filter_width in [3, 5, 7, 13]:
        filtered = median_filter(x, filter_width)

        # use NumPy reflect padding because SciPy's edge behavior is different
        pad_width = filter_width // 2
        padded_x = np.pad(
            x, [(0, 0)] * (x.ndim - 1) + [(pad_width, pad_width)], mode="reflect"
        )
        scipy_filtered = scipy.ndimage.median_filter(
            padded_x, [1] * (x.ndim - 1) + [filter_width]
        )
        scipy_filtered = scipy_filtered[..., pad_width:-pad_width]

        assert np.allclose(filtered, scipy_filtered)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("shape", shapes)
def test_median_filter_accelerator_equivalence(shape, accelerator_device: torch.device):
    x = torch.randn(*shape)

    for filter_width in [3, 5, 7, 13]:
        filtered_cpu = median_filter(x, filter_width)
        filtered_accelerator = median_filter(
            x.to(accelerator_device), filter_width
        ).cpu()

        assert np.allclose(filtered_cpu, filtered_accelerator)


@pytest.mark.requires_cuda
@pytest.mark.requires_accelerator
def test_median_filter_accelerator_kernel_equivalence(
    accelerator_device: torch.device,
):
    """Call the Triton kernel directly so fallback cannot mask a failure."""
    from whisper.triton_ops import median_filter_cuda

    x = torch.randn(4, 5, 345)
    for filter_width in [3, 5, 7, 13]:
        expected = median_filter(x, filter_width)
        pad_width = filter_width // 2
        padded = F.pad(x, (pad_width, pad_width, 0, 0), mode="reflect")
        actual = median_filter_cuda(padded.to(accelerator_device), filter_width).cpu()
        assert np.allclose(expected, actual)


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
