import importlib
import os
import random as rand

import numpy
import pytest
import torch

_VENDOR_EXTENSIONS = ("torch_npu", "torch_mlu", "torch_musa")


def _load_vendor_extensions():
    for name in _VENDOR_EXTENSIONS:
        try:
            importlib.import_module(name)
        except Exception:
            # A platform environment only needs its own extension.
            pass


def _accelerator_device():
    configured = os.getenv("WHISPER_TEST_DEVICE")
    if configured:
        return torch.device(configured)

    _load_vendor_extensions()
    if torch.cuda.is_available():
        return torch.device("cuda")

    get_backend_name = getattr(torch._C, "_get_privateuse1_backend_name", None)
    if get_backend_name is None:
        return None

    backend = get_backend_name()
    if backend == "privateuseone":
        return None

    module = getattr(torch, backend, None)
    if module is None or not getattr(module, "is_available", lambda: False)():
        return None

    return torch.device(backend)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_accelerator: requires a PyTorch accelerator device"
    )
    config.addinivalue_line(
        "markers", "requires_cuda: upstream-compatible alias for accelerator tests"
    )


def pytest_collection_modifyitems(config, items):
    if _accelerator_device() is not None:
        return

    skip_marker = pytest.mark.skip(reason="no PyTorch accelerator available")
    for item in items:
        if "requires_accelerator" in item.keywords or "requires_cuda" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


@pytest.fixture
def accelerator_device():
    device = _accelerator_device()
    if device is None:
        pytest.skip("no PyTorch accelerator available")
    return device
