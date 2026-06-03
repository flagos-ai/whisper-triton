import random as rand
import importlib

import numpy
import pytest
import torch


def _accelerator_device():
    for package in ["torch_npu", "torch_mlu", "torch_musa"]:
        try:
            importlib.import_module(package)
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda")

    get_privateuse1 = getattr(torch._C, "_get_privateuse1_backend_name", None)
    if get_privateuse1 is None:
        return None

    backend = get_privateuse1()
    if backend == "privateuseone":
        return None

    module = getattr(torch, backend, None)
    if module is None or not getattr(module, "is_available", lambda: False)():
        return None

    return torch.device(backend)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_accelerator: requires a GPU or accelerator"
    )
    config.addinivalue_line("markers", "requires_cuda: alias for requires_accelerator")


def pytest_collection_modifyitems(config, items):
    if _accelerator_device() is not None:
        return

    skip_marker = pytest.mark.skip(reason="no accelerator available")
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
        pytest.skip("no accelerator available")

    return device
