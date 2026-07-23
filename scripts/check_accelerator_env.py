#!/usr/bin/env python3
"""Inspect the active Whisper/PyTorch/Triton environment without changing it."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PLATFORM_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "nvidia": "nvidia",
    "cuda": "nvidia",
    "ascend": "ascend",
    "npu": "ascend",
    "cambricon": "cambricon",
    "mlu": "cambricon",
    "musa": "musa",
    "mtt": "musa",
    "hygon": "hygon",
    "dcu": "hygon",
    "rocm": "hygon",
    "t-head": "thead",
    "thead": "thead",
    "ppu": "thead",
}

EXPECTED = {
    "cpu": {"devices": {"cpu"}, "backends": set(), "extension": None},
    "nvidia": {"devices": {"cuda"}, "backends": {"nvidia", "cuda"}, "extension": None},
    "ascend": {
        "devices": {"npu"},
        "backends": {"ascend", "npu"},
        "extension": "torch_npu",
    },
    "cambricon": {
        "devices": {"mlu"},
        "backends": {"mlu", "cambricon"},
        "extension": "torch_mlu",
    },
    "musa": {
        "devices": {"musa"},
        "backends": {"musa", "mtgpu"},
        "extension": "torch_musa",
    },
    "hygon": {
        "devices": {"cuda", "hip"},
        "backends": {"amd", "hip", "rocm"},
        "extension": None,
    },
    "thead": {
        "devices": {"cuda"},
        "backends": {"nvidia", "cuda", "ppu"},
        "extension": None,
    },
}


def canonical_platform(value: str) -> str:
    key = value.strip().lower()
    if key in PLATFORM_ALIASES:
        return PLATFORM_ALIASES[key]
    for alias, canonical in PLATFORM_ALIASES.items():
        if alias != "auto" and alias in key:
            return canonical
    raise ValueError(f"unknown platform label: {value}")


def module_version(module: Any, distribution: Optional[str] = None) -> Optional[str]:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    if distribution:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def module_origin(module: Any) -> str:
    filename = getattr(module, "__file__", None)
    if not filename:
        return "unknown"
    try:
        origin = Path(filename).resolve()
        prefix = Path(sys.prefix).resolve()
        origin.relative_to(prefix)
        return "current-python-environment"
    except (OSError, ValueError):
        return "outside-current-python-prefix"


def import_optional(name: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def detect_device(torch: Any) -> Tuple[str, bool, int, Optional[str]]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count else None
        return "cuda", True, count, name

    get_backend_name = getattr(torch._C, "_get_privateuse1_backend_name", None)
    if get_backend_name is not None:
        backend = get_backend_name()
        if backend != "privateuseone":
            module = getattr(torch, backend, None)
            available = bool(
                module is not None and getattr(module, "is_available", lambda: False)()
            )
            count = (
                int(getattr(module, "device_count", lambda: 0)()) if available else 0
            )
            name = None
            if available and count:
                try:
                    name = str(getattr(module, "get_device_name")(0))
                except Exception:
                    pass
            return backend, available, count, name
    return "cpu", True, 0, platform.processor() or None


def detect_triton_backend(triton: Any) -> Tuple[Optional[str], Optional[str]]:
    try:
        target = triton.runtime.driver.active.get_current_target()
        return str(target.backend), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def check(platform_label: str) -> Dict[str, Any]:
    requested = canonical_platform(platform_label)
    extension_results: Dict[str, Any] = {}
    for name in ("torch_npu", "torch_mlu", "torch_musa"):
        module, error = import_optional(name)
        extension_results[name] = {
            "available": module is not None,
            "version": module_version(module, name) if module else None,
            "origin": module_origin(module) if module else None,
            "error": error,
        }

    torch, torch_error = import_optional("torch")
    whisper, whisper_error = import_optional("whisper")
    triton, triton_error = import_optional("triton")

    checks: List[Dict[str, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "passed" if ok else "failed", "detail": detail}
        )

    add("torch import", torch is not None, torch_error or "available")
    add("whisper import", whisper is not None, whisper_error or "available")
    add(
        "ffmpeg",
        shutil.which("ffmpeg") is not None,
        "available in PATH" if shutil.which("ffmpeg") else "not found in PATH",
    )

    device_type, device_available, device_count, device_name = (
        "unknown",
        False,
        0,
        None,
    )
    if torch is not None:
        device_type, device_available, device_count, device_name = detect_device(torch)

    resolved = requested
    if requested == "auto":
        by_device = {"npu": "ascend", "mlu": "cambricon", "musa": "musa", "cpu": "cpu"}
        if device_type in by_device:
            resolved = by_device[device_type]
        elif getattr(getattr(torch, "version", None), "hip", None):
            resolved = "hygon"
        elif os.getenv("PPU_SDK"):
            resolved = "thead"
        else:
            resolved = "nvidia"

    expected = EXPECTED[resolved]
    add(
        "device available",
        device_available,
        f"device={device_type}, count={device_count}",
    )
    add(
        "device matches platform",
        device_type in expected["devices"],
        f"expected={sorted(expected['devices'])}, actual={device_type}",
    )

    required_extension = expected["extension"]
    if required_extension:
        ext = extension_results[required_extension]
        add(
            f"{required_extension} import",
            ext["available"],
            ext["error"] or "available",
        )

    backend, backend_error = (None, None)
    triton_required = resolved != "cpu"
    add(
        "triton import",
        triton is not None or not triton_required,
        triton_error or ("available" if triton else "optional on CPU"),
    )
    if triton is not None and triton_required:
        backend, backend_error = detect_triton_backend(triton)
        add(
            "triton backend detected",
            backend is not None,
            backend_error or str(backend),
        )
        if backend is not None:
            add(
                "triton backend matches platform",
                backend.lower() in expected["backends"],
                f"expected={sorted(expected['backends'])}, actual={backend}",
            )

    distributions = sorted(
        set(importlib.metadata.packages_distributions().get("triton", []))
    )
    if len(distributions) > 1:
        add(
            "single Triton provider",
            False,
            f"multiple distributions provide triton: {distributions}",
        )

    runtime_versions = {
        "python": platform.python_version(),
        "torch": module_version(torch, "torch") if torch else None,
        "torch_cuda": (
            getattr(getattr(torch, "version", None), "cuda", None) if torch else None
        ),
        "torch_hip": (
            getattr(getattr(torch, "version", None), "hip", None) if torch else None
        ),
        "triton": module_version(triton, "triton") if triton else None,
        "whisper": module_version(whisper, "openai-whisper") if whisper else None,
    }

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_platform": platform_label,
        "resolved_platform": resolved,
        "status": (
            "passed" if all(item["status"] == "passed" for item in checks) else "failed"
        ),
        "system": {"os": platform.system(), "machine": platform.machine()},
        "python": {
            "version": platform.python_version(),
            "executable": "current-python",
        },
        "device": {
            "type": device_type,
            "available": device_available,
            "count": device_count,
            "name": device_name,
        },
        "versions": runtime_versions,
        "extensions": extension_results,
        "triton": {
            "available": triton is not None,
            "backend": backend,
            "origin": module_origin(triton) if triton else None,
            "providers": distributions,
            "error": triton_error,
            "backend_error": backend_error,
        },
        "environment_variables_present": sorted(
            name
            for name in (
                "CUDA_HOME",
                "CUDA_PATH",
                "ROCM_HOME",
                "ROCM_PATH",
                "HIP_PATH",
                "ASCEND_HOME_PATH",
                "TRITON_ASCEND_ARCH",
                "NEUWARE_HOME",
                "MUSA_HOME",
                "MUSA_LIBDEVICE_PATH",
                "PPU_SDK",
                "TRITON_CACHE_DIR",
            )
            if os.getenv(name)
        ),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        default="auto",
        help="auto, cpu, nvidia, ascend, cambricon, musa, hygon, or t-head",
    )
    parser.add_argument("--json", type=Path, help="write a sanitized JSON report")
    args = parser.parse_args()

    try:
        report = check(args.platform)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Whisper CNPort environment: {report['status']}")
    print(f"platform: {report['resolved_platform']}")
    print(f"device: {report['device']['type']} ({report['device']['count']})")
    print(f"torch: {report['versions']['torch'] or 'not installed'}")
    print(f"triton: {report['versions']['triton'] or 'not installed'}")
    print(f"triton backend: {report['triton']['backend'] or 'not detected'}")
    for item in report["checks"]:
        marker = "PASS" if item["status"] == "passed" else "FAIL"
        print(f"[{marker}] {item['name']}: {item['detail']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
