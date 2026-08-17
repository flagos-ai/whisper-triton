"""
Environment check script for Whisper NVIDIA platform validation.
Collects system info, GPU/CUDA status, and dependency versions.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime

WHISPER_CACHE = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
os.environ.setdefault("XDG_CACHE_HOME", WHISPER_CACHE)


def run_cmd(cmd):
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "N/A"


def check_system():
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "hostname": run_cmd("hostname"),
    }


def check_gpu():
    smi = run_cmd(
        "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader"
    )
    if smi == "N/A":
        return {"available": False}
    gpus = []
    for line in smi.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_total": parts[2],
                    "compute_capability": parts[3],
                }
            )
    cuda_ver = (
        run_cmd("nvidia-smi --query-gpu=cuda_version --format=csv,noheader")
        .split("\n")[0]
        .strip()
    )
    return {
        "available": True,
        "count": len(gpus),
        "cuda_driver_version": cuda_ver,
        "gpus": gpus,
    }


def check_torch():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        info = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
        }
        if cuda_available:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.get_device_name(0)
        return info
    except ImportError:
        return {"error": "torch not installed"}


def check_triton():
    try:
        import triton

        return {"version": triton.__version__, "available": True}
    except ImportError:
        return {"available": False}


def check_whisper():
    try:
        import whisper

        models_in_cache = []
        cache_dir = os.path.join(WHISPER_CACHE, "whisper")
        if os.path.isdir(cache_dir):
            models_in_cache = [
                f.replace(".pt", "") for f in os.listdir(cache_dir) if f.endswith(".pt")
            ]
        return {
            "version": (
                whisper.__version__ if hasattr(whisper, "__version__") else "N/A"
            ),
            "available_models": whisper.available_models(),
            "cached_models": sorted(models_in_cache),
            "cache_dir": cache_dir,
        }
    except ImportError:
        return {"error": "whisper not installed"}


def check_packages():
    pkgs = [
        "torch",
        "triton",
        "tiktoken",
        "numba",
        "numpy",
        "scipy",
        "torchaudio",
        "jiwer",
        "soundfile",
    ]
    result = {}
    for pkg in pkgs:
        ver = run_cmd(
            f"pip show {pkg} 2>/dev/null | grep ^Version | awk '{{print $2}}'"
        )
        result[pkg] = ver if ver else "not installed"
    return result


def check_whisper_inference():
    """Quick inference smoke test with tiny model."""
    try:
        import torch

        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny", device=device)
        audio_path = os.path.join(os.path.dirname(__file__), "../../tests/jfk.flac")
        audio_path = os.path.normpath(audio_path)
        result = model.transcribe(audio_path, language="en", temperature=0.0)
        text = result["text"].strip().lower()
        passed = "my fellow americans" in text and "your country" in text
        return {
            "status": "PASSED" if passed else "FAILED",
            "device": device,
            "model": "tiny",
            "transcription_snippet": result["text"].strip()[:80],
        }
    except Exception as e:
        return {"status": "ERROR", "detail": str(e)}


def main():
    print("=" * 60)
    print("Whisper NVIDIA Platform — Environment Check")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    sections = {
        "system": check_system(),
        "gpu": check_gpu(),
        "torch": check_torch(),
        "triton": check_triton(),
        "whisper": check_whisper(),
        "packages": check_packages(),
        "smoke_test": check_whisper_inference(),
    }

    for name, data in sections.items():
        print(f"\n[{name.upper()}]")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    print(f"  {k}: {', '.join(str(i) for i in v)}")
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f"  {k}.{kk}: {vv}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {data}")

    print("\n" + "=" * 60)

    # Save JSON for report
    out_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "env_check.json")
    with open(out_path, "w") as f:
        json.dump(sections, f, indent=2)
    print(f"Results saved to: {out_path}")

    return sections


if __name__ == "__main__":
    main()
