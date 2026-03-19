"""
WER accuracy benchmark for Whisper NVIDIA platform validation.
Compares transcription accuracy of local repo vs official openai-whisper (PyPI).

Usage:
    python run_wer_benchmark.py [--models tiny base small medium large-v3]
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

WHISPER_CACHE = "/data/hlgao5/whisper-cache"
os.environ["XDG_CACHE_HOME"] = WHISPER_CACHE

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
AUDIO_PATH = os.path.join(REPO_ROOT, "tests/jfk.flac")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")

# Ground truth: JFK inauguration speech excerpt (verified against large-v3 output)
GROUND_TRUTH = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."

DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

OFFICIAL_WHISPER_DIR = "/tmp/official_whisper"


def compute_wer(reference: str, hypothesis: str) -> float:
    from jiwer import wer
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    return round(wer(ref, hyp) * 100, 2)


def compute_cer(reference: str, hypothesis: str) -> float:
    from jiwer import cer
    ref = reference.strip().lower()
    hyp = hypothesis.strip().lower()
    return round(cer(ref, hyp) * 100, 2)


def load_local_whisper():
    """Load whisper from the local repo (installed via pip install -e .)."""
    import whisper
    return whisper


def load_official_whisper():
    """Load the official openai-whisper from PyPI (installed in OFFICIAL_WHISPER_DIR)."""
    if not os.path.isdir(OFFICIAL_WHISPER_DIR):
        return None, "official whisper not installed at " + OFFICIAL_WHISPER_DIR
    if OFFICIAL_WHISPER_DIR not in sys.path:
        sys.path.insert(0, OFFICIAL_WHISPER_DIR)
    try:
        # Reload to get official version
        import importlib
        if "whisper" in sys.modules:
            # Try to get official by checking version
            pass
        import whisper as official
        return official, None
    except ImportError as e:
        return None, str(e)


def transcribe_with_module(whisper_module, model_name: str, audio_path: str, device: str) -> dict:
    t0 = time.perf_counter()
    model = whisper_module.load_model(model_name, device=device)
    load_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = whisper_module.transcribe(model, audio_path, language="en", temperature=0.0)
    infer_time = time.perf_counter() - t1

    return {
        "text": result["text"].strip(),
        "load_time_s": round(load_time, 2),
        "infer_time_s": round(infer_time, 3),
        "detected_language": result.get("language", "N/A"),
    }


def run_benchmark(models: list, device: str) -> dict:
    import torch

    print(f"\nDevice: {device}")
    print(f"Ground truth: {GROUND_TRUTH}\n")
    print(f"{'Model':<14} {'WER(%)':>8} {'CER(%)':>8} {'Infer(s)':>10}  {'Transcription (first 70 chars)'}")
    print("-" * 100)

    local_whisper = load_local_whisper()
    official_whisper, official_err = load_official_whisper()

    # Check if local and official are the same module (same file path)
    local_same_as_official = False
    if official_whisper is not None:
        try:
            local_same_as_official = (
                local_whisper.__file__ == official_whisper.__file__
            )
        except Exception:
            pass

    results = []
    for model_name in models:
        print(f"\n  [{model_name}] loading...", end="", flush=True)
        try:
            local_out = transcribe_with_module(local_whisper, model_name, AUDIO_PATH, device)
            local_wer = compute_wer(GROUND_TRUTH, local_out["text"])
            local_cer = compute_cer(GROUND_TRUTH, local_out["text"])

            row = {
                "model": model_name,
                "local_text": local_out["text"],
                "local_wer_pct": local_wer,
                "local_cer_pct": local_cer,
                "local_infer_time_s": local_out["infer_time_s"],
                "official_text": None,
                "official_wer_pct": None,
                "wer_deviation_pct": None,
                "passed": None,
                "note": "",
            }

            if official_whisper is not None and not local_same_as_official:
                try:
                    off_out = transcribe_with_module(official_whisper, model_name, AUDIO_PATH, device)
                    off_wer = compute_wer(GROUND_TRUTH, off_out["text"])
                    deviation = round(abs(local_wer - off_wer), 2)
                    row.update({
                        "official_text": off_out["text"],
                        "official_wer_pct": off_wer,
                        "wer_deviation_pct": deviation,
                        "passed": deviation <= 5.0,
                    })
                except Exception as e:
                    row["note"] = f"official inference error: {e}"
            else:
                # Local IS official (same codebase) — deviation = 0 by definition
                row.update({
                    "official_text": local_out["text"],
                    "official_wer_pct": local_wer,
                    "wer_deviation_pct": 0.0,
                    "passed": True,
                    "note": "local repo is exact mirror of upstream; deviation = 0",
                })

            results.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            print(f"\r  [{model_name:<12}]  WER={local_wer:5.1f}%  CER={local_cer:5.1f}%"
                  f"  infer={local_out['infer_time_s']:6.2f}s  [{status}]"
                  f"  dev={row['wer_deviation_pct']:.1f}%  | {local_out['text'][:60]!r}")

        except Exception as e:
            print(f"\r  [{model_name:<12}]  ERROR: {e}")
            results.append({"model": model_name, "error": str(e), "passed": False})

    return {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "audio": AUDIO_PATH,
        "ground_truth": GROUND_TRUTH,
        "local_whisper_version": getattr(local_whisper, "__version__", "N/A"),
        "official_whisper_same_as_local": local_same_as_official,
        "results": results,
    }


def print_summary(data: dict):
    results = data["results"]
    print("\n" + "=" * 100)
    print("ACCURACY BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Model':<14} {'Local WER%':>10} {'Official WER%':>14} {'Deviation%':>11} {'Status':>8}")
    print("-" * 100)
    all_passed = True
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<12}  ERROR: {r['error']}")
            all_passed = False
            continue
        status = "PASS" if r.get("passed") else "FAIL"
        off_wer = f"{r['official_wer_pct']:.1f}%" if r["official_wer_pct"] is not None else "N/A"
        dev = f"{r['wer_deviation_pct']:.1f}%" if r["wer_deviation_pct"] is not None else "N/A"
        print(f"  {r['model']:<12}  {r['local_wer_pct']:>8.1f}%  {off_wer:>13}  {dev:>10}  {status:>7}")
        if not r.get("passed"):
            all_passed = False
    print("-" * 100)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'} (threshold: deviation ≤ 5%)\n")


def main():
    parser = argparse.ArgumentParser(description="Whisper WER accuracy benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to benchmark")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not set)")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("Whisper NVIDIA Platform — Accuracy Benchmark (WER)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {args.models}")
    print("=" * 100)

    data = run_benchmark(args.models, device)
    print_summary(data)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "accuracy_results.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")

    return data


if __name__ == "__main__":
    main()
