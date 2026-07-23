"""
Performance benchmark for Whisper on NVIDIA GPU.
Measures RTF (Real-Time Factor), throughput, and peak GPU memory.

Comparison dimensions:
  - Model size: tiny / base / small / medium / large-v3
  - Precision:  FP32 vs FP16
  - Audio length: 30s / 60s / 300s

Usage:
    python run_perf_benchmark.py [--models tiny base small] [--durations 30 60]
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

WHISPER_CACHE = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
os.environ.setdefault("XDG_CACHE_HOME", WHISPER_CACHE)

AUDIO_DIR = "/tmp/whisper_perf_audio"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")

DEFAULT_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
DEFAULT_DURATIONS = [30, 60, 300]
WARMUP_RUNS = 1
BENCH_RUNS = 3  # Average over N runs


def measure_rtf(model, audio_path: str, audio_duration_s: float, fp16: bool) -> dict:
    import torch
    import whisper

    device = next(model.parameters()).device

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = whisper.transcribe(model, audio_path, language="en", temperature=0.0, fp16=fp16)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(BENCH_RUNS):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = whisper.transcribe(model, audio_path, language="en", temperature=0.0, fp16=fp16)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    peak_mem_mib = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    avg_time = sum(times) / len(times)
    rtf = avg_time / audio_duration_s
    throughput_audio_hrs = (audio_duration_s / 3600) / (avg_time / 3600)  # audio_hrs / gpu_hr

    return {
        "avg_infer_time_s": round(avg_time, 3),
        "rtf": round(rtf, 4),
        "throughput_audio_hrs_per_gpu_hr": round(throughput_audio_hrs, 2),
        "peak_gpu_mem_mib": round(peak_mem_mib, 1),
        "runs": BENCH_RUNS,
    }


def run_benchmark(models: list, durations: list) -> dict:
    import torch
    import whisper

    assert torch.cuda.is_available(), "CUDA is required for performance benchmark"
    device = "cuda"

    results = []
    total = len(models) * len(durations) * 2  # *2 for fp32/fp16
    done = 0

    for model_name in models:
        print(f"\n>>> Model: {model_name}")
        print(f"  {'Precision':<8} {'Duration':>9} {'Infer(s)':>9} {'RTF':>7} {'Throughput(x)':>14} {'GPU Mem(MiB)':>13}")
        print("  " + "-" * 70)

        # Load model once in FP32, reuse for both precisions
        model = whisper.load_model(model_name, device=device)
        model.eval()

        for precision in ["fp32", "fp16"]:
            fp16 = (precision == "fp16")
            for dur in durations:
                audio_path = os.path.join(AUDIO_DIR, f"{dur}s.flac")
                if not os.path.exists(audio_path):
                    print(f"  {precision:<8} {dur:>6}s    SKIP (file not found: {audio_path})")
                    done += 1
                    continue

                try:
                    stats = measure_rtf(model, audio_path, float(dur), fp16)
                    row = {
                        "model": model_name,
                        "precision": precision,
                        "audio_duration_s": dur,
                        **stats,
                    }
                    results.append(row)
                    print(f"  {precision:<8} {dur:>6}s  {stats['avg_infer_time_s']:>9.2f}"
                          f"  {stats['rtf']:>7.4f}  {stats['throughput_audio_hrs_per_gpu_hr']:>14.2f}"
                          f"  {stats['peak_gpu_mem_mib']:>13.1f}")
                except Exception as e:
                    print(f"  {precision:<8} {dur:>6}s    ERROR: {e}")
                    results.append({
                        "model": model_name, "precision": precision,
                        "audio_duration_s": dur, "error": str(e),
                    })
                done += 1

        del model
        torch.cuda.empty_cache()

    return {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0),
        "warmup_runs": WARMUP_RUNS,
        "bench_runs": BENCH_RUNS,
        "results": results,
    }


def print_summary(data: dict):
    results = data["results"]
    print("\n" + "=" * 100)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print(f"GPU: {data['gpu_name']}  |  Averaged over {data['bench_runs']} runs (after {data['warmup_runs']} warmup)")
    print("=" * 100)

    # Group by model
    models = []
    seen = set()
    for r in results:
        if r["model"] not in seen:
            models.append(r["model"])
            seen.add(r["model"])

    for model_name in models:
        print(f"\n  Model: {model_name}")
        print(f"  {'Precision':<8} {'Audio':>7} {'RTF':>8} {'Throughput(x)':>14} {'Mem(MiB)':>10}  FP16 speedup")
        print("  " + "-" * 65)

        model_rows = {(r["precision"], r["audio_duration_s"]): r
                      for r in results if r["model"] == model_name and "error" not in r}

        durations = sorted(set(r["audio_duration_s"] for r in results if r["model"] == model_name))
        for dur in durations:
            fp32 = model_rows.get(("fp32", dur))
            fp16 = model_rows.get(("fp16", dur))
            if fp32:
                speedup = f"{fp32['rtf']/fp16['rtf']:.2f}x" if fp16 else "N/A"
                print(f"  {'fp32':<8} {dur:>4}s  {fp32['rtf']:>8.4f}  {fp32['throughput_audio_hrs_per_gpu_hr']:>14.2f}"
                      f"  {fp32['peak_gpu_mem_mib']:>10.1f}  {speedup}")
            if fp16:
                print(f"  {'fp16':<8} {dur:>4}s  {fp16['rtf']:>8.4f}  {fp16['throughput_audio_hrs_per_gpu_hr']:>14.2f}"
                      f"  {fp16['peak_gpu_mem_mib']:>10.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Whisper GPU performance benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--durations", nargs="+", type=int, default=DEFAULT_DURATIONS)
    args = parser.parse_args()

    print("=" * 100)
    print("Whisper NVIDIA Platform — Performance Benchmark (RTF / Throughput)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {args.models}  |  Audio durations: {args.durations}s")
    print("=" * 100)

    # Generate test audio if needed
    from generate_test_audio import main as gen_audio
    gen_audio()

    data = run_benchmark(args.models, args.durations)
    print_summary(data)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "perf_results.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {out_path}")

    return data


if __name__ == "__main__":
    main()
