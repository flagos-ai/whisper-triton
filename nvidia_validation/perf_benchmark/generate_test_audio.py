"""
Generate fixed-length test audio clips for Whisper performance benchmarking.
Tiles the jfk.flac test audio to produce 30s, 60s, and 300s clips.

Output: /tmp/whisper_perf_audio/{30s,60s,300s}.flac
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
SOURCE_AUDIO = os.path.join(REPO_ROOT, "tests/jfk.flac")
OUTPUT_DIR = "/tmp/whisper_perf_audio"
DURATIONS = [30, 60, 300]


def get_audio_duration(path: str) -> float:
    result = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        text=True
    )
    return float(result.strip())


def generate_audio(source: str, duration_s: int, output_path: str):
    """Tile source audio to fill target duration using ffmpeg."""
    src_dur = get_audio_duration(source)
    # Number of loops needed (ceiling)
    loops = int(duration_s / src_dur) + 2

    # Build filter: concatenate source `loops` times, then trim to target duration
    concat_inputs = "".join([f"[0:a]" for _ in range(loops)])
    filter_complex = f"{concat_inputs}concat=n={loops}:v=0:a=1[out]"

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", str(loops - 1), "-i", source,
        "-t", str(duration_s),
        "-ar", "16000",   # resample to 16kHz (whisper native rate)
        "-ac", "1",        # mono
        output_path
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src_dur = get_audio_duration(SOURCE_AUDIO)
    print(f"Source audio: {SOURCE_AUDIO} ({src_dur:.1f}s)")
    print(f"Output directory: {OUTPUT_DIR}")

    generated = []
    for dur in DURATIONS:
        out_path = os.path.join(OUTPUT_DIR, f"{dur}s.flac")
        if os.path.exists(out_path):
            actual = get_audio_duration(out_path)
            print(f"  {dur}s.flac already exists ({actual:.1f}s), skipping")
        else:
            print(f"  Generating {dur}s.flac ...", end="", flush=True)
            generate_audio(SOURCE_AUDIO, dur, out_path)
            actual = get_audio_duration(out_path)
            print(f" done ({actual:.1f}s)")
        generated.append({"duration_s": dur, "path": out_path})

    print("\nGenerated audio files:")
    for g in generated:
        size_mb = os.path.getsize(g["path"]) / 1024 / 1024
        print(f"  {g['path']}  ({g['duration_s']}s, {size_mb:.1f} MB)")

    return generated


if __name__ == "__main__":
    main()
