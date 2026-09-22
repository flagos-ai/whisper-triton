#!/usr/bin/env python3
"""Run the Whisper test suite and archive a portable CNPort result bundle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pytest_counts(junit_path: Path) -> Dict[str, int]:
    empty = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    if not junit_path.exists():
        return empty
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failed = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    return {
        "collected": tests,
        "passed": max(0, tests - failed - errors - skipped),
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def stream_command(command: List[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def write_report(path: Path, summary: Dict[str, Any]) -> None:
    counts = summary["pytest"]
    models = summary.get("models", [])
    lines = [
        "# Whisper CNPort platform test",
        "",
        f"- Platform: `{summary['platform']}`",
        f"- Status: **{summary['status']}**",
        f"- Started: `{summary['started_at']}`",
        f"- Finished: `{summary['finished_at']}`",
        f"- Return code: `{summary['returncode']}`",
        "",
        "## Pytest",
        "",
        "| Collected | Passed | Failed | Skipped | Errors |",
        "|---:|---:|---:|---:|---:|",
        f"| {counts['collected']} | {counts['passed']} | {counts['failed']} | "
        f"{counts['skipped']} | {counts['errors']} |",
        "",
        f"Models declared by this release ({len(models)}): "
        + (", ".join(f"`{m}`" for m in models) if models else "not collected"),
        "",
        "## Artifacts",
        "",
        "- `environment.json`",
        "- `junit.xml`",
        "- `pytest.log`",
        "- `summary.json`",
        "- `command.json`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", required=True, help="platform key recorded in the result"
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "tests" / "results"
    )
    parser.add_argument(
        "--test-paths",
        nargs="+",
        default=["tests/"],
        help="test paths to pass to pytest (default: tests/)",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="override the timestamped result directory"
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="extra pytest arguments after --"
    )
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
    output_dir = args.output_dir or args.output_root / args.platform / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    environment_path = output_dir / "environment.json"
    junit_path = output_dir / "junit.xml"
    log_path = output_dir / "pytest.log"
    started_at = now()

    env_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "check_accelerator_env.py"),
        "--platform",
        args.platform,
        "--json",
        str(environment_path),
    ]
    env_result = subprocess.run(env_command, cwd=REPO_ROOT, text=True)

    extra_args = list(args.pytest_args)
    if extra_args and extra_args[0] == "--":
        extra_args.pop(0)
    pytest_command = [
        sys.executable,
        "-m",
        "pytest",
        *args.test_paths,
        "-v",
        f"--junitxml={junit_path}",
        *extra_args,
    ]
    write_json(
        output_dir / "command.json",
        {
            "schema_version": 1,
            "environment_check": [
                "current-python",
                "scripts/check_accelerator_env.py",
                "--platform",
                args.platform,
            ],
            "pytest": [
                "current-python",
                "-m",
                "pytest",
                *args.test_paths,
                "-v",
                "--junitxml=<result>/junit.xml",
                *extra_args,
            ],
        },
    )

    returncode = env_result.returncode
    if returncode == 0:
        returncode = stream_command(pytest_command, log_path)
    else:
        log_path.write_text(
            "pytest was not run because environment validation failed\n",
            encoding="utf-8",
        )

    try:
        sys.path.insert(0, str(REPO_ROOT))
        import whisper

        models = whisper.available_models()
    except Exception:
        models = []

    counts = pytest_counts(junit_path)
    summary = {
        "schema_version": 1,
        "platform": args.platform,
        "status": (
            "passed"
            if returncode == 0
            else ("error" if env_result.returncode else "failed")
        ),
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": now(),
        "models": models,
        "pytest": counts,
        "artifacts": {
            "environment": "environment.json",
            "junit": "junit.xml",
            "log": "pytest.log",
            "report": "report.md",
            "command": "command.json",
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    print(f"Result directory: {output_dir}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
