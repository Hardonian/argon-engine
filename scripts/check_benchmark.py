"""Run the release benchmark and enforce a deliberately portable latency ceiling."""

from __future__ import annotations

import json
import subprocess


def main() -> None:
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "-q",
            "-p",
            "argon-cli",
            "--",
            "bench",
            "--medium",
            "--iterations",
            "10000",
            "--warmup",
            "1000",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    p99_ns = int(report["p99_ns"])
    if p99_ns > 1_000_000:
        raise RuntimeError(f"closed-loop p99 regression: {p99_ns}ns exceeds 1ms")
    print(f"Argon closed-loop performance gate passed: p99={p99_ns}ns")


if __name__ == "__main__":
    main()
